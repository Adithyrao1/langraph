# pip install -U streamlit langgraph langchain-openai python-dotenv

"""
Basic Streamlit Chatbot using LangGraph + DeepSeek
====================================================
- Uses LangGraph's StateGraph with a checkpointer (InMemorySaver) for persistence.
- Each conversation is tied to a "thread_id" -> lets you keep multiple
  separate chat threads, and each thread remembers its own history.
- Streaming response is shown token-by-token in the UI.

Run with:
    streamlit run streamlit_chatbot.py
"""

import os
import uuid

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import MessagesState

load_dotenv()

os.environ["LANGCHAIN_PROJECT"] = "Chatbot using Langgraph"  # optional, but useful for grouping runs
# ---------------------------------------------------------------------------
# 1. Set up the LLM (DeepSeek via OpenAI-compatible API)
# ---------------------------------------------------------------------------
api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    st.error("DEEPSEEK_API_KEY not found in environment. Please set it in your .env file.")
    st.stop()

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=api_key,
    base_url="https://api.deepseek.com",
    streaming=True,  # <-- required for real token-by-token output
)

# ---------------------------------------------------------------------------
# 2. Build the LangGraph workflow
#    MessagesState is a prebuilt state schema with a single "messages" key
#    that automatically appends new messages to a running list.
# ---------------------------------------------------------------------------
def chat_node(state: MessagesState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


graph = StateGraph(MessagesState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

# Checkpointer persists conversation state per thread_id.
# CRITICAL: Streamlit reruns this entire script on every interaction (every
# message, every button click). If we create a new InMemorySaver/workflow
# each time, all prior state is wiped immediately -> history "disappears"
# and threads never persist. So we build it ONCE and stash it in
# st.session_state, which survives across reruns for the same browser session.
# NOTE: InMemorySaver still only persists for the lifetime of the Streamlit
# process itself -- restarting `streamlit run` wipes everything (use
# SqliteSaver/PostgresSaver for durable, cross-restart persistence).
if "workflow" not in st.session_state:
    checkpointer = InMemorySaver()
    st.session_state.workflow = graph.compile(checkpointer=checkpointer)

workflow = st.session_state.workflow

# ---------------------------------------------------------------------------
# 3. Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="DeepSeek Chatbot", page_icon="🤖")
st.title("🤖 DeepSeek Chatbot (LangGraph + Threads)")

# --- Session state: track available threads and the active thread ---
if "threads" not in st.session_state:
    first_thread = str(uuid.uuid4())
    st.session_state.threads = [first_thread]
    st.session_state.active_thread = first_thread

# --- Sidebar: thread management ---
with st.sidebar:
    st.header("Chat Threads")

    if st.button("➕ New Thread"):
        new_thread = str(uuid.uuid4())
        st.session_state.threads.append(new_thread)
        st.session_state.active_thread = new_thread
        st.rerun()

    # Let user pick which thread (conversation) to view/continue
    st.session_state.active_thread = st.radio(
        "Select a thread:",
        options=st.session_state.threads,
        index=st.session_state.threads.index(st.session_state.active_thread),
        format_func=lambda t: f"Thread {t[:8]}",
    )

thread_id = st.session_state.active_thread
config = {"configurable": {"thread_id": thread_id},
          "metadata": {
              "thread_id": st.session_state["thread_id"]     #Adding this will group the traces into threads inside the Langsmith.
          },
          "run_name":"chat_turn"
}

# --- Render existing conversation history for the active thread ---
# get_state() pulls the persisted state for this thread_id from the checkpointer.
state = workflow.get_state(config)
messages = state.values.get("messages", []) if state.values else []

for msg in messages:
    role = "user" if msg.type == "human" else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# --- Chat input ---
user_input = st.chat_input("Type your message...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        # stream_mode="messages" streams LLM tokens as they're generated
        for chunk, metadata in workflow.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
            stream_mode="messages",
        ):
            if chunk.content:
                full_response += chunk.content
                placeholder.markdown(full_response + "▌")

        placeholder.markdown(full_response)