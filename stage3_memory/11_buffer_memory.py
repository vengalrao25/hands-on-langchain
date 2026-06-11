"""
Topic 11: Automatic Memory — LangGraph (production) vs RunnableWithMessageHistory
--------------------------------------------------------------------------------
File 10 made you carry the history list by hand. Now we automate it, keyed by a
SESSION/THREAD id so many users each get their own conversation.

IMPORTANT (you're on LangChain 1.x):
  - The PRODUCTION-CURRENT way is LangGraph with a checkpointer. It persists
    state per thread_id automatically and is the path the ecosystem is moving to.
  - RunnableWithMessageHistory still exists and you'll see it in older code /
    tutorials, so you should recognise it — but treat LangGraph as the default
    for anything new.

WHAT THIS FILE COVERS:
  1. LangGraph memory — the modern production approach (thread_id = session)
  2. Trimming inside the graph — keep memory bounded (ties to stage1/10)
  3. RunnableWithMessageHistory — the legacy approach, so you know it
  4. Failure modes — session isolation, unbounded growth, in-memory = volatile
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, trim_messages

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ---------------------------------------------------------------------------
# 1. LANGGRAPH MEMORY — the modern production way
# ---------------------------------------------------------------------------
def demo_langgraph_memory():
    """
    A LangGraph app is a tiny state machine. MessagesState is built-in state
    that holds a 'messages' list and AUTO-APPENDS new messages (no manual list).
    A 'checkpointer' saves that state per thread_id between calls — that's the
    memory. Same thread_id = same conversation; new thread_id = fresh chat.

    MemorySaver keeps state in RAM (great for dev). Swap it for a DB checkpointer
    in production (Topic 12) and memory survives restarts — without touching the
    graph logic.
    """
    from langgraph.graph import StateGraph, START, MessagesState
    from langgraph.checkpoint.memory import MemorySaver

    def call_model(state: MessagesState):
        # state["messages"] already holds the full running conversation.
        response = llm.invoke(state["messages"])
        return {"messages": response}   # auto-appended to state

    workflow = StateGraph(MessagesState)
    workflow.add_node("model", call_model)
    workflow.add_edge(START, "model")
    app = workflow.compile(checkpointer=MemorySaver())

    # thread_id is the "session" — everything under it shares memory.
    cfg = {"configurable": {"thread_id": "vengal-1"}}

    print("=== LangGraph memory ===")
    r1 = app.invoke({"messages": [HumanMessage("Hi, I'm Vengal. I like Python.")]}, cfg)
    print("AI:", r1["messages"][-1].content)
    r2 = app.invoke({"messages": [HumanMessage("What's my name and language?")]}, cfg)
    print("AI:", r2["messages"][-1].content)  # remembers — same thread_id

    # Different thread_id → totally separate memory.
    other = app.invoke({"messages": [HumanMessage("What's my name?")]},
                       {"configurable": {"thread_id": "someone-else"}})
    print("AI (new thread):", other["messages"][-1].content)  # doesn't know
    print()


# ---------------------------------------------------------------------------
# 2. BOUNDED MEMORY — trim inside the graph so it never overflows
# ---------------------------------------------------------------------------
def demo_bounded_memory():
    """
    Memory that never trims eventually overflows the context window (you proved
    this in file 10). Fix: trim the messages INSIDE the node before calling the
    model. The full history is still stored, but the model only ever SEES the
    last N tokens. This is the production pattern for long chats.
    """
    from langgraph.graph import StateGraph, START, MessagesState
    from langgraph.checkpoint.memory import MemorySaver

    def call_model(state: MessagesState):
        trimmed = trim_messages(
            state["messages"],
            max_tokens=80,            # small on purpose
            token_counter=llm,
            strategy="last",          # keep most recent
            include_system=True,
            start_on="human",
        )
        return {"messages": llm.invoke(trimmed)}

    workflow = StateGraph(MessagesState)
    workflow.add_node("model", call_model)
    workflow.add_edge(START, "model")
    app = workflow.compile(checkpointer=MemorySaver())

    cfg = {"configurable": {"thread_id": "trim-demo"}}
    print("=== bounded memory (trim inside the graph) ===")
    app.invoke({"messages": [SystemMessage("You are concise.")]}, cfg)
    for msg in ["I'm Vengal.", "I live in Chennai.", "I love coffee.", "What do I love?"]:
        out = app.invoke({"messages": [HumanMessage(msg)]}, cfg)
        print(f"User: {msg}\nAI  : {out['messages'][-1].content}\n")
    print("(older turns may be trimmed away — the model only sees the recent window)")
    print()


# ---------------------------------------------------------------------------
# 3. LEGACY: RunnableWithMessageHistory (recognise it in old code)
# ---------------------------------------------------------------------------
def demo_runnable_with_history():
    """
    The pre-LangGraph way. You provide a function that returns a history store
    for a given session_id; the wrapper auto-loads/saves it around each call.
    It works, but LangGraph is the recommended path now. Shown so you can READ
    and migrate older codebases.
    """
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.chat_history import InMemoryChatMessageHistory
    from langchain_core.runnables.history import RunnableWithMessageHistory

    store = {}  # session_id -> history (volatile, in RAM)

    def get_session_history(session_id: str):
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are concise."),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ])
    chain = prompt | llm

    with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    cfg = {"configurable": {"session_id": "abc"}}
    print("=== RunnableWithMessageHistory (legacy) ===")
    print(with_history.invoke({"input": "I'm Vengal."}, config=cfg).content)
    print(with_history.invoke({"input": "Who am I?"}, config=cfg).content)
    print()


# ---------------------------------------------------------------------------
# 4. FAILURE MODES (read this)
# ---------------------------------------------------------------------------
def demo_failure_modes():
    """
    - SESSION LEAKAGE: reuse one thread_id/session_id for two users and they read
      each other's conversation. The id MUST be per-user-per-conversation, derived
      from auth — never a global constant.
    - VOLATILE STORE: MemorySaver / InMemoryChatMessageHistory live in RAM. Restart
      the process (or scale to 2 servers) and memory is GONE or not shared. Use a
      DB-backed store in production (Topic 12).
    - UNBOUNDED GROWTH: no trimming → cost + latency climb every turn → eventual
      context overflow. Always trim or summarise (demo 2).
    - CONCURRENCY: the same thread hit by two requests at once can interleave/lose
      messages. Real backends serialise per-thread writes.
    """
    print(demo_failure_modes.__doc__)


if __name__ == "__main__":
    # Run ONE at a time — uncomment as you go.
    # demo_langgraph_memory()
    # demo_bounded_memory()
    demo_runnable_with_history()
    # demo_failure_modes()
