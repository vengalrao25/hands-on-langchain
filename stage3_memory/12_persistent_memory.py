"""
Topic 12: Persistent Memory (survive restarts, scale across servers)
--------------------------------------------------------------------
In-memory stores (MemorySaver, InMemoryChatMessageHistory) vanish on restart and
aren't shared between servers. Any real product needs memory in a DATABASE so:
  - a user's chat is still there after a deploy/restart/crash
  - two API servers behind a load balancer see the SAME conversation

THE ONLY CHANGE from Topic 11: swap the in-memory checkpointer/store for a
DB-backed one. Your graph/chain logic is untouched. That's the whole point of the
abstraction.

DEPENDENCIES (not in the base image — install before running):
    pip install langgraph-checkpoint-sqlite      # for demo 1
    pip install langchain-community              # for demo 2

PROGRESSION: SQLite (one file, zero setup, great to learn / small apps)
             -> Postgres / Redis in real production (same idea, network DB).
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ---------------------------------------------------------------------------
# 1. LANGGRAPH + SQLITE CHECKPOINTER — production memory, persisted to disk
# ---------------------------------------------------------------------------
def demo_sqlite_checkpointer():
    """
    Identical graph to Topic 11, but the checkpointer writes to a SQLite FILE.
    Run this script TWICE: the second run already 'remembers' the first, because
    the state was saved to memory.db on disk. That's persistence.

        pip install langgraph-checkpoint-sqlite
    """
    from langgraph.graph import StateGraph, START, MessagesState
    from langgraph.checkpoint.sqlite import SqliteSaver

    def call_model(state: MessagesState):
        return {"messages": llm.invoke(state["messages"])}

    workflow = StateGraph(MessagesState)
    workflow.add_node("model", call_model)
    workflow.add_edge(START, "model")

    # SqliteSaver is a context manager; it opens the db file.
    with SqliteSaver.from_conn_string("memory.db") as checkpointer:
        app = workflow.compile(checkpointer=checkpointer)
        cfg = {"configurable": {"thread_id": "persistent-user"}}

        print("=== SQLite-backed memory (persists to memory.db) ===")
        out = app.invoke({"messages": [HumanMessage("what question i asked previously ?")]}, cfg)
        print("AI:", out["messages"][-1].content)

        # Ask something that needs prior turns (including from a PREVIOUS run).
        # out2 = app.invoke({"messages": [HumanMessage("What's my favourite language?")]}, cfg)
        # print("AI:", out2["messages"][-1].content)
        print("\nRun this file again — it will still remember, even after restart.")
    print()


# ---------------------------------------------------------------------------
# 2. SQLChatMessageHistory — DB-backed history for RunnableWithMessageHistory
# ---------------------------------------------------------------------------
def demo_sql_message_history():
    """
    If you're using the legacy RunnableWithMessageHistory, its persistent twin is
    SQLChatMessageHistory — same interface, but rows live in a SQL table keyed by
    session_id. Reload by session_id on startup and the chat is restored.

        pip install langchain-community
    """
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.runnables.history import RunnableWithMessageHistory
    from langchain_community.chat_message_histories import SQLChatMessageHistory

    def get_session_history(session_id: str):
        # One SQLite file; each session_id is its own conversation row-set.
        return SQLChatMessageHistory(session_id=session_id, connection="sqlite:///chat_history.db")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are concise."),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ])
    with_history = RunnableWithMessageHistory(
        prompt | llm,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    cfg = {"configurable": {"session_id": "user-42"}}
    print("=== SQLChatMessageHistory (persists to chat_history.db) ===")
    print(with_history.invoke({"input": "Remember: my project is a RAG bot."}, config=cfg).content)
    print(with_history.invoke({"input": "What is my project?"}, config=cfg).content)
    print("\nStored in chat_history.db — survives restarts.")
    print()


# ---------------------------------------------------------------------------
# 3. PRODUCTION CONSIDERATIONS (read this)
# ---------------------------------------------------------------------------
def demo_production_notes():
    """
    - CHOICE OF DB: SQLite = single file, single machine (dev / small). Postgres =
      multi-server, concurrent, durable (real prod). Redis = fast, often with a TTL
      so old chats auto-expire. LangGraph has checkpointers for each.
    - PRIVACY / RETENTION: chat logs are user data. Have a deletion path (GDPR),
      and consider a TTL so you don't store conversations forever.
    - GROWTH STILL APPLIES: persistence does NOT remove the need to trim/summarise
      before sending to the model — you persist EVERYTHING but only SEND a window.
    - KEY DESIGN: thread_id/session_id must be unique per user-conversation and
      come from your auth layer, or users will collide (see Topic 11 failure modes).
    - MIGRATIONS / SCHEMA: a DB-backed store owns tables — manage schema changes
      like any other production database.
    """
    print(demo_production_notes.__doc__)


if __name__ == "__main__":
    # Run ONE at a time — uncomment as you go (installs noted in each docstring).
    demo_sqlite_checkpointer()
    # demo_sql_message_history()
    # demo_production_notes()
