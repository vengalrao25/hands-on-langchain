"""
Topic 7: RunnablePassthrough & RunnableMap
-------------------------------------------
The two tools that give you control over DATA FLOW inside a chain.

RunnablePassthrough       → pass data through unchanged (don't transform it)
RunnablePassthrough.assign() → add new keys to the dict without losing old ones ← workhorse
RunnableMap (dict syntax) → run multiple chains IN PARALLEL and merge results into one dict

THE PROBLEM THEY SOLVE:
A plain pipe only passes one thing at a time. Once step 1 transforms your input,
the original is gone. These tools let you branch, carry forward, and merge — so
downstream steps see everything they need.

PRODUCTION RELEVANCE: high.
RunnablePassthrough.assign() is in almost every real RAG and multi-step chain.
RunnableMap powers parallel fan-out (retrieve from 3 sources at once, merge results).
"""

from operator import itemgetter

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableMap, RunnableLambda

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()


# ---------------------------------------------------------------------------
# 1. RunnablePassthrough — pass the input through untouched
# ---------------------------------------------------------------------------
def demo_passthrough():
    """
    RunnablePassthrough does nothing — it passes whatever it receives straight
    through. Sounds useless, but it's essential when you need ONE branch of a
    parallel map to just echo the original input while other branches transform it.
    """
    chain = RunnablePassthrough() | RunnableLambda(lambda x: f"Got: {x}")

    result = chain.invoke("hello")
    print("=== RunnablePassthrough ===")
    print(result)   # "Got: hello"
    print()


# ---------------------------------------------------------------------------
# 2. RunnablePassthrough.assign() — add keys to dict, keep existing ones
# ---------------------------------------------------------------------------
def demo_assign():
    """
    The workhorse. Takes an input dict and adds new computed keys alongside
    the existing ones. The downstream step sees ALL of them.

    This is how multi-step chains carry context forward — you keep building
    on the same dict rather than replacing it at each step.
    """
    summarise = ChatPromptTemplate.from_messages([
        ("human", "Summarise in one sentence: {topic}"),
    ]) | llm | parser

    sentiment = ChatPromptTemplate.from_messages([
        ("human", "What is the general sentiment (positive/negative/neutral) of this topic: {topic}? One word only."),
    ]) | llm | parser

    chain = (
        RunnablePassthrough.assign(
            summary=itemgetter("topic") | summarise,
            sentiment=itemgetter("topic") | sentiment,
        )
        # dict now has: topic, summary, sentiment
    )

    result = chain.invoke({"topic": "climate change"})

    print("=== RunnablePassthrough.assign() ===")
    print("Topic    :", result["topic"])
    print("Summary  :", result["summary"])
    print("Sentiment:", result["sentiment"])
    print()


# ---------------------------------------------------------------------------
# 3. RunnableMap — parallel fan-out, merged into one dict
# ---------------------------------------------------------------------------
def demo_runnable_map():
    """
    RunnableMap runs multiple runnables IN PARALLEL on the SAME input
    and collects all outputs into a single dict.

    Identical to the dict syntax: {"key": runnable, "key2": runnable2}
    LangChain auto-wraps a plain dict in a RunnableMap when it appears in a pipe.

    Use this when you want to compute multiple things from the same input
    without doing them sequentially (faster: parallel network calls).
    """
    topic_chain = ChatPromptTemplate.from_messages([
        ("human", "Name 3 subtopics of: {input}. Comma separated, no explanation."),
    ]) | llm | parser

    example_chain = ChatPromptTemplate.from_messages([
        ("human", "Give one real-world example of: {input}. One sentence."),
    ]) | llm | parser

    # Dict syntax = RunnableMap under the hood
    parallel_chain = {
        "subtopics": topic_chain,
        "example": example_chain,
    }

    result = RunnableMap(parallel_chain).invoke({"input": "machine learning"})

    print("=== RunnableMap (parallel) ===")
    print("Subtopics:", result["subtopics"])
    print("Example  :", result["example"])
    print()


# ---------------------------------------------------------------------------
# 4. PRODUCTION PATTERN — RAG context builder (preview of Stage 4)
# ---------------------------------------------------------------------------
def demo_rag_context_pattern():
    """
    This is the exact pattern used in real RAG chains.

    Input: {"question": "..."}
    Step 1: run the question through a retriever to get context (we mock it here)
    Step 2: final prompt sees BOTH question AND context

    RunnablePassthrough.assign() makes this clean — question stays in the dict,
    context is added alongside it.
    """
    # Mock retriever — in production this hits a vector DB
    def fake_retriever(question: str) -> str:
        return f"[Retrieved context for: {question}] Python is a high-level language..."

    final_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer using only the context provided. If unsure, say so."),
        ("human", "Context: {context}\n\nQuestion: {question}"),
    ])

    chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | RunnableLambda(fake_retriever)
        )
        | final_prompt
        | llm
        | parser
    )

    result = chain.invoke({"question": "What is Python?"})

    print("=== RAG context pattern ===")
    print(result)
    print()


# ---------------------------------------------------------------------------
# 5. FAILURE MODE — forgetting itemgetter when a step expects a STRING
# ---------------------------------------------------------------------------
def demo_failure_mode():
    """
    assign() passes the WHOLE input dict to each subchain. That's FINE if the
    subchain is a prompt template (it just reads the keys it needs). The trap is
    when a step expects a plain STRING/VALUE — a retriever, or a function that
    does string operations. It receives the dict and blows up.

    Watch: word_counter does text.split() — that works on a string but a dict
    has no .split(), so the 'broken' version raises AttributeError. itemgetter
    pulls the string out FIRST, which is the fix.

    Rule: if the next step wants a value (not the whole dict), use itemgetter("key").
    """
    # A step that expects a STRING (it calls .split() on its input).
    word_counter = RunnableLambda(lambda text: f"{len(text.split())} words")

    print("=== failure mode (itemgetter missing) ===")
    try:
        # WRONG: the full dict {"topic": "..."} reaches word_counter,
        # which calls dict.split() → AttributeError.
        broken = RunnablePassthrough.assign(stats=word_counter)
        broken.invoke({"topic": "machine learning is fun"})
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
    print()

    # CORRECT: itemgetter extracts the string before word_counter sees it.
    fixed = RunnablePassthrough.assign(stats=itemgetter("topic") | word_counter)
    result = fixed.invoke({"topic": "machine learning is fun"})
    print("=== fixed (with itemgetter) ===")
    print(result)   # {"topic": "...", "stats": "4 words"}
    print()


if __name__ == "__main__":
    demo_passthrough()
    demo_assign()
    demo_runnable_map()
    demo_rag_context_pattern()
    demo_failure_mode()
