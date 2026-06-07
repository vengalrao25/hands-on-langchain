"""
Topic 9: Async Chains
----------------------
Async is NOT optional in production. Every real LangChain app (FastAPI, chatbots,
pipelines) runs async. If you use .invoke() in a web server you block the entire
thread while waiting for the LLM — no other request can be handled.

SYNC vs ASYNC — the mental model:
  sync:  you call the LLM → your thread sits and waits → response comes back
  async: you call the LLM → your thread is freed → response comes back → you continue
                              ↑ other requests can be served here

WHAT THIS FILE COVERS:
  1. ainvoke()        — async version of invoke()
  2. astream()        — async token streaming (what you use in APIs)
  3. abatch()         — parallel async batch calls
  4. asyncio.gather() — run multiple independent chains concurrently (fast)
  5. Concurrency limits — rate limits + how to control parallelism
  6. Async in FastAPI  — the real-world integration pattern
  7. Error handling    — what happens when one async call fails in a batch
"""

import asyncio
import time

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

prompt = ChatPromptTemplate.from_messages([
    ("system", "Be concise. One sentence answers only."),
    ("human", "{question}"),
])

chain = prompt | llm | parser


# ---------------------------------------------------------------------------
# 1. ainvoke() — async version of invoke()
# ---------------------------------------------------------------------------
async def demo_ainvoke():
    """
    Drop-in async replacement for .invoke(). Use this inside any async context
    (FastAPI endpoint, async event loop). The LLM call doesn't block the thread.
    """
    result = await chain.ainvoke({"question": "What is Python?"})

    print("=== ainvoke ===")
    print(result)
    print()


# ---------------------------------------------------------------------------
# 2. astream() — async token streaming
# ---------------------------------------------------------------------------
async def demo_astream():
    """
    Stream tokens as they arrive without blocking. This is how chatbot UIs
    show responses word by word — server streams tokens to the frontend in real time.

    In FastAPI you'd yield each chunk to a StreamingResponse.
    """
    print("=== astream (tokens arrive live) ===")
    async for chunk in chain.astream({"question": "Explain what a neural network is."}):
        print(chunk, end="", flush=True)
    print("\n")


# ---------------------------------------------------------------------------
# 3. abatch() — parallel async batch
# ---------------------------------------------------------------------------
async def demo_abatch():
    """
    abatch() sends all inputs at once and runs them concurrently.
    Much faster than calling ainvoke() in a loop sequentially.

    Use when you have N independent inputs to process — summarising N docs,
    classifying N messages, translating N sentences.
    """
    questions = [
        {"question": "What is JavaScript?"},
        {"question": "What is a REST API?"},
        {"question": "What is Docker?"},
        {"question": "What is Kubernetes?"},
    ]

    start = time.time()
    results = await chain.abatch(questions)
    elapsed = time.time() - start

    print("=== abatch (concurrent) ===")
    for q, r in zip(questions, results):
        print(f"Q: {q['question']}")
        print(f"A: {r[:80]}")
        print()
    print(f"4 calls in {elapsed:.1f}s (vs ~{elapsed * 4:.0f}s sequential)")
    print()


# ---------------------------------------------------------------------------
# 4. asyncio.gather() — run DIFFERENT chains concurrently
# ---------------------------------------------------------------------------
async def demo_gather():
    """
    abatch() runs the SAME chain on multiple inputs.
    asyncio.gather() runs DIFFERENT chains on different inputs concurrently.

    Use this when you need to hit multiple independent chains simultaneously
    and wait for ALL of them before proceeding (fan-out + join).

    Real example: for a user profile page, concurrently fetch:
      - their activity summary
      - their risk score
      - their recommended items
    All three are independent → run them together, not one after another.
    """
    summary_chain = (
        ChatPromptTemplate.from_messages([("human", "Summarise in one sentence: {topic}")])
        | llm | parser
    )
    keyword_chain = (
        ChatPromptTemplate.from_messages([("human", "Give 3 keywords for: {topic}. Comma separated.")])
        | llm | parser
    )
    question_chain = (
        ChatPromptTemplate.from_messages([("human", "Give one follow-up question about: {topic}")])
        | llm | parser
    )

    start = time.time()
    summary, keywords, question = await asyncio.gather(
        summary_chain.ainvoke({"topic": "quantum computing"}),
        keyword_chain.ainvoke({"topic": "quantum computing"}),
        question_chain.ainvoke({"topic": "quantum computing"}),
    )
    elapsed = time.time() - start

    print("=== asyncio.gather (3 chains, same time) ===")
    print("Summary :", summary)
    print("Keywords:", keywords)
    print("Question:", question)
    print(f"All 3 done in {elapsed:.1f}s (would be ~3x longer sequentially)")
    print()


# ---------------------------------------------------------------------------
# 5. CONCURRENCY LIMITS — don't hammer the API
# ---------------------------------------------------------------------------
async def demo_concurrency_limits():
    """
    Running 100 requests with asyncio.gather() at once will hit OpenAI's
    rate limit (requests per minute / tokens per minute) and start getting
    429 errors.

    Production pattern: use max_concurrency in abatch() to cap parallelism.
    Or chunk your inputs and process N at a time with asyncio.Semaphore.
    """
    questions = [{"question": f"What is concept number {i}?"} for i in range(6)]

    print("=== concurrency limits ===")
    print("Running 6 calls with max_concurrency=2 (2 at a time)")

    start = time.time()
    results = await chain.abatch(questions, config={"max_concurrency": 2})
    elapsed = time.time() - start

    print(f"Done in {elapsed:.1f}s")
    print(f"Sample: {results[0][:60]}")
    print()


# ---------------------------------------------------------------------------
# 6. ERROR HANDLING — one failure shouldn't kill the whole batch
# ---------------------------------------------------------------------------
async def demo_error_handling():
    """
    When one call in a gather fails, asyncio.gather() raises the first exception
    and cancels everything else by default.

    Production fix: return_exceptions=True — failed calls return the exception
    object instead of raising, so successful results are still usable.
    """
    async def call_with_possible_error(question: str, should_fail: bool):
        if should_fail:
            raise ValueError(f"Simulated failure for: {question}")
        return await chain.ainvoke({"question": question})

    print("=== error handling in async batch ===")
    results = await asyncio.gather(
        call_with_possible_error("What is Python?", should_fail=False),
        call_with_possible_error("What is Java?", should_fail=True),   # this one fails
        call_with_possible_error("What is Go?", should_fail=False),
        return_exceptions=True,   # don't cancel others when one fails
    )

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Call {i}: FAILED — {result}")
        else:
            print(f"Call {i}: OK — {result[:60]}")
    print()


# ---------------------------------------------------------------------------
# 7. FASTAPI INTEGRATION PATTERN — how async chains live in a real API
# ---------------------------------------------------------------------------
def demo_fastapi_pattern():
    """
    This shows the pattern — not runnable directly (needs uvicorn/FastAPI).
    Copy this into a FastAPI app and it works as-is.

    Key rules:
    - FastAPI endpoints are async → always use ainvoke/astream, never invoke
    - For streaming: return StreamingResponse with an async generator
    - Never call sync .invoke() inside an async endpoint — blocks the server
    """
    pattern = '''
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

app = FastAPI()
chain = ChatPromptTemplate.from_messages([
    ("human", "{question}")
]) | ChatOpenAI(model="gpt-4o-mini") | StrOutputParser()

@app.post("/ask")
async def ask(question: str):
    result = await chain.ainvoke({"question": question})
    return {"answer": result}

@app.post("/stream")
async def stream(question: str):
    async def generate():
        async for chunk in chain.astream({"question": question}):
            yield chunk
    return StreamingResponse(generate(), media_type="text/plain")
'''
    print("=== FastAPI pattern ===")
    print(pattern)


# ---------------------------------------------------------------------------
# RUNNER
# ---------------------------------------------------------------------------
async def main():
    await demo_ainvoke()
    await demo_astream()
    await demo_abatch()
    await demo_gather()
    await demo_concurrency_limits()
    await demo_error_handling()
    demo_fastapi_pattern()


if __name__ == "__main__":
    asyncio.run(main())
