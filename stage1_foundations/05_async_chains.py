"""
Topic 5: Async Chains
----------------------
Every method you learned has an async twin. Add an "a" prefix:

    .invoke()  → .ainvoke()
    .stream()  → .astream()
    .batch()   → .abatch()

THE ONE THING TO UNDERSTAND:
Async does NOT make a single LLM call faster. One call still takes ~2 seconds.
What async gives you is the ability to WAIT for many calls AT THE SAME TIME,
instead of one after another.

An LLM call is "I/O-bound" — your code sends a request, then sits idle for
~2 seconds doing nothing while OpenAI's servers think. Async lets you fire off
the next call during that idle time instead of blocking.

WHY THIS MATTERS IN PRODUCTION:
- A web API (FastAPI) runs ONE event loop. If request A makes a blocking
  sync LLM call, the whole server freezes for 2s — request B, C, D all wait.
  Async lets the server serve B, C, D while A waits on OpenAI.
- Batch jobs: summarising 500 documents sequentially = 500 × 2s = ~17 min.
  Concurrently (bounded) = a couple of minutes.

WHAT BREAKS UNDER LOAD (covered in the demos below):
- Fire 500 calls at once → 429 Rate Limit errors + memory blowup.
- One call fails → naive gather kills the whole batch.
- A call hangs → no timeout means you wait forever.
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
    ("system", "You are a helpful assistant. Answer in one short sentence."),
    ("human", "{question}"),
])

chain = prompt | llm | parser


# ---------------------------------------------------------------------------
# 1. ainvoke — the async version of invoke
# ---------------------------------------------------------------------------
async def demo_ainvoke():
    """
    Same as .invoke(), but you 'await' it.
    'await' = "pause this function here, let other work run, resume when the
    result is ready." On its own (one call) it gives NO speedup — the win only
    shows up when you await many things together (next demo).
    """
    print("=== ainvoke() ===")
    result = await chain.ainvoke({"question": "What is async in one line?"})
    print(result)
    print()


# ---------------------------------------------------------------------------
# 2. astream — stream tokens in an async context
# ---------------------------------------------------------------------------
async def demo_astream():
    """
    'async for' instead of 'for'. This is what a real streaming API endpoint
    uses — tokens are pushed to the frontend as they arrive, while the same
    server stays free to handle other users.
    """
    print("=== astream() ===")
    async for chunk in chain.astream({"question": "Explain the event loop in 2 sentences."}):
        print(chunk, end="", flush=True)
    print("\n")


# ---------------------------------------------------------------------------
# 3. THE CORE LESSON: sequential vs concurrent
# ---------------------------------------------------------------------------
async def demo_sequential_vs_concurrent():
    """
    This is the whole point of async. Watch the timing difference.

    NAIVE (looks async, runs sequential):
        for q in questions:
            await chain.ainvoke(q)   # awaits each ONE fully before the next
    This is the #1 async mistake — using 'await' in a loop. The 'a' prefix
    tricks you into thinking it's concurrent. It is NOT. Each await blocks the
    loop until that call finishes. Total time = sum of all calls.

    PRODUCTION (truly concurrent):
        asyncio.gather(*coroutines)   # fires all at once, waits for all
    gather schedules every call immediately, then waits for the whole set.
    Total time ≈ the SLOWEST single call, not the sum.
    """
    questions = [
        {"question": "What is Python?"},
        {"question": "What is Docker?"},
        {"question": "What is an API?"},
        {"question": "What is JSON?"},
    ]

    # --- Sequential (the trap) ---
    start = time.perf_counter()
    for q in questions:
        await chain.ainvoke(q)
    sequential_time = time.perf_counter() - start

    # --- Concurrent (gather) ---
    start = time.perf_counter()
    # Build a list of coroutines WITHOUT awaiting them yet, then hand to gather.
    coroutines = [chain.ainvoke(q) for q in questions]
    await asyncio.gather(*coroutines)
    concurrent_time = time.perf_counter() - start

    print("=== sequential vs concurrent ===")
    print(f"Sequential (await in loop): {sequential_time:.2f}s")
    print(f"Concurrent (asyncio.gather): {concurrent_time:.2f}s")
    print(f"Speedup: {sequential_time / concurrent_time:.1f}x")
    print()


# ---------------------------------------------------------------------------
# 4. FAILURE MODE: unbounded concurrency → rate limits + memory blowup
#    FIX: a Semaphore to cap how many calls run at once
# ---------------------------------------------------------------------------
async def demo_bounded_concurrency():
    """
    gather(*1000 calls) fires ALL 1000 at once. In production that gets you:
      - 429 Rate Limit errors (you blow past OpenAI's requests-per-minute)
      - memory blowup (1000 in-flight HTTP requests + 1000 pending results)
      - a thundering-herd hit on any downstream service

    THE FIX: a Semaphore — a counter that only lets N tasks run at a time.
    Acquire before the call, release after. The other tasks queue and wait
    their turn. This is THE standard production pattern for "do many things,
    but not too many at once."
    """
    semaphore = asyncio.Semaphore(3)  # max 3 in-flight at any moment

    async def bounded_call(q):
        async with semaphore:  # waits here if 3 are already running
            return await chain.ainvoke(q)

    questions = [{"question": f"Give me fact #{i} about space."} for i in range(8)]

    start = time.perf_counter()
    results = await asyncio.gather(*[bounded_call(q) for q in questions])
    elapsed = time.perf_counter() - start

    print("=== bounded concurrency (Semaphore, max 3) ===")
    print(f"Ran {len(results)} calls, 3 at a time, in {elapsed:.2f}s")
    print(f"Sample: {results[0]}")
    print()


# ---------------------------------------------------------------------------
# 5. The BUILT-IN production way: abatch with max_concurrency
# ---------------------------------------------------------------------------
async def demo_abatch():
    """
    Honest answer: you usually DON'T hand-roll the Semaphore for simple cases.
    LangChain's .abatch() already does bounded concurrency for you via a config
    flag. This is the idiomatic production call for "run this chain over a list".

        config={"max_concurrency": 5}

    Use abatch when: same chain, many inputs, you just want all results.
    Use the manual Semaphore when: you need custom per-item logic, mixed
    operations, or progress/error handling that abatch doesn't expose.
    """
    questions = [{"question": f"One fun fact about the number {i}."} for i in range(6)]

    start = time.perf_counter()
    results = await chain.abatch(questions, config={"max_concurrency": 3})
    elapsed = time.perf_counter() - start

    print("=== abatch(max_concurrency=3) ===")
    print(f"{len(results)} results in {elapsed:.2f}s")
    for r in results[:2]:
        print(" -", r)
    print()


# ---------------------------------------------------------------------------
# 6. FAILURE MODE: one call fails → it kills the whole gather
#    FIX: return_exceptions=True, then filter
# ---------------------------------------------------------------------------
async def demo_error_handling():
    """
    Default gather behaviour: if ANY one coroutine raises, gather immediately
    raises that exception and you lose the results of every other call — even
    the ones that succeeded. In a 500-doc batch, one bad input wastes 499 good
    API calls (and you still paid for them).

    FIX: gather(..., return_exceptions=True). Now a failure comes back as an
    Exception OBJECT in the results list instead of blowing everything up. You
    inspect each result and decide what to retry / log / drop.
    """
    async def maybe_fail(q, should_fail):
        if should_fail:
            raise ValueError(f"simulated failure for: {q}")
        return await chain.ainvoke({"question": q})

    tasks = [
        maybe_fail("What is gravity?", should_fail=False),
        maybe_fail("This one breaks", should_fail=True),   # boom
        maybe_fail("What is light?", should_fail=False),
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    print("=== error handling (return_exceptions=True) ===")
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            print(f"[{i}] FAILED → {type(r).__name__}: {r}")
        else:
            print(f"[{i}] OK     → {r}")
    print()


# ---------------------------------------------------------------------------
# 7. FAILURE MODE: a call hangs forever → no timeout means your server hangs
#    FIX: asyncio.timeout (or wait_for)
# ---------------------------------------------------------------------------
async def demo_timeout():
    """
    Networks stall. An OpenAI call can occasionally hang. With no timeout, that
    request holds a server slot forever and slowly starves you of capacity.
    Always bound LLM calls with a timeout in production.

    asyncio.timeout(seconds) cancels whatever's inside if it overruns.
    (Set a deliberately tiny timeout here so you can SEE it trigger.)
    """
    print("=== timeout ===")
    try:
        async with asyncio.timeout(0.001):  # 1ms — will always trip, on purpose
            await chain.ainvoke({"question": "Why is the sky blue?"})
    except asyncio.TimeoutError:
        print("Call exceeded the timeout and was cancelled — handle/retry here.")
    print()


# ---------------------------------------------------------------------------
# Async needs an event loop. asyncio.run() starts one, runs the coroutine,
# then shuts it down. You cannot 'await' at the top level — it must live
# inside an async function that asyncio.run() drives.
# ---------------------------------------------------------------------------
async def main():
    # Run ONE at a time — uncomment as you go.
    # await demo_ainvoke()
    # await demo_astream()
    # await demo_sequential_vs_concurrent()
    await demo_bounded_concurrency()
    # await demo_abatch()
    # await demo_error_handling()
    # await demo_timeout()


if __name__ == "__main__":
    asyncio.run(main())
