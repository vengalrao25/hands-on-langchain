"""
Topic 6: Sequential Chains
---------------------------
A sequential chain = the output of one step becomes the input of the next.

    topic → [Chain 1] → summary → [Chain 2] → tweet

You already did a baby version of this in 04 (nested chain). Now we do it the
production way and — more importantly — learn the thing that actually trips
people up: MATCHING THE DATA SHAPE between steps.

THE GOLDEN RULE OF CHAINING:
What step 1 OUTPUTS must match what step 2 EXPECTS as INPUT.
  - An LLM step with StrOutputParser outputs a plain string.
  - A prompt step expects a dict like {"summary": "..."}.
  - A string is NOT a dict → the chain breaks unless you reshape it.
Most "my chain doesn't work" bugs are a shape mismatch, not a logic error.

WHAT THIS FILE COVERS:
  1. The naive way (manual wiring) — so you see what's happening
  2. The production way (one LCEL pipe) + how to reshape between steps
  3. RunnableLambda — the reshaping tool
  4. Carrying the ORIGINAL input forward (the real-world case)
  5. Failure modes: shape mismatch, cost/latency stacking, error compounding
  6. When NOT to use a sequential chain (collapse it into one prompt instead)
"""

from operator import itemgetter

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

summary_prompt = ChatPromptTemplate.from_messages([
    ("human", "Summarise this topic in one clear sentence: {topic}"),
])

tweet_prompt = ChatPromptTemplate.from_messages([
    ("human", "Write a punchy tweet (with one emoji) based on this: {summary}"),
])


# ---------------------------------------------------------------------------
# 1. THE NAIVE WAY — wire it by hand so you understand the data flow
# ---------------------------------------------------------------------------
def demo_manual():
    """
    Run chain 1, grab its output, manually feed it into chain 2.
    This WORKS and is perfectly readable. For 2 steps it's honestly fine.
    The downside shows up at scale: 5+ steps becomes a tangle of temp
    variables, and you can't .stream() / .batch() the pipeline as one unit.
    """
    summary_chain = summary_prompt | llm | parser
    tweet_chain = tweet_prompt | llm | parser

    summary = summary_chain.invoke({"topic": "quantum computing"})
    tweet = tweet_chain.invoke({"summary": summary})

    print("=== manual wiring ===")
    print("Summary:", summary)
    print("Tweet  :", tweet)
    print()


# ---------------------------------------------------------------------------
# 2. THE SHAPE PROBLEM — why you can't just pipe them directly
# ---------------------------------------------------------------------------
def demo_shape_problem():
    """
    Intuition says: summary_chain | tweet_chain. But it breaks.

    summary_chain outputs a STRING ("Quantum computing is...").
    tweet_prompt expects a DICT {"summary": "..."}.
    Piping a string where a dict is expected → KeyError / type error.

    This is THE most common chaining bug. Run this to see the real error,
    then look at demo_lcel for the fix.
    """
    summary_chain = summary_prompt | llm | parser
    tweet_chain = tweet_prompt | llm | parser

    broken = summary_chain | tweet_chain  # string -> dict mismatch

    print("=== shape problem (this will error) ===")
    try:
        broken.invoke({"topic": "quantum computing"})
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
    print()


# ---------------------------------------------------------------------------
# 3. THE PRODUCTION WAY — one pipe, reshape between steps with RunnableLambda
# ---------------------------------------------------------------------------
def demo_lcel():
    """
    Fix the shape mismatch by inserting a tiny transform: take the string and
    wrap it back into the dict the next prompt wants.

        summary_chain | (str -> {"summary": str}) | tweet_chain

    RunnableLambda turns a plain Python function into a chain step. Any callable
    in a pipe is auto-wrapped, but writing RunnableLambda explicitly makes the
    intent obvious. Now the WHOLE thing is one runnable — you can .invoke(),
    .stream(), .batch(), or .ainvoke() it as a single unit. That's the win.
    """
    summary_chain = summary_prompt | llm | parser
    tweet_chain = tweet_prompt | llm | parser

    full_chain = (
        summary_chain
        | RunnableLambda(lambda summary: {"summary": summary})  # reshape: str -> dict
        | tweet_chain
    )

    result = full_chain.invoke({"topic": "quantum computing"})

    print("=== production LCEL (one pipe) ===")
    print(result)
    print()


# ---------------------------------------------------------------------------
# 4. THE REAL-WORLD CASE — carry the ORIGINAL input forward
# ---------------------------------------------------------------------------
def demo_carry_input():
    """
    Real pipelines rarely throw away earlier data. Often the LAST step needs
    BOTH the original input AND every intermediate result.

    Here the final prompt wants the original {topic} AND the generated {summary}.
    A plain pipe can't do this — once you transform to a string, the topic is
    gone. The pattern: build a DICT that accumulates fields as you go.

        {"topic": <original>}
            -> add "summary" computed from topic
            -> final prompt uses both

    RunnablePassthrough.assign() adds a new key to the dict WITHOUT dropping the
    existing ones. This is the workhorse of multi-step chains (full deep-dive in
    topic 7). itemgetter pulls a single field out of the dict for a sub-chain.
    """
    final_prompt = ChatPromptTemplate.from_messages([
        ("human",
         "Topic: {topic}\nSummary: {summary}\n\n"
         "Write a tweet that references the topic name explicitly."),
    ])

    summary_subchain = summary_prompt | llm | parser

    full_chain = (
        # input is {"topic": "..."}; .assign adds "summary" alongside it
        RunnablePassthrough.assign(
            summary=itemgetter("topic") | summary_prompt | llm | parser
        )
        # dict is now {"topic": ..., "summary": ...} — final prompt sees both
        | final_prompt
        | llm
        | parser
    )

    result = full_chain.invoke({"topic": "the James Webb Space Telescope"})

    print("=== carry original input forward ===")
    print(result)
    print()


# ---------------------------------------------------------------------------
# 5. FAILURE MODE: cost & latency STACK with every LLM step
# ---------------------------------------------------------------------------
def demo_cost_latency_reality():
    """
    Every LLM step is a separate paid API call with its own ~2s latency.

        3-step chain  =  3x the cost  +  3x the latency  +  3x the failure surface

    A user-facing request that chains 4 LLM calls can take 8-10 seconds — too
    slow. And errors COMPOUND: if step 1 produces a slightly wrong summary,
    step 2 faithfully builds garbage on top of it (garbage in, garbage out).
    The LLM can't tell that its input was already bad.

    Production habit: count the LLM calls in your chain. Each one is money +
    latency + a thing that can fail. Don't add a step unless it earns its place.
    """
    summary_chain = summary_prompt | llm | parser
    full_chain = (
        summary_chain
        | RunnableLambda(lambda s: {"summary": s})
        | tweet_prompt | llm | parser
    )

    # Inspect how many runnables are in the pipeline.
    print("=== cost / latency reality ===")
    print(f"Steps in this chain: {len(full_chain.steps)}")
    print("LLM calls per invoke: 2 (one summary + one tweet) → 2x cost & latency")
    print(full_chain.invoke({"topic": "black holes"}))
    print()


# ---------------------------------------------------------------------------
# 6. WHEN NOT TO CHAIN — collapse multiple LLM steps into ONE prompt
# ---------------------------------------------------------------------------
def demo_collapse_into_one():
    """
    Honest production answer: a 'summary then tweet' chain is usually a WASTE.
    Two LLM calls to do what one prompt can do in a single call — half the
    cost, half the latency.

    Use a SEQUENTIAL chain only when a step genuinely needs the model's full
    attention on its own (e.g. extract structured data, THEN reason over it),
    or when an intermediate result is reused elsewhere, or each step uses a
    different model / tool.

    Use a SINGLE prompt when the steps are just "do A, then do B with A" and the
    model can clearly handle both in one shot. When in doubt, start with one
    prompt and only split when quality demands it.
    """
    one_shot = ChatPromptTemplate.from_messages([
        ("human",
         "For the topic '{topic}': first summarise it in one sentence, then "
         "write a punchy tweet based on that summary. Return only the tweet."),
    ])
    chain = one_shot | llm | parser

    print("=== collapse into one prompt (1 call instead of 2) ===")
    print(chain.invoke({"topic": "quantum computing"}))
    print()

 

if __name__ == "__main__":
    # Run ONE at a time — uncomment as you go.
    # demo_manual()
    # demo_shape_problem()
    # demo_lcel()
    # demo_carry_input()
    # demo_cost_latency_reality()
    demo_collapse_into_one()
