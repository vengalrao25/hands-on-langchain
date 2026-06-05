"""
Topic 8: Debugging & Observability (SEE what your chain is doing)
-----------------------------------------------------------------
When a chain misbehaves, the #1 question is: "what did the model ACTUALLY
receive and return?" Without visibility you're guessing. In production you also
need to know, for every call: how many tokens, how much money, how long, and
which user/request it belonged to.

This file is about making the invisible visible.

WHAT THIS FILE COVERS:
  1. set_debug / set_verbose — the quick "show me everything" switch
  2. Run config — run_name, tags, metadata: label runs so you can find them
  3. Custom callback — hook into every LLM call to log tokens, cost, latency
  4. Why this matters: this is the foundation of monitoring, cost control, and
     debugging in prod (and exactly what LangSmith automates — Stage 6).

MENTAL MODEL:
  A "callback" is a function LangChain calls AT KEY MOMENTS during a run
  (on_llm_start, on_llm_end, on_chain_error, ...). You attach callbacks to log,
  measure, or alert — without touching your chain logic.
"""

import time

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import BaseCallbackHandler

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = StrOutputParser()
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are concise."),
    ("human", "{question}"),
])
chain = prompt | llm | parser


# ---------------------------------------------------------------------------
# 1. set_debug / set_verbose — the firehose
# ---------------------------------------------------------------------------
def demo_debug_verbose():
    """
    Two global switches for quick debugging:
      set_verbose(True) — prints a readable summary of each step.
      set_debug(True)   — prints EVERYTHING (the exact prompt sent, raw response,
                          every intermediate input/output). Very noisy — use it
                          to catch "what prompt did the model actually get?" bugs,
                          then turn it OFF. Never leave debug on in production.
    """
    from langchain.globals import set_debug, set_verbose

    print("=== set_verbose(True) ===")
    set_verbose(True)
    chain.invoke({"question": "What is 2+2?"})
    set_verbose(False)

    print("\n=== set_debug(True) — full firehose ===")
    set_debug(True)
    chain.invoke({"question": "Name a color."})
    set_debug(False)
    print()


# ---------------------------------------------------------------------------
# 2. RUN CONFIG — label your runs so you can find them later
# ---------------------------------------------------------------------------
def demo_run_config():
    """
    Every .invoke() takes a config dict. In production you ALWAYS pass:
      run_name  — a human label for this call ("sentiment-classify")
      tags      — categories to filter by (["prod", "checkout-flow"])
      metadata  — structured context (user_id, request_id, feature)

    These flow into your callbacks AND into LangSmith traces, so you can later
    answer "show me all failed calls for user X in the checkout flow." Without
    them, every call in your logs looks identical and debugging is misery.
    """
    print("=== run config (labels flow to observability) ===")
    result = chain.invoke(
        {"question": "What is Docker?"},
        config={
            "run_name": "explain-docker",
            "tags": ["demo", "stage1"],
            "metadata": {"user_id": "u_123", "feature": "learning"},
        },
    )
    print(result)
    print("(run_name/tags/metadata are now attached to this run for tracing)")
    print()


# ---------------------------------------------------------------------------
# 3. CUSTOM CALLBACK — log tokens, cost, and latency for every call
# ---------------------------------------------------------------------------
COST_PER_1M_INPUT = 0.150
COST_PER_1M_OUTPUT = 0.600


class UsageLogger(BaseCallbackHandler):
    """
    A callback that times each LLM call and logs its token usage + cost.
    This is a hand-rolled version of what observability platforms do for you.
    Override the lifecycle hooks you care about; ignore the rest.
    """

    def on_llm_start(self, serialized, prompts, **kwargs):
        self._start = time.perf_counter()

    def on_llm_end(self, response, **kwargs):
        elapsed = time.perf_counter() - getattr(self, "_start", time.perf_counter())

        # Token usage lives in llm_output for chat models.
        usage = (response.llm_output or {}).get("token_usage", {})
        in_tok = usage.get("prompt_tokens", 0)
        out_tok = usage.get("completion_tokens", 0)
        cost = in_tok / 1_000_000 * COST_PER_1M_INPUT + out_tok / 1_000_000 * COST_PER_1M_OUTPUT

        print(f"[USAGE] {elapsed:.2f}s | in={in_tok} out={out_tok} | ${cost:.8f}")

    def on_llm_error(self, error, **kwargs):
        print(f"[ERROR] LLM call failed: {type(error).__name__}: {error}")


def demo_custom_callback():
    """
    Attach the callback via config={"callbacks": [...]}. It fires automatically
    around every LLM call in the chain — your chain code stays untouched. This
    is THE pattern for per-request cost tracking and latency monitoring.
    """
    print("=== custom callback (token/cost/latency logging) ===")
    logger = UsageLogger()
    result = chain.invoke(
        {"question": "Explain APIs in one sentence."},
        config={"callbacks": [logger]},
    )
    print("Answer:", result)
    print()


if __name__ == "__main__":
    # Run ONE at a time — uncomment as you go.
    demo_debug_verbose()
    # demo_run_config()
    # demo_custom_callback()
