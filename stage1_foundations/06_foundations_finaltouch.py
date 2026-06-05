"""
Topic 6: Foundations — Final Touch (the gaps that make you production-ready)
----------------------------------------------------------------------------
01-05 taught you to TALK to a model and build chains. But a real app also has
to be CONTROLLABLE, CHEAP, OBSERVABLE, and RESILIENT. This file fills the gaps
that separate a notebook demo from something you'd put in front of users.

What's here (each is a foundational skill you'll use in every project):
  1. Model parameters     — temperature, max_tokens, timeout, max_retries
  2. Determinism          — temperature=0 / seed: making output repeatable
  3. Token usage & cost   — read tokens off the response, compute real $ cost
  4. init_chat_model      — stop hardcoding the provider (model-agnostic)
  5. .with_retry()        — survive transient rate-limit / network blips
  6. .with_fallbacks()    — fall back to another model when the primary dies
  7. Caching              — same prompt twice = 0 cost, instant (set_llm_cache)
  8. Real error types     — what actually gets raised, and how to catch it

These touch a few Stage 6 topics on purpose — but as FOUNDATIONS. You should
know they exist NOW so every chain you build from here is already production-shaped.
"""

import time

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

parser = StrOutputParser()


# ---------------------------------------------------------------------------
# 1. MODEL PARAMETERS — the knobs you control on every model
# ---------------------------------------------------------------------------
def demo_model_parameters():
    """
    The defaults are fine for playing. In production you set these explicitly:

      temperature  — randomness/creativity. 0 = focused & near-deterministic,
                     1+ = creative & varied. Use 0 for extraction/classification/
                     anything you want consistent; higher for brainstorming/copy.
      max_tokens   — hard cap on the RESPONSE length. Caps cost AND stops a
                     runaway model from generating forever. (output tokens are
                     the expensive ones — you cared about this earlier.)
      timeout      — give up on a single call after N seconds (don't hang forever).
      max_retries  — LangChain auto-retries transient errors this many times.
    """
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,        # consistent answers
        max_tokens=60,        # never generate more than ~60 tokens
        timeout=20,           # fail the call if it takes > 20s
        max_retries=2,        # retry transient failures twice before giving up
    )

    print("=== model parameters ===")
    print(llm.invoke("List 3 uses of Python.").content)
    print()


# ---------------------------------------------------------------------------
# 2. DETERMINISM — making the SAME input give the SAME output
# ---------------------------------------------------------------------------
def demo_determinism():
    """
    LLMs are random by default — ask twice, get two different answers. That's
    death for tests, caching, and reproducible pipelines.

    temperature=0 gets you MOSTLY repeatable output (not a 100% guarantee —
    the model is still probabilistic under the hood, but practically stable).
    For a stronger guarantee, OpenAI also supports a 'seed' (pass it via
    model_kwargs={"seed": 42}); same seed + same input → same output, mostly.

    Rule: any step whose output another step depends on → temperature=0.
    """
    creative = ChatOpenAI(model="gpt-4o-mini", temperature=1.2)
    strict = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    q = "Give me a one-word name for a coffee shop."

    print("=== determinism ===")
    print("temp=1.2 (varies):", creative.invoke(q).content, "/", creative.invoke(q).content)
    print("temp=0  (stable):", strict.invoke(q).content, "/", strict.invoke(q).content)
    print()


# ---------------------------------------------------------------------------
# 3. TOKEN USAGE & REAL COST — read it straight off the response
# ---------------------------------------------------------------------------
# Prices change — check OpenAI's pricing page. As of writing, gpt-4o-mini:
COST_PER_1M_INPUT = 0.150   # USD per 1,000,000 input tokens
COST_PER_1M_OUTPUT = 0.600  # USD per 1,000,000 output tokens


def demo_token_usage_and_cost():
    """
    Every response carries a usage_metadata dict with the exact token counts.
    This is how you track spend per request in production (log it, alert on it).
    Remember: OUTPUT tokens cost ~4x INPUT tokens here — long answers hurt.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm.invoke("Explain recursion in 2 sentences.")

    usage = response.usage_metadata  # {'input_tokens', 'output_tokens', 'total_tokens'}
    cost = (
        usage["input_tokens"] / 1_000_000 * COST_PER_1M_INPUT
        + usage["output_tokens"] / 1_000_000 * COST_PER_1M_OUTPUT
    )

    print("=== token usage & cost ===")
    print("Answer       :", response.content)
    print("Input tokens :", usage["input_tokens"])
    print("Output tokens:", usage["output_tokens"])
    print(f"This call cost: ${cost:.8f}")
    print()


# ---------------------------------------------------------------------------
# 4. init_chat_model — stop hardcoding ChatOpenAI everywhere
# ---------------------------------------------------------------------------
def demo_init_chat_model():
    """
    Hardcoding ChatOpenAI in 50 files means switching to Anthropic/Google later
    = editing 50 files. init_chat_model is the model-AGNOSTIC initializer: pass
    a model string (+ provider) and it returns the right chat model. Swap the
    string, swap the whole backend. This is how production code stays portable.

        init_chat_model("gpt-4o-mini", model_provider="openai")
        init_chat_model("claude-3-5-sonnet-latest", model_provider="anthropic")

    Best practice: read the model name from an env var / config, never literal.
    """
    from langchain.chat_models import init_chat_model

    llm = init_chat_model("gpt-4o-mini", model_provider="openai", temperature=0)

    print("=== init_chat_model (model-agnostic) ===")
    print("Type:", type(llm).__name__)  # still a ChatOpenAI under the hood
    print(llm.invoke("Say hi in 3 words.").content)
    print()


# ---------------------------------------------------------------------------
# 5. .with_retry() — survive transient failures automatically
# ---------------------------------------------------------------------------
def demo_with_retry():
    """
    Rate limits (429) and network blips are NORMAL at scale, not exceptional.
    .with_retry() wraps any runnable so it auto-retries on failure with
    exponential backoff (wait longer between each attempt).

    Note: ChatOpenAI already retries internally (max_retries). .with_retry()
    is the GENERIC LCEL way — it works on ANY runnable in a chain, not just the
    model, and lets you control attempts/backoff at the pipeline level.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    resilient = llm.with_retry(
        stop_after_attempt=3,            # try up to 3 times total
        wait_exponential_jitter=True,    # back off with randomized delay
    )

    print("=== with_retry() ===")
    print(resilient.invoke("What is 7 * 8?").content)
    print("(retries happen invisibly on transient errors)")
    print()


# ---------------------------------------------------------------------------
# 6. .with_fallbacks() — when the primary model fails, use a backup
# ---------------------------------------------------------------------------
def demo_with_fallbacks():
    """
    Retries handle TRANSIENT failures. Fallbacks handle a model being DOWN,
    deprecated, overloaded, or rejecting the request. If the primary raises
    after its retries, LangChain transparently tries the next model in the list.

    Real uses: primary = cheap/fast model, fallback = bigger model that handles
    harder inputs; or primary = OpenAI, fallback = Anthropic for provider outages.
    """
    primary = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    backup = ChatOpenAI(model="gpt-4o", temperature=0)  # different model as backup

    llm_with_fallback = primary.with_fallbacks([backup])

    print("=== with_fallbacks() ===")
    print(llm_with_fallback.invoke("Name a primary color.").content)
    print("(if 'primary' had failed, 'backup' would have answered)")
    print()


# ---------------------------------------------------------------------------
# 7. CACHING — identical prompt twice = no API call, instant, free
# ---------------------------------------------------------------------------
def demo_caching():
    """
    If you send the EXACT same prompt (and params) again, why pay again? An LLM
    cache stores responses keyed by the request. Second identical call returns
    instantly at zero cost. Huge for repeated questions, tests, and dev loops.

    Caveat: the key is the FULL request. Any difference (temperature, one extra
    space, a different model) = cache miss. And temperature>0 makes caching
    pointless because you actually want variety. Caching pairs with temperature=0.

    InMemoryCache here is per-process (gone on restart). Production uses Redis /
    SQLite so the cache survives restarts and is shared across servers.
    """
    from langchain_core.globals import set_llm_cache
    from langchain_core.caches import InMemoryCache

    set_llm_cache(InMemoryCache())  # turn caching ON globally

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = "What is the capital of France?"

    start = time.perf_counter()
    llm.invoke(prompt)             # real API call
    first = time.perf_counter() - start

    start = time.perf_counter()
    llm.invoke(prompt)             # served from cache — no API call
    second = time.perf_counter() - start

    print("=== caching ===")
    print(f"1st call (API)  : {first:.3f}s")
    print(f"2nd call (cache): {second:.3f}s  ← basically instant, $0")
    print()


# ---------------------------------------------------------------------------
# 8. REAL ERROR TYPES — what actually gets raised in production
# ---------------------------------------------------------------------------
def demo_real_errors():
    """
    Things you WILL see in production logs:
      - RateLimitError (429)        : too many requests — back off / retry
      - APITimeoutError             : call took too long — retry / lower load
      - BadRequestError (400)       : your request is malformed (bad model name,
                                      context too long, invalid params) — DON'T
                                      retry, it'll fail again; FIX the request
      - AuthenticationError (401)   : bad/missing API key
      - APIConnectionError          : network couldn't reach OpenAI

    Key distinction: retry the TRANSIENT ones (429, timeout, connection); do NOT
    retry the PERMANENT ones (400, 401) — fix the cause instead.

    Below we force a BadRequestError with a model name that doesn't exist, so you
    can see a real one get raised and caught.
    """
    print("=== real error types ===")
    broken = ChatOpenAI(model="gpt-this-model-does-not-exist", max_retries=0)
    try:
        broken.invoke("hello")
    except Exception as e:
        print(f"Caught: {type(e).__name__}")
        print(f"Message: {str(e)[:120]}...")
        print("→ This is a PERMANENT error. Retrying won't help — fix the request.")
    print()


if __name__ == "__main__":
    # Run ONE at a time — uncomment as you go.
    demo_model_parameters()
    # demo_determinism()
    # demo_token_usage_and_cost()
    # demo_init_chat_model()
    # demo_with_retry()
    # demo_with_fallbacks()
    # demo_caching()
    # demo_real_errors()
