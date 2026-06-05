"""
Topic 10: Token Budgeting & Message Trimming (don't blow the context window)
----------------------------------------------------------------------------
Every model has a CONTEXT WINDOW — a hard limit on how many tokens (prompt +
response) fit in one call. Go over it and the API rejects the request outright.

This is THE failure mode that hits every growing app:
  - A chat conversation gets longer each turn → eventually too long → 400 error.
  - You stuff 20 retrieved documents into a RAG prompt → over the limit → 400.
  - A user pastes a huge blob of text → over the limit → 400.

A demo never hits this. A real app with real users hits it constantly. So you
budget tokens BEFORE sending, and TRIM history to stay under the limit.

WHAT THIS FILE COVERS:
  1. Counting tokens before you send (get_num_tokens)
  2. The context-window failure (conceptually — what the 400 looks like)
  3. trim_messages — keep a conversation under a token budget automatically
  4. Strategy choices: keep the LATEST vs keep the SYSTEM prompt
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    trim_messages,
)

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ---------------------------------------------------------------------------
# 1. COUNT TOKENS BEFORE SENDING
# ---------------------------------------------------------------------------
def demo_count_tokens():
    """
    Tokens are not words — roughly 1 token ≈ 4 characters of English. Counting
    BEFORE you send lets you: reject oversized input early, estimate cost, and
    decide whether to trim. get_num_tokens uses the model's real tokenizer.
    """
    short = "Hello, how are you?"
    long = "LangChain is a framework for building applications with LLMs. " * 20

    print("=== count tokens before sending ===")
    print(f"short text : {llm.get_num_tokens(short)} tokens")
    print(f"long text  : {llm.get_num_tokens(long)} tokens")

    messages = [
        SystemMessage("You are a helpful assistant."),
        HumanMessage("Summarise the history of Rome."),
    ]
    print(f"message list: {llm.get_num_tokens_from_messages(messages)} tokens")
    print()


# ---------------------------------------------------------------------------
# 2. THE CONTEXT-WINDOW FAILURE (what goes wrong)
# ---------------------------------------------------------------------------
def demo_context_failure():
    """
    There is no 'auto-truncate'. If prompt + expected response > the model's
    context window, the API returns a BadRequestError (400) like:

        "This model's maximum context length is N tokens, however your
         messages resulted in M tokens..."

    It's a PERMANENT error (Topic 6) — retrying sends the same too-big request
    and fails again. The fix is to send LESS: trim history, chunk the input, or
    summarise older turns. We won't actually send millions of tokens here (that'd
    just cost money) — the point is to know this error by sight and design for it.
    """
    print("=== context-window failure (conceptual) ===")
    print("Symptom : BadRequestError 'maximum context length is N tokens'")
    print("Cause   : prompt + response exceeds the model's window")
    print("Fix     : trim/chunk/summarise BEFORE sending — never just retry")
    print()


# ---------------------------------------------------------------------------
# 3. trim_messages — keep a conversation under a token budget
# ---------------------------------------------------------------------------
def demo_trim_messages():
    """
    The production answer for long chats: trim_messages drops old messages until
    the history fits a token budget. strategy="last" keeps the most RECENT
    messages (what the model needs most), dropping the oldest.

    This is the engine behind 'memory' in Stage 3 — memory that never trims will
    eventually overflow the context window. Trimming is what keeps it alive.
    """
    conversation = [
        SystemMessage("You are a helpful assistant."),
        HumanMessage("Hi, my name is Vengal."),
        AIMessage("Hello Vengal! How can I help?"),
        HumanMessage("I'm learning LangChain."),
        AIMessage("Great choice — where are you in it?"),
        HumanMessage("Stage 1. What's my name again?"),
    ]

    trimmed = trim_messages(
        conversation,
        max_tokens=40,              # tiny budget on purpose so trimming is visible
        token_counter=llm,          # use the model's real tokenizer to count
        strategy="last",            # keep the most recent messages
        include_system=True,        # always keep the system prompt
        start_on="human",           # the kept history should begin on a human turn
    )

    print("=== trim_messages (fit history into a budget) ===")
    print(f"before: {len(conversation)} messages, "
          f"{llm.get_num_tokens_from_messages(conversation)} tokens")
    print(f"after : {len(trimmed)} messages, "
          f"{llm.get_num_tokens_from_messages(trimmed)} tokens")
    print("kept:")
    for m in trimmed:
        print(f"  {m.type:9}: {m.content}")
    print()


# ---------------------------------------------------------------------------
# 4. STRATEGY: keep latest vs keep system — and the tradeoff
# ---------------------------------------------------------------------------
def demo_trim_tradeoff():
    """
    TRADEOFF you must own:
      - Trimming OLD messages = the bot FORGETS early context (the user's name,
        an earlier instruction). Cheap and simple, but lossy.
      - The alternative is to SUMMARISE old turns into one short message instead
        of dropping them — keeps the gist, costs an extra LLM call. (Stage 3.)
      - ALWAYS keep the system prompt (include_system=True) — it holds the rules;
        dropping it changes the bot's whole behaviour mid-conversation.

    Rule: trim the middle/old, protect the system prompt and the latest turns.
    Decide consciously what the app is allowed to forget.
    """
    print(demo_trim_tradeoff.__doc__)


if __name__ == "__main__":
    # Run ONE at a time — uncomment as you go.
    demo_count_tokens()
    # demo_context_failure()
    # demo_trim_messages()
    # demo_trim_tradeoff()
