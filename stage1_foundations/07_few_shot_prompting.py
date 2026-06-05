"""
Topic 7: Few-Shot Prompting (teach the model BY EXAMPLE)
--------------------------------------------------------
Zero-shot = you only give INSTRUCTIONS ("classify the sentiment").
Few-shot  = you also give a handful of EXAMPLES of input → desired output.

The model is a pattern-matcher. Show it 3 examples of the exact format/style you
want and it copies that pattern far more reliably than instructions alone. This
is one of the cheapest, highest-leverage reliability tricks in production —
before you reach for fine-tuning, you reach for better examples.

WHAT THIS FILE COVERS:
  1. Zero-shot baseline — instructions only, watch the format drift
  2. Static few-shot — fixed examples that lock the format (FewShotChat...)
  3. Dynamic example selection — pick examples to FIT A TOKEN BUDGET
  4. Failure modes — examples cost tokens, bad/contradictory examples mislead
  5. When to use few-shot vs structured output vs fine-tuning

PRODUCTION REALITY:
  - More examples = more tokens = more cost on EVERY call. Examples aren't free.
  - The BEST examples are the ones most similar to the current input. That's
    "semantic example selection" (needs embeddings + a vector store — Stage 4).
    Here we use a length-based selector so it runs with zero extra setup.
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    PromptTemplate,
    FewShotPromptTemplate,
)
from langchain_core.example_selectors import LengthBasedExampleSelector
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = StrOutputParser()


# ---------------------------------------------------------------------------
# 1. ZERO-SHOT BASELINE — instructions only
# ---------------------------------------------------------------------------
def demo_zero_shot():
    """
    Just tell the model what to do. It works, but the FORMAT is unpredictable —
    sometimes 'Positive', sometimes 'positive sentiment', sometimes a sentence.
    That inconsistency is what breaks downstream parsing in production.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Classify the sentiment of the text as positive, negative, or neutral."),
        ("human", "{text}"),
    ])
    chain = prompt | llm | parser

    print("=== zero-shot (format may drift) ===")
    for t in ["I love this!", "It's fine I guess.", "Worst purchase ever."]:
        print(f"{t!r:30} -> {chain.invoke({'text': t})!r}")
    print()


# ---------------------------------------------------------------------------
# 2. STATIC FEW-SHOT — fixed examples lock the output format
# ---------------------------------------------------------------------------
def demo_static_few_shot():
    """
    Give 3 examples in EXACTLY the format you want back. The model copies it.
    Notice we never wrote 'reply with one lowercase word' — the EXAMPLES taught
    that. Examples beat instructions for enforcing format/style.

    FewShotChatMessagePromptTemplate expands each example into a human/ai message
    pair, then we slot the real question in after them.
    """
    examples = [
        {"input": "I love this!", "output": "positive"},
        {"input": "It's fine I guess.", "output": "neutral"},
        {"input": "Worst purchase ever.", "output": "negative"},
    ]

    # How ONE example is rendered (as a human turn + the ideal ai turn).
    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}"),
    ])

    few_shot = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )

    final_prompt = ChatPromptTemplate.from_messages([
        ("system", "Classify the sentiment."),
        few_shot,                 # the examples get injected here
        ("human", "{text}"),      # the real input
    ])
    chain = final_prompt | llm | parser

    print("=== static few-shot (format is now locked) ===")
    for t in ["This made my day.", "Meh, nothing special.", "I want a refund."]:
        print(f"{t!r:30} -> {chain.invoke({'text': t})!r}")
    print()


# ---------------------------------------------------------------------------
# 3. DYNAMIC EXAMPLE SELECTION — fit examples into a token budget
# ---------------------------------------------------------------------------
def demo_dynamic_selection():
    """
    You may have 100 examples but can't afford to send all 100 every call.
    An example SELECTOR chooses WHICH examples to include at runtime.

    LengthBasedExampleSelector packs in as many examples as fit under max_length,
    and quietly drops the rest — so a long user input automatically gets fewer
    examples (leaving room) and a short input gets more. This keeps you under the
    context limit and controls cost.

    The production-grade version is SemanticSimilarityExampleSelector: it embeds
    the input and picks the most RELEVANT examples (needs a vector store — Stage 4).
    Same idea, smarter selection.
    """
    examples = [
        {"input": "happy", "output": "positive"},
        {"input": "great", "output": "positive"},
        {"input": "terrible", "output": "negative"},
        {"input": "awful", "output": "negative"},
        {"input": "okay", "output": "neutral"},
        {"input": "average", "output": "neutral"},
    ]

    example_prompt = PromptTemplate(
        input_variables=["input", "output"],
        template="Word: {input}\nSentiment: {output}",
    )

    selector = LengthBasedExampleSelector(
        examples=examples,
        example_prompt=example_prompt,
        max_length=12,  # tiny budget on purpose → only a few examples survive
    )

    dynamic_prompt = FewShotPromptTemplate(
        example_selector=selector,
        example_prompt=example_prompt,
        prefix="Classify the sentiment of the word.",
        suffix="Word: {word}\nSentiment:",
        input_variables=["word"],
    )

    print("=== dynamic selection (examples chosen to fit budget) ===")
    rendered = dynamic_prompt.format(word="fantastic")
    print(rendered)
    print("--- model answer ---")
    print((dynamic_prompt | llm | parser).invoke({"word": "fantastic"}))
    print()


# ---------------------------------------------------------------------------
# 4. FAILURE MODES & WHEN TO USE WHAT (read this — no code)
# ---------------------------------------------------------------------------
def demo_when_to_use():
    """
    FAILURE MODES:
      - Examples cost tokens on EVERY call. 10 examples can dwarf the real input.
      - Contradictory or sloppy examples actively MISLEAD the model — one wrong
        example can poison the output. Curate examples like production data.
      - Too many examples can push you over the context window (see Topic 10).
      - The model may copy an example VERBATIM if the input is close to one.

    WHEN TO USE:
      - few-shot           : when FORMAT/STYLE/EDGE-CASES must be consistent, and
                             instructions alone aren't enough. Cheap, no training.
      - structured output  : when you need GUARANTEED JSON shape (use Topic 3's
                             with_structured_output) — often COMBINE with few-shot.
      - fine-tuning        : only when you have hundreds+ examples AND few-shot
                             still isn't enough AND the cost of long prompts hurts.
                             It's the last resort, not the first.

    Rule of thumb: instructions → few-shot → structured output → fine-tuning,
    in that order. Stop as soon as quality is good enough.
    """
    print(demo_when_to_use.__doc__)


if __name__ == "__main__":
    # Run ONE at a time — uncomment as you go.
    demo_zero_shot()
    # demo_static_few_shot()
    # demo_dynamic_selection()
    # demo_when_to_use()
