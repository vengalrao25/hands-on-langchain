"""
Topic 9: Runtime Config & .bind() (change behaviour WITHOUT rebuilding chains)
------------------------------------------------------------------------------
You build a chain once. But real apps need to tweak HOW it runs per request:
a creative response here, a deterministic one there; a cheap model for simple
inputs, a powerful one for hard inputs; stop generation at a delimiter.

Rebuilding the chain every time is wasteful and messy. LangChain gives you three
tools to change behaviour at call time on the SAME chain object:

  1. .bind()                    — lock in fixed args (e.g. stop sequences)
  2. .configurable_fields()     — expose a setting to change per-call
  3. .configurable_alternatives() — swap a whole component (e.g. the model)

WHY IT MATTERS IN PRODUCTION:
  - One chain, many behaviours = less code, fewer bugs, easy A/B testing.
  - Route cheap vs expensive models by input difficulty → real cost savings.
  - Flip a setting from a request/config without redeploying.
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import ConfigurableField

load_dotenv()

parser = StrOutputParser()
prompt = ChatPromptTemplate.from_messages([("human", "{question}")])


# ---------------------------------------------------------------------------
# 1. .bind() — pin fixed arguments onto a runnable
# ---------------------------------------------------------------------------
def demo_bind():
    """
    .bind() returns a NEW runnable with some arguments pre-set. Classic use:
    'stop' sequences — tell the model to stop generating as soon as it produces
    a given string. Here we stop at "3.", so we only ever get items 1 and 2.

    .bind() is also how tools get attached to a model (llm.bind_tools([...])) —
    you'll see that again in Stage 5 Agents.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    capped = llm.bind(stop=["3."])  # stop the moment the model writes "3."

    print("=== .bind(stop=...) ===")
    print(capped.invoke("List benefits of exercise as 1. 2. 3. 4.").content)
    print("(generation halted at '3.' — only items 1 and 2 came through)")
    print()


# ---------------------------------------------------------------------------
# 2. .configurable_fields() — expose a setting to override per call
# ---------------------------------------------------------------------------
def demo_configurable_fields():
    """
    Mark temperature as configurable ONCE, then override it per call with
    .with_config(configurable={...}). Same chain object → different behaviour.
    No rebuilding, no second chain. Great for "creative mode" toggles.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).configurable_fields(
        temperature=ConfigurableField(
            id="temperature",
            name="LLM Temperature",
            description="Higher = more creative",
        )
    )
    chain = prompt | llm | parser

    print("=== configurable_fields (override temperature per call) ===")
    print("default (0):", chain.invoke({"question": "Name a startup in one word."}))
    creative = chain.with_config(configurable={"temperature": 1.3})
    print("temp 1.3   :", creative.invoke({"question": "Name a startup in one word."}))
    print()


# ---------------------------------------------------------------------------
# 3. .configurable_alternatives() — swap a whole component (the model)
# ---------------------------------------------------------------------------
def demo_configurable_alternatives():
    """
    Define a DEFAULT model plus named ALTERNATIVES, then pick one per call.
    This is the foundation of model routing: send easy/cheap requests to the mini
    model and hard ones to the big model — from a single chain.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).configurable_alternatives(
        ConfigurableField(id="model"),
        default_key="mini",
        powerful=ChatOpenAI(model="gpt-4o", temperature=0),  # the alternative
    )
    chain = prompt | llm | parser

    q = {"question": "Reply with exactly one word: hello"}

    print("=== configurable_alternatives (swap the model per call) ===")
    print("default (mini):", chain.invoke(q))
    print("powerful (4o) :", chain.with_config(configurable={"model": "powerful"}).invoke(q))
    print()


if __name__ == "__main__":
    # Run ONE at a time — uncomment as you go.
    demo_bind()
    # demo_configurable_fields()
    # demo_configurable_alternatives()
