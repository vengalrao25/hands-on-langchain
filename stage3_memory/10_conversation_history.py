"""
Topic 10: Conversation History (the manual way — understand what "memory" IS)
-----------------------------------------------------------------------------
Core truth you already know: the LLM is STATELESS. It remembers nothing between
calls. "Memory" is an illusion you create by RESENDING the past conversation as
a list of messages on every single call.

    Turn 1: send [Human: "I'm Vengal"]                    -> AI: "Hi Vengal"
    Turn 2: send [Human: "I'm Vengal", AI: "Hi Vengal",   -> AI: "You're Vengal"
                  Human: "what's my name?"]
            ^ you resend turn 1 so the model "remembers"

Everything fancier (RunnableWithMessageHistory, LangGraph memory) is just
automation of THIS. Build it by hand once and the rest will never be magic.

WHAT THIS FILE COVERS:
  1. Statelessness proof — model forgets when you DON'T pass history
  2. Manual history list — the real mechanism behind all "memory"
  3. MessagesPlaceholder — slot history into a prompt template cleanly
  4. The failure mode — history grows forever → cost + context overflow
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ---------------------------------------------------------------------------
# 1. PROOF: no history = no memory
# ---------------------------------------------------------------------------
def demo_stateless():
    """
    Two separate calls. The second has NO idea the first happened. This is the
    default reality of every LLM — there is no hidden server-side memory.
    """
    print("=== stateless (model forgets) ===")
    print(llm.invoke("My name is Vengal.").content)
    print(llm.invoke("What is my name?").content)  # it does NOT know
    print()


# ---------------------------------------------------------------------------
# 2. THE MECHANISM: a growing list of messages
# ---------------------------------------------------------------------------
def demo_manual_history():
    """
    Keep a Python list of messages. Each turn: append the human message, call
    the model with the WHOLE list, then append the AI's reply. That list IS the
    memory. Nothing more sophisticated is happening under any framework.
    """
    history = [SystemMessage("You are a friendly assistant.")]

    def turn(user_text: str):
        history.append(HumanMessage(user_text))
        response = llm.invoke(history)      # send the entire history every time
        history.append(response)            # response is an AIMessage
        return response.content

    print("=== manual history (real memory mechanism) ===")
    print("User: My name is Vengal and I love Python.")
    print("AI  :", turn("My name is Vengal and I love Python."))
    print("User: What's my name and what do I love?")
    print("AI  :", turn("What's my name and what do I love?"))  # now it knows
    print(f"\n(history is now {len(history)} messages long)")
    print()
    print("\n\n\n\n\n" , history)


# ---------------------------------------------------------------------------
# 3. MessagesPlaceholder — inject history into a prompt template
# ---------------------------------------------------------------------------
def demo_messages_placeholder():
    """
    In a real app the prompt has a fixed system message + dynamic history + the
    new input. MessagesPlaceholder is the slot where the history list gets
    injected. This is the prompt shape EVERY memory system uses.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are concise."),
        MessagesPlaceholder("history"),   # the running conversation goes here
        ("human", "{input}"),
    ])
    chain = prompt | llm

    history = [
        HumanMessage("I'm learning LangChain."),
        AIMessage("Nice — which stage are you on?"),
    ]

    result = chain.invoke({"history": history, "input": "Remind me what I'm learning."})
    print("=== MessagesPlaceholder ===")
    print(result.content)
    print()


# ---------------------------------------------------------------------------
# 4. FAILURE MODE: history grows without bound
# ---------------------------------------------------------------------------
def demo_growth_problem():
    """
    Manual history has a fatal flaw at scale: it grows FOREVER. Every turn you
    resend the entire conversation, so:
      - cost rises each turn (you pay for ALL past tokens again, every call)
      - latency rises (bigger prompt = slower)
      - eventually you blow the context window → BadRequestError (Topic 10, Stage 1)

    This is exactly why production memory must TRIM or SUMMARISE (you built
    trim_messages in stage1/10_token_budgeting). A naive ever-growing list is a
    demo, not a product. The next files automate history AND its trimming.
    """
    print(demo_growth_problem.__doc__)


if __name__ == "__main__":
    # Run ONE at a time — uncomment as you go.
    # demo_stateless()
    demo_manual_history()
    # demo_messages_placeholder()
    # demo_growth_problem()
