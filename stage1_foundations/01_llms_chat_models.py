"""
Topic 2: LLMs & Chat Models
----------------------------
ChatOpenAI is a Chat Model — it speaks in messages (system, human, AI).
The response is an AIMessage object, not a plain string. Use .content to get the text.

Key difference:
  LLM  → takes a raw string, returns a string  (older style)
  Chat → takes a list of messages, returns an AIMessage  (current standard)
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()


def demo_plain_string():
    """Simplest possible call — pass a string, get .content back."""
    llm = ChatOpenAI(model="gpt-4o-mini")
    response = llm.invoke("What is the capital of France?")

    print("=== Plain string invoke ===")
    print("Type:", type(response))          # AIMessage
    print("Content:", response.content)
    print("Model used:", response.response_metadata.get("model_name"))
    print()
    print(response)


def demo_human_message():
    """Explicit HumanMessage — same result, but you control the type."""
    llm = ChatOpenAI(model="gpt-4o-mini")
    response = llm.invoke([HumanMessage(content="Name three programming languages.")])

    print("=== HumanMessage invoke ===")
    print(response.content)
    print()


def demo_system_plus_human():
    """System message sets the LLM's persona/rules. Human message is the actual prompt."""
    llm = ChatOpenAI(model="gpt-4o-mini")
    messages = [
        SystemMessage(content="You are a pirate. Respond in pirate speak."),
        HumanMessage(content="What is 2 + 2?"),
    ]
    response = llm.invoke(messages)

    print("=== System + Human messages ===")
    print(response.content)
    print()


def demo_model_comparison():
    """Same prompt, two models — compare quality vs cost."""
    prompt = [HumanMessage(content="Explain recursion in one sentence.")]

    for model_name in ["gpt-4o-mini", "gpt-4o"]:
        llm = ChatOpenAI(model=model_name)
        response = llm.invoke(prompt)
        print(f"=== {model_name} ===")
        print(response.content)
        print()


def demo_response_metadata():
    """Inspect the full response object — tokens used, model, finish reason."""
    llm = ChatOpenAI(model="gpt-4o-mini")
    response = llm.invoke("Say hello.")

    print("=== Full response object ===")
    print("content       :", response.content)
    print("type          :", response.type)           # 'ai'
    print("usage_metadata:", response.usage_metadata) # prompt/completion tokens
    print("finish_reason :", response.response_metadata.get("finish_reason"))
    print()


if __name__ == "__main__":
    # demo_plain_string()
    # demo_human_message()
    # demo_system_plus_human()
    # demo_model_comparison()
    demo_response_metadata()
