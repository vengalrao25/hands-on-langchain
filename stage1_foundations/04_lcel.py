"""
Topic 5: LCEL — LangChain Expression Language
-----------------------------------------------
The | pipe syntax is the backbone of LangChain. Every chain you build uses it.

prompt | llm | parser   ← a 3-step pipeline

Three ways to run a chain:
  .invoke()  → run once, wait for full response
  .stream()  → get tokens as they arrive (real-time)
  .batch()   → run multiple inputs at once (parallel)
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Be concise."),
    ("human", "{question}"),
])

# The chain — built once, reused everywhere
chain = prompt | llm | parser


def demo_invoke():
    """Run once — waits for the full response, returns a string."""
    result = chain.invoke({"question": "What is Python?"})

    print("=== .invoke() ===")
    print(result)
    print()


def demo_stream():
    """Stream tokens as they arrive — you see output word by word in real time."""
    print("=== .stream() ===")

    for chunk in chain.stream({"question": "Explain how the internet works in 3 sentences."}):
        print(chunk, end="", flush=True)  # flush=True prints immediately without buffering

    print("\n")


def demo_batch():
    """Run multiple inputs at once — LangChain sends them in parallel."""
    questions = [
        {"question": "What is JavaScript?"},
        {"question": "What is a database?"},
        {"question": "What is an API?"},
    ]

    results = chain.batch(questions)

    print("=== .batch() ===")
    for q, r in zip(questions, results):
        print(f"Q: {q['question']}")
        print(f"A: {r[:80]}...")
        print()


def demo_chain_inspection():
    """You can inspect what a chain is made of before running it."""
    print("=== Chain steps ===")
    print(chain.steps)  # [ChatPromptTemplate, ChatOpenAI, StrOutputParser]
    print()


def demo_nested_chain():
    """
    Chains can be combined — output of one chain feeds into the next.
    Chain 1: topic → summary
    Chain 2: summary → tweet
    """
    summary_prompt = ChatPromptTemplate.from_messages([
        ("human", "Summarise this topic in one sentence: {topic}"),
    ])

    tweet_prompt = ChatPromptTemplate.from_messages([
        ("human", "Write a tweet about this: {summary}"),
    ])

    # Two separate chains
    summary_chain = summary_prompt | llm | parser
    tweet_chain = tweet_prompt | llm | parser

    # Wire them together — summary becomes input to tweet_chain
    full_chain = summary_chain | (lambda summary: tweet_chain.invoke({"summary": summary}))
    # print('chain.steps ' ,full_chain.steps)
    result = full_chain.invoke({"topic": "quantum computing"})

    print("=== Nested chain (summary → tweet) ===")
    print(result)
    print()


if __name__ == "__main__":
    # demo_invoke()
    # demo_stream()
    # demo_batch()
    # demo_chain_inspection()
    demo_nested_chain()
