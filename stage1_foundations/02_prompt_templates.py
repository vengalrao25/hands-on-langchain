"""
Topic 3: Prompt Templates
--------------------------
Stop hardcoding prompts. Templates let you define a prompt once with placeholders,
then fill them in at runtime — reusable, testable, composable.

PromptTemplate     → plain string prompt with {variable} placeholders
ChatPromptTemplate → list of (role, template) pairs — the standard for chat models
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")


def demo_prompt_template():
    """PromptTemplate with one variable — produces a plain string prompt."""
    template = PromptTemplate(
        input_variables=["topic"],
        template="Explain {topic} in simple terms, like I'm 10 years old.",
    )

    # Format it first — you can inspect the prompt before sending
    formatted = template.format(topic="machine learning")
    print("=== Formatted prompt ===")
    print(formatted)
    print()

    # Send to LLM
    response = llm.invoke(formatted)
    print("=== LLM response ===")
    print(response.content)
    print()


def demo_chat_prompt_template():
    """ChatPromptTemplate — system + human messages with variables."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert in {domain}. Be concise."),
        ("human", "What is {concept}?"),
    ])

    # .format_messages() returns a list of BaseMessage — inspect before sending
    messages = prompt.format_messages(domain="databases", concept="indexing")
    print("=== Formatted messages ===")
    for msg in messages:
        print(f"  [{msg.type}] {msg.content}")
    print()

    # Send to LLM
    response = llm.invoke(messages)
    print("=== LLM response ===")
    print(response.content)
    print()


def demo_multiple_variables():
    """Template with multiple variables — useful for structured prompts."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a code reviewer. Language: {language}. Be direct."),
        ("human", "Review this code and find bugs:\n\n{code}"),
    ])

    code_snippet = """
def divide(a, b):
    return a / b
"""

    messages = prompt.format_messages(language="Python", code=code_snippet)
    response = llm.invoke(messages)

    print("=== Code review ===")
    print(response.content)
    print()


def demo_partial_format():
    """Partially fill a template — lock in some variables, leave others open."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a translator. Target language: {language}."),
        ("human", "Translate: {text}"),
    ])

    # Fix the language, leave text open
    spanish_translator = prompt.partial(language="Spanish")

    response = llm.invoke(spanish_translator.format_messages(text="Hello, how are you?"))
    print("=== Partial template (Spanish) ===")
    print(response.content)
    print()


if __name__ == "__main__":
    demo_prompt_template()
    demo_chat_prompt_template()
    demo_multiple_variables()
    demo_partial_format()
