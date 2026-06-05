"""
Topic 4: Output Parsers
------------------------
By default the LLM returns an AIMessage. Parsers transform that into something
your code can actually use: a plain string, a Python dict, or a typed Pydantic model.

StrOutputParser     → AIMessage → plain string
JsonOutputParser    → AIMessage → Python dict  (asks LLM to produce JSON)
PydanticOutputParser → AIMessage → typed Pydantic object
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")


def demo_str_output_parser():
    """StrOutputParser strips the AIMessage wrapper — you get a plain str back."""
    prompt = ChatPromptTemplate.from_messages([
        ("human", "What is the largest planet in the solar system?"),
    ])
    parser = StrOutputParser()

    # Without parser → AIMessage
    raw = llm.invoke(prompt.format_messages())
    print("=== Without parser ===")
    print(type(raw), raw.content[:60])

    # With parser → str
    chain = prompt | llm | parser
    result = chain.invoke({})
    print("\n=== With StrOutputParser ===")
    print(type(result), result[:60])
    print()


def demo_json_output_parser():
    """JsonOutputParser — tell the LLM to respond in JSON and parse it into a dict."""
    parser = JsonOutputParser()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Always respond with valid JSON. No extra text."),
        ("human", "Give me a person's profile: name, age, city. Make one up.\n{format_instructions}"),
    ])

    chain = prompt | llm | parser
    result = chain.invoke({"format_instructions": parser.get_format_instructions()})

    print("=== JsonOutputParser ===")
    print("Type:", type(result))
    print("Dict:", result)
    print("Name:", result.get("name"))
    print()


def demo_pydantic_output_parser():
    """PydanticOutputParser — define the exact shape you want with type safety."""
    from langchain_core.output_parsers import PydanticOutputParser

    class MovieReview(BaseModel):
        title: str = Field(description="The movie title")
        rating: float = Field(description="Rating from 0.0 to 10.0")
        summary: str = Field(description="One sentence summary of the review")
        recommended: bool = Field(description="Whether you recommend watching it")

    parser = PydanticOutputParser(pydantic_object=MovieReview)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a movie critic. {format_instructions}"),
        ("human", "Write a review for the movie: {movie}"),
    ])

    chain = prompt | llm | parser
    result = chain.invoke({
        "movie": "Inception",
        "format_instructions": parser.get_format_instructions(),
    })

    print("=== PydanticOutputParser ===")
    print("Type:", type(result))
    print("Title:", result.title)
    print("Rating:", result.rating)
    print("Summary:", result.summary)
    print("Recommended:", result.recommended)
    print()


def demo_json_with_schema():
    """JsonOutputParser with a Pydantic schema — combines type hints + dict output."""
    class WeatherReport(BaseModel):
        city: str = Field(description="The city name")
        temperature_celsius: int = Field(description="Temperature in Celsius")
        condition: str = Field(description="Weather condition e.g. sunny, rainy")

    parser = JsonOutputParser(pydantic_object=WeatherReport)

    prompt = ChatPromptTemplate.from_messages([
        ("human", "Make up a weather report for Tokyo.\n{format_instructions}"),
    ])

    chain = prompt | llm | parser
    result = chain.invoke({"format_instructions": parser.get_format_instructions()})

    print("=== JsonOutputParser with schema ===")
    print(result)
    print()


def demo_with_structured_output():
    """
    with_structured_output() — the production way.
    Uses function calling under the hood so the model is FORCED to return
    the right shape at the API level. No format instructions needed in the prompt.

    is_relevant flag lets the model signal when the input doesn't make sense
    for the task — instead of force-fitting garbage data into the fields.
    """
    class PersonProfile(BaseModel):
        is_relevant: bool = Field(description="True if the input is asking for a person's profile, False otherwise")
        name: str | None = Field(default=None, description="The person's name — only if is_relevant is True")
        age: int | None = Field(default=None, description="The person's age — only if is_relevant is True")
        city: str | None = Field(default=None, description="The city they live in — only if is_relevant is True")

    structured_llm = llm.with_structured_output(PersonProfile)

    def show_result(label: str, result: PersonProfile):
        print(f"=== {label} ===")
        if not result.is_relevant:
            print("Status: IRRELEVANT — input is not a person profile request")
        else:
            print("Name:", result.name)
            print("Age :", result.age)
            print("City:", result.city)
        print()

    show_result("Relevant prompt", structured_llm.invoke("Make up a random person's profile."))
    show_result("Irrelevant prompt", structured_llm.invoke("Tell me a joke."))


if __name__ == "__main__":
    demo_str_output_parser()
    demo_json_output_parser()
    demo_pydantic_output_parser()
    demo_json_with_schema()
    demo_with_structured_output()
