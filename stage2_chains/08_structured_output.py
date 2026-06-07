"""
Topic 8: Structured Output & Tool Calling
------------------------------------------
Force the LLM to return a specific data shape every time — reliably.

THREE WAYS (naive → production):
  1. JsonOutputParser          — asks nicely via prompt text, LLM can ignore it
  2. PydanticOutputParser      — same, but adds Pydantic validation after
  3. with_structured_output()  — uses function calling at the API level, LLM is FORCED

PRODUCTION STANDARD: with_structured_output() with a Pydantic model.
You saw a taste of this in 03_output_parsers.py. This file goes deeper:
  - Nested models (objects within objects)
  - Optional fields and defaults
  - Enums for constrained values
  - Validation errors and how to handle them
  - Real extraction pipelines
  - The guardrail pattern (is_relevant flag — real production use)
"""

from enum import Enum
from typing import Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")


# ---------------------------------------------------------------------------
# 1. BASIC with_structured_output() — the production baseline
# ---------------------------------------------------------------------------
def demo_basic():
    """
    Simplest case — one flat Pydantic model. The LLM is forced to return
    this shape via function calling. No format_instructions in the prompt needed.
    """
    class PersonProfile(BaseModel):
        name: str = Field(description="Full name of the person")
        age: int = Field(description="Age in years")
        city: str = Field(description="City they live in")

    structured_llm = llm.with_structured_output(PersonProfile)
    result = structured_llm.invoke("Make up a random person.")

    print("=== basic structured output ===")
    print(type(result))     # PersonProfile — real Pydantic object
    print(result.name, result.age, result.city)
    print()


# ---------------------------------------------------------------------------
# 2. NESTED MODELS — objects inside objects
# ---------------------------------------------------------------------------
def demo_nested():
    """
    Real extraction tasks usually have nested structures.
    Pydantic handles nesting naturally — just reference one model from another.
    """
    class Address(BaseModel):
        street: str
        city: str
        country: str

    class Company(BaseModel):
        name: str = Field(description="Company name")
        industry: str = Field(description="The industry it operates in")
        founded_year: int = Field(description="Year the company was founded")
        headquarters: Address = Field(description="Physical headquarters address")
        public: bool = Field(description="Whether the company is publicly traded")

    structured_llm = llm.with_structured_output(Company)
    result = structured_llm.invoke("Tell me about Apple Inc.")

    print("=== nested models ===")
    print("Company :", result.name)
    print("Industry:", result.industry)
    print("Founded :", result.founded_year)
    print("HQ      :", result.headquarters.city, result.headquarters.country)
    print("Public  :", result.public)
    print()


# ---------------------------------------------------------------------------
# 3. OPTIONAL FIELDS — handle missing information gracefully
# ---------------------------------------------------------------------------
def demo_optional_fields():
    """
    Real-world text often doesn't have all the data you want.
    Optional fields let the model return None instead of hallucinating a value.

    This is critical for extraction tasks — you'd rather get None than a
    confident wrong answer.
    """
    class JobPosting(BaseModel):
        title: str = Field(description="Job title")
        company: str = Field(description="Company name")
        salary_min: Optional[int] = Field(default=None, description="Minimum salary in USD, if mentioned")
        salary_max: Optional[int] = Field(default=None, description="Maximum salary in USD, if mentioned")
        remote: Optional[bool] = Field(default=None, description="Whether the role is remote, if mentioned")
        experience_years: Optional[int] = Field(default=None, description="Years of experience required, if mentioned")

    structured_llm = llm.with_structured_output(JobPosting)

    # Sparse text — missing most fields
    sparse = "Google is hiring a Software Engineer."
    result = structured_llm.invoke(sparse)

    print("=== optional fields (sparse input) ===")
    print("Title  :", result.title)
    print("Company:", result.company)
    print("Salary :", result.salary_min, "-", result.salary_max)   # None - None
    print("Remote :", result.remote)                                # None
    print()

    # Rich text — has everything
    rich = ("Stripe is hiring a Senior Backend Engineer. Salary $160k-$200k. "
            "Remote OK. 5+ years experience required.")
    result2 = structured_llm.invoke(rich)

    print("=== optional fields (rich input) ===")
    print("Title     :", result2.title)
    print("Salary    :", result2.salary_min, "-", result2.salary_max)
    print("Remote    :", result2.remote)
    print("Experience:", result2.experience_years, "years")
    print()


# ---------------------------------------------------------------------------
# 4. ENUMS — constrain the model to a fixed set of values
# ---------------------------------------------------------------------------
def demo_enum():
    """
    Without an Enum, 'sentiment' might come back as "positive", "Positive",
    "good", "optimistic" — all meaning the same thing but breaking your downstream
    code that does if sentiment == "positive".

    Enum pins the model to your exact allowed values.
    """
    class Sentiment(str, Enum):
        positive = "positive"
        negative = "negative"
        neutral = "neutral"

    class Priority(str, Enum):
        low = "low"
        medium = "medium"
        high = "high"
        critical = "critical"

    class SupportTicket(BaseModel):
        summary: str = Field(description="One sentence summary of the issue")
        sentiment: Sentiment = Field(description="Customer sentiment")
        priority: Priority = Field(description="Ticket priority level")
        requires_human: bool = Field(description="Whether this needs a human agent")

    structured_llm = llm.with_structured_output(SupportTicket)

    ticket_text = ("I've been charged twice for my subscription and can't reach "
                   "anyone. This is unacceptable. I want a refund NOW.")

    result = structured_llm.invoke(ticket_text)

    print("=== enum constrained output ===")
    print("Summary       :", result.summary)
    print("Sentiment     :", result.sentiment)          # always "positive"/"negative"/"neutral"
    print("Priority      :", result.priority)           # always your 4 values
    print("Requires human:", result.requires_human)
    print()


# ---------------------------------------------------------------------------
# 5. GUARDRAIL PATTERN — is_relevant flag (production use case)
# ---------------------------------------------------------------------------
def demo_guardrail():
    """
    The model signals whether it can answer the question meaningfully,
    rather than hallucinating data to fill required fields.

    Production use: route irrelevant inputs early before hitting expensive
    downstream chains. One cheap structured call, fast decision.
    """
    class ExtractionResult(BaseModel):
        is_relevant: bool = Field(
            description="True if the input contains a person's profile information, False otherwise"
        )
        name: Optional[str] = Field(default=None, description="Person's name — only if is_relevant")
        age: Optional[int] = Field(default=None, description="Person's age — only if is_relevant")
        city: Optional[str] = Field(default=None, description="Person's city — only if is_relevant")

    structured_llm = llm.with_structured_output(ExtractionResult)

    def extract(text: str):
        result = structured_llm.invoke(text)
        print(f"Input: {text!r}")
        if not result.is_relevant:
            print("→ BLOCKED: not a person profile")
        else:
            print(f"→ Extracted: {result.name}, {result.age}, {result.city}")
        print()

    print("=== guardrail pattern ===")
    extract("John Smith, 34, lives in Bangalore.")
    extract("Tell me a joke about programmers.")
    extract("The weather in Chennai is 38°C today.")


# ---------------------------------------------------------------------------
# 6. EXTRACTION PIPELINE — structured output inside a full LCEL chain
# ---------------------------------------------------------------------------
def demo_extraction_pipeline():
    """
    Real production pattern: extract structured data from unstructured text,
    then use that data to drive the next step.

    Step 1: extract structured fields from raw text
    Step 2: use the extracted data to generate a formatted response
    """
    class ProductReview(BaseModel):
        product_name: str
        rating: int = Field(description="Rating from 1 to 5")
        pros: list[str] = Field(description="List of positive points")
        cons: list[str] = Field(description="List of negative points")
        recommend: bool

    extractor = llm.with_structured_output(ProductReview)

    response_prompt = ChatPromptTemplate.from_messages([
        ("human",
         "A customer reviewed '{product}' and gave it {rating}/5. "
         "Pros: {pros}. Cons: {cons}. Recommended: {recommend}.\n\n"
         "Write a 2-sentence response thanking them and addressing one con."),
    ])
    responder = response_prompt | llm | StrOutputParser()

    raw_review = (
        "I bought the Sony WH-1000XM5 last month. The noise cancellation is insane "
        "and battery life lasts forever. Build quality feels premium. Downside is "
        "the price — very expensive. Also the case is huge. Overall 4/5, would recommend."
    )

    extracted = extractor.invoke(raw_review)

    response = responder.invoke({
        "product": extracted.product_name,
        "rating": extracted.rating,
        "pros": ", ".join(extracted.pros),
        "cons": ", ".join(extracted.cons),
        "recommend": extracted.recommend,
    })

    print("=== extraction pipeline ===")
    print("Extracted:", extracted)
    print()
    print("Response:", response)
    print()


if __name__ == "__main__":
    demo_basic()
    demo_nested()
    demo_optional_fields()
    demo_enum()
    demo_guardrail()
    demo_extraction_pipeline()
