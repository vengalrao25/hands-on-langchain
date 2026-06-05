FROM python:3.12-slim

WORKDIR /app

RUN pip install langchain langchain-openai langchain-core python-dotenv

ENTRYPOINT ["python"]
