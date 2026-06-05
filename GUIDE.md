# LangChain Study Guide
### Your step-by-step execution plan

---

## Folder Structure to Create

```
langChain/
├── ROADMAP.md              ← topic checklist
├── GUIDE.md                ← this file
├── .env                    ← API keys (never commit this)
├── Dockerfile
├── run.sh
├── start.sh
│
├── stage1_foundations/
│   ├── 01_llms_chat_models.py
│   ├── 02_prompt_templates.py
│   ├── 03_output_parsers.py
│   ├── 04_lcel.py
│   └── 05_async_chains.py
│
├── stage2_chains/
│   ├── 06_sequential_chains.py
│   ├── 07_runnable_passthrough.py
│   ├── 08_structured_output.py
│   └── 09_async_chains.py
│
├── stage3_memory/
│   ├── 10_conversation_history.py
│   ├── 11_buffer_memory.py
│   └── 12_persistent_memory.py
│
├── stage4_rag/
│   ├── 13_document_loaders.py
│   ├── 14_text_splitters.py
│   ├── 15_embeddings.py
│   ├── 16_vector_stores.py
│   ├── 17_basic_rag_chain.py
│   ├── 18_advanced_rag.py
│   └── 19_rag_evaluation.py
│
├── stage5_agents/
│   ├── 20_tools.py
│   ├── 21_builtin_tools.py
│   ├── 22_agents.py
│   └── 23_agent_executor.py
│
├── stage6_production/
│   ├── 24_streaming.py
│   ├── 25_langserve.py
│   ├── 26_langsmith.py
│   ├── 27_callbacks.py
│   ├── 28_error_handling.py
│   ├── 29_cost_management.py
│   └── 30_testing_chains.py
│
└── projects/
    ├── A_cli_chatbot/
    ├── B_rag_pdf/
    ├── C_agent_web_search/
    └── D_rag_api/
```

---

## Stage 1 — Foundations

### Topic 1: LLMs & Chat Models
**What you will learn:** How to talk to an LLM using LangChain. The difference between `LLM` and `ChatModel`. What the response object looks like.
**What to code:**
- Call `ChatOpenAI` with a plain string
- Call it with a `HumanMessage`
- Print `response.content`
- Try different models (`gpt-4o`, `gpt-3.5-turbo`)
**File:** `stage1_foundations/01_llms_chat_models.py`

---

### Topic 2: Prompt Templates
**What you will learn:** How to build reusable prompts with variables instead of hardcoding strings.
**What to code:**
- `PromptTemplate` with one variable
- `ChatPromptTemplate` with system + human messages
- Format a prompt and print it before sending to LLM
**File:** `stage1_foundations/02_prompt_templates.py`

---

### Topic 3: Output Parsers
**What you will learn:** How to parse the LLM response into clean usable formats (string, JSON, Pydantic model).
**What to code:**
- `StrOutputParser` — get plain string back
- `JsonOutputParser` — get a dict back
- `PydanticOutputParser` — get a typed object back
**File:** `stage1_foundations/03_output_parsers.py`

---

### Topic 4: LCEL (LangChain Expression Language)
**What you will learn:** The `|` pipe syntax that chains steps together. The backbone of everything in LangChain.
**What to code:**
- `prompt | llm | parser` — a basic 3-step chain
- `.invoke()` — run once
- `.stream()` — stream tokens
- `.batch()` — run multiple inputs at once
**File:** `stage1_foundations/04_lcel.py`

---

### Topic 5: Async Chains
**What you will learn:** How to run chains asynchronously. Critical for web apps and APIs.
**What to code:**
- `await chain.ainvoke()`
- `async for chunk in chain.astream()`
**File:** `stage1_foundations/05_async_chains.py`

---

## Stage 2 — Chains

### Topic 6: Sequential Chains
**What you will learn:** How to pass the output of one chain as the input to the next.
**What to code:**
- Chain 1 generates a topic summary
- Chain 2 takes that summary and writes a tweet
- Wire them together with LCEL
**File:** `stage2_chains/06_sequential_chains.py`

---

### Topic 7: RunnablePassthrough & RunnableMap
**What you will learn:** How to pass original input alongside transformed data through a chain.
**What to code:**
- `RunnablePassthrough` — pass input unchanged
- `RunnableMap` — run multiple chains in parallel and merge results
**File:** `stage2_chains/07_runnable_passthrough.py`

---

### Topic 8: Structured Output & Tool Calling
**What you will learn:** How to force the LLM to return a specific structure every time using Pydantic.
**What to code:**
- Define a Pydantic model (e.g. `Person` with name, age, city)
- Use `llm.with_structured_output(Person)`
- Extract structured data from a sentence
**File:** `stage2_chains/08_structured_output.py`

---

### Topic 9: Async Chains (advanced)
**What you will learn:** Batch async calls, concurrency, real-world async patterns.
**What to code:**
- `asyncio.gather` with multiple `ainvoke` calls
- Streaming in an async context
**File:** `stage2_chains/09_async_chains.py`

---

## Stage 3 — Memory

### Topic 10: Conversation History
**What you will learn:** How to manually pass chat history so the LLM remembers previous messages.
**What to code:**
- Build a list of `HumanMessage` and `AIMessage`
- Pass the full list to the LLM each turn
- See how context changes the response
**File:** `stage3_memory/10_conversation_history.py`

---

### Topic 11: Conversation Buffer Memory
**What you will learn:** How LangChain manages history automatically using `RunnableWithMessageHistory`.
**What to code:**
- Set up `ChatMessageHistory`
- Wrap a chain with `RunnableWithMessageHistory`
- Have a multi-turn conversation with a session ID
**File:** `stage3_memory/11_buffer_memory.py`

---

### Topic 12: Persistent Memory
**What you will learn:** How to store chat history in a database so it survives restarts.
**What to code:**
- Store history in a SQLite file (simple)
- Reload history by session ID on startup
**File:** `stage3_memory/12_persistent_memory.py`

---

## Stage 4 — RAG

### Topic 13: Document Loaders
**What you will learn:** How to load data from different sources into LangChain.
**What to code:**
- `TextLoader` — load a `.txt` file
- `PyPDFLoader` — load a PDF
- `WebBaseLoader` — load a webpage
**File:** `stage4_rag/13_document_loaders.py`

---

### Topic 14: Text Splitters
**What you will learn:** How to break large documents into smaller chunks for embedding.
**What to code:**
- `RecursiveCharacterTextSplitter` with chunk size and overlap
- Print the chunks and see how the splitting works
**File:** `stage4_rag/14_text_splitters.py`

---

### Topic 15: Embeddings
**What you will learn:** What embeddings are (numbers that represent meaning), and how to create them.
**What to code:**
- `OpenAIEmbeddings` — embed a sentence
- Embed two similar and two different sentences
- Compare similarity manually
**File:** `stage4_rag/15_embeddings.py`

---

### Topic 16: Vector Stores
**What you will learn:** How to store embeddings and search them by similarity.
**What to code:**
- Store chunks in FAISS (local, no setup needed)
- Run a similarity search query
- Retrieve top 3 matching chunks
**File:** `stage4_rag/16_vector_stores.py`

---

### Topic 17: Basic RAG Chain
**What you will learn:** How to combine a retriever + LLM to answer questions from your own documents.
**What to code:**
- Load a PDF → split → embed → store in FAISS
- Build a retrieval chain with `create_retrieval_chain`
- Ask questions and see answers grounded in the doc
**File:** `stage4_rag/17_basic_rag_chain.py`

---

### Topic 18: Advanced RAG
**What you will learn:** How to improve RAG quality beyond the basics.
**What to code:**
- Hybrid search (keyword + vector with `EnsembleRetriever`)
- Reranking results with `CohereRerank`
- Parent-child chunking
**File:** `stage4_rag/18_advanced_rag.py`

---

### Topic 19: RAG Evaluation
**What you will learn:** How to measure if your RAG system is actually giving good answers.
**What to code:**
- Build a small Q&A eval dataset
- Run questions through your RAG chain
- Score answers using an LLM as judge
**File:** `stage4_rag/19_rag_evaluation.py`

---

## Stage 5 — Agents & Tools

### Topic 20: Tools
**What you will learn:** How to give the LLM custom abilities (functions it can call).
**What to code:**
- Define a tool with `@tool` decorator
- Tool that fetches weather for a city (mock)
- Tool that does a calculation
**File:** `stage5_agents/20_tools.py`

---

### Topic 21: Built-in Tools
**What you will learn:** Tools that LangChain already provides out of the box.
**What to code:**
- `DuckDuckGoSearchRun` — web search
- `PythonREPLTool` — run Python code
- `WikipediaQueryRun` — search Wikipedia
**File:** `stage5_agents/21_builtin_tools.py`

---

### Topic 22: Agents
**What you will learn:** How the LLM decides which tool to use and when (ReAct loop).
**What to code:**
- Understand the think → act → observe loop
- Build a simple agent with 2 tools
- Watch the reasoning steps in the output
**File:** `stage5_agents/22_agents.py`

---

### Topic 23: AgentExecutor
**What you will learn:** How to run an agent properly with error handling, max iterations, verbose output.
**What to code:**
- Wrap agent in `AgentExecutor`
- Set `max_iterations`, `handle_parsing_errors`
- Run a multi-step task that requires multiple tool calls
**File:** `stage5_agents/23_agent_executor.py`

---

## Stage 6 — Production

### Topic 24: Streaming
**What you will learn:** How to stream tokens to the user in real time instead of waiting for the full response.
**What to code:**
- Stream a chain response token by token
- Stream an agent's intermediate steps
**File:** `stage6_production/24_streaming.py`

---

### Topic 25: LangServe
**What you will learn:** How to deploy any LangChain chain as a REST API with one file.
**What to code:**
- Wrap a chain with `add_routes`
- Run it with FastAPI
- Hit the `/invoke` endpoint with curl
**File:** `stage6_production/25_langserve.py`

---

### Topic 26: LangSmith
**What you will learn:** How to trace every LLM call, debug failures, and compare runs.
**What to code:**
- Set up LangSmith API key
- Run any chain and see the full trace in the dashboard
- Add custom metadata to a run
**File:** `stage6_production/26_langsmith.py`

---

### Topic 27: Callbacks & Observability
**What you will learn:** How to hook into chain execution to log, monitor, or track anything.
**What to code:**
- Custom callback that logs every LLM call
- Track token usage and cost per run
**File:** `stage6_production/27_callbacks.py`

---

### Topic 28: Error Handling & Retries
**What you will learn:** How to make chains resilient to failures.
**What to code:**
- `.with_fallbacks()` — fallback to a different model if primary fails
- `.with_retry()` — auto retry on rate limit errors
**File:** `stage6_production/28_error_handling.py`

---

### Topic 29: Cost & Token Management
**What you will learn:** How to track and control how much you're spending on API calls.
**What to code:**
- Count tokens before sending with `get_num_tokens`
- Enable LLM caching with `set_llm_cache` (same input = no API call)
- Log cost per chain run
**File:** `stage6_production/29_cost_management.py`

---

### Topic 30: Testing Chains
**What you will learn:** How to write tests for your LangChain code so it doesn't break silently.
**What to code:**
- Mock an LLM response in a unit test
- Test that a prompt template formats correctly
- Test a full chain with a fake LLM
**File:** `stage6_production/30_testing_chains.py`

---

## Milestone Projects

### Project A — CLI Chatbot (after Stage 3)
A terminal chatbot that remembers the full conversation.
- Multi-turn conversation
- Persistent memory (saves history to file)
- Clean exit with `quit`

### Project B — RAG over a PDF (after Stage 4)
Ask questions to any PDF you upload.
- Load + chunk + embed a PDF
- Answer questions from it
- Show which part of the doc the answer came from

### Project C — Agent with Web Search + Code Execution (after Stage 5)
An agent that can look things up and run code to solve problems.
- DuckDuckGo search tool
- Python REPL tool
- Handles multi-step tasks

### Project D — Production RAG API (after Stage 6)
A fully production-ready RAG system.
- FastAPI + LangServe endpoint
- Streaming responses
- LangSmith tracing
- Error handling + retries

---

## Rules While Studying

1. **Code every topic** — don't just read, type it out
2. **One file per topic** — keep it clean and referenceable
3. **Break things on purpose** — change inputs, see what errors look like
4. **Check off ROADMAP.md** after each topic
5. **Don't move to the next stage** until the milestone project works
