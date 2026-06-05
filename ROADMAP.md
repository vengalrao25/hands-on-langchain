# LangChain Learning Roadmap
### Goal: Production-Level AI Engineer

Work through these topics in order. Check off each one as you complete it.

---

## Stage 1 — Foundations
> Get comfortable with how LangChain works before building anything complex.

- [ ] 1. **Setup** — API keys, `.env`, Docker workflow, project structure
- [ ] 2. **LLMs & Chat Models** — `ChatOpenAI`, invoking with a string, understanding the response object
- [ ] 3. **Prompt Templates** — `PromptTemplate`, `ChatPromptTemplate`, variables in prompts
- [ ] 4. **Output Parsers** — `StrOutputParser`, `JsonOutputParser`, parsing structured responses
- [ ] 5. **LCEL (LangChain Expression Language)** — chaining with `|`, how pipelines work, `.invoke()` vs `.stream()` vs `.batch()`

---

## Stage 2 — Chains & Data Flow
> Learn to build multi-step pipelines that process and transform data.

- [ ] 6. **Sequential Chains** — passing output of one step as input to the next
- [ ] 7. **RunnablePassthrough & RunnableMap** — branching and merging data in chains
- [ ] 8. **Structured Output & Tool Calling** — `with_structured_output()`, Pydantic models, reliable JSON from LLMs
- [ ] 9. **Async Chains** — `ainvoke`, `astream` — essential for any real app

---

## Stage 3 — Memory & State
> Make your app remember things across turns.

- [ ] 10. **Conversation History** — `MessagesPlaceholder`, passing history manually
- [ ] 11. **Conversation Buffer Memory** — `RunnableWithMessageHistory`, session-based memory
- [ ] 12. **Persistent Memory** — storing chat history in Redis or a database

---

## Stage 4 — Retrieval Augmented Generation (RAG)
> The most common production use case — answering questions from your own data.

- [ ] 13. **Document Loaders** — PDFs, text files, web pages, Notion, etc.
- [ ] 14. **Text Splitters** — chunking strategies (size, overlap, semantic)
- [ ] 15. **Embeddings** — `OpenAIEmbeddings`, what vectors are and why they matter
- [ ] 16. **Vector Stores** — FAISS (local), Chroma, Pinecone (production)
- [ ] 17. **Basic RAG Chain** — retriever + prompt + LLM pipeline
- [ ] 18. **Advanced RAG** — reranking, hybrid search (BM25 + vector), parent-child chunks
- [ ] 19. **RAG Evaluation** — measuring retrieval quality, answer quality (using LangSmith)

---

## Stage 5 — Agents & Tools
> Let the LLM decide what actions to take using tools.

- [ ] 20. **Tools** — defining custom tools with `@tool`, input/output schemas
- [ ] 21. **Built-in Tools** — web search, Python REPL, calculator
- [ ] 22. **Agents** — how agents work, the ReAct loop (think → act → observe)
- [ ] 23. **AgentExecutor** — running an agent with tools in LangChain

---

## Stage 6 — Production & Deployment
> What separates a demo from a real product.

- [ ] 27. **Streaming** — streaming tokens to the frontend in real time
- [ ] 28. **LangServe** — deploying any chain as a REST API in minutes
- [ ] 29. **LangSmith** — tracing every LLM call, debugging failures, comparing runs
- [ ] 30. **Callbacks & Observability** — logging, cost tracking, latency monitoring
- [ ] 31. **Error Handling & Retries** — fallbacks, `.with_fallbacks()`, retry logic
- [ ] 32. **Cost & Token Management** — counting tokens, limiting spend, caching with `set_llm_cache`
- [ ] 33. **Testing Chains** — unit testing prompts, mocking LLM calls, regression tests

---

## Milestone Projects
> Build these after completing each stage to solidify the knowledge.

| # | Project | Covers |
|---|---------|--------|
| A | CLI chatbot with memory | Stages 1–3 |
| B | RAG app over a PDF | Stage 4 |
| C | Agent that can search the web + run code | Stages 1–5 |
| D | Full RAG API with tracing + streaming | Stages 1–6 |

---

## Current Topic
> Update this line as you progress. Start with **Topic 1 — Setup**.
