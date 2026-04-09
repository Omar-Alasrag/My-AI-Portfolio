# 🤖 Agentic RAG Documentation Assistant

An intelligent Retrieval-Augmented Generation (RAG) agent designed to answer technical questions with **zero hallucinations** by strictly adhering to provided documentation.

## 🛠️ Advanced Features
* **Asynchronous Ingestion:** Uses `asyncio` and `TavilyCrawl` to parallelize documentation scraping, significantly reducing vector DB setup time.
* **Agentic Reasoning:** Powered by **LangGraph/LangChain** with a custom filtering logic for `ToolMessages` to maintain a clean, high-token-efficiency conversation history.
* **Vector Store:** Uses **ChromaDB** with `Gemini-embedding-001` for high-dimensional semantic search (1024-dim).
* **Strict Mode:** A specialized system prompt ensures the agent admits when information is missing rather than hallucinating answers.

## 🧱 Tech Stack
* **LLM:** Google Gemini 2.0 Flash Lite
* **Orchestration:** LangChain / LangGraph
* **Data Handling:** RecursiveCharacterTextSplitter for optimal context windowing.

## 🚀 How to Run
1. Add your `GOOGLE_API_KEY` to a `.env` file.
2. Run `python ingestion.py` to crawl the LangChain docs.
3. Run `python main.py` to chat with the agent.