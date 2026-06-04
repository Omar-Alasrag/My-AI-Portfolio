# 🤖 Reflection Agent (Iterative Content Refinement)

This project implements an autonomous AI agent designed to improve response quality through a self-reflection loop. Instead of providing a single, static answer, the agent evaluates its own output, identifies missing information, performs targeted web searches, and iteratively improves its response.

## 🚀 Key Technical Features
* **LangGraph Framework:** Manages complex workflows and iterative loops (cycles) using state-based graph logic.
* **Self-Reflection Loop:** The agent evaluates its own initial response to identify knowledge gaps or unnecessary content.
* **Structured Outputs:** Utilizes `Pydantic` models to enforce strict schema requirements for answers and reviews.
* **Tool Calling:** Integrates `Tavily Search` to dynamically retrieve real-world data based on self-generated queries.
* **State Management:** Leverages `MessagesState` to maintain conversation history across nodes in the graph.

## 🏗️ Project Architecture


The agent follows a circular workflow:
1. **Initial Response (Start Node):** Generates a preliminary answer along with a self-review.
2. **Search (Tool Node):** Executes web searches based on the queries identified in the self-review.
3. **Improve (Improve Node):** Refines the answer by incorporating search results and adhering to constraints.
4. **Conditional Edge:** Automatically determines whether to terminate the cycle or perform further research based on defined logic.

## 🛠️ Prerequisites
Ensure you have the necessary dependencies installed:
```bash
pip install -r requirements.txt

```

### Environment Variables

Create a `.env` file in the project root with the following keys:

* `GOOGLE_API_KEY=your_gemini_api_key`
* `TAVILY_API_KEY=your_tavily_api_key`

## 📂 Project Structure

* `main.py`: Entry point and `StateGraph` configuration.
* `nodes.py`: Defines the logic nodes (initial, search, improve) that drive the agent.
* `chains.py`: Manages Prompt Templates and LLM configuration.
* `schemas.py`: Pydantic models for structured data validation.

## 💡 How It Works

When you ask the agent a question:

1. It generates an `InitialAnswer` model, which forces the LLM to output an answer, a critique, and search queries simultaneously.
2. The agent uses the `TavilySearch` tool to fetch external information.
3. It passes the new context to the `ImproveNode`, which generates an `ImprovedAnswer` complete with sources.
4. The process continues until the logic in `stop_or_continue` is satisfied.

```

