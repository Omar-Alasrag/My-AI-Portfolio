from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import (ChatGoogleGenerativeAI,
                                    GoogleGenerativeAIEmbeddings)

load_dotenv()


@tool
def search(query: str):
    """This tool searches the LangChain documentation and returns content with sources"""

    docs = vectorstore.as_retriever().invoke(query, k=3)

    return "\n\n".join(
        f"Content: {doc.page_content}\nSource: {doc.metadata.get('source')}"
        for doc in docs
    )


embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001", output_dimensionality=1024
)

vectorstore = Chroma(embedding_function=embeddings, persist_directory="chroma_db")


class DocumentSearchAgent:
    def __init__(self):
        # load_dotenv()
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001", output_dimensionality=1024, max_retries=6
        )
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite", temperature=0.3
        )
        self.system_prompt = """
You are a strict retrieval assistant.

RULES:
- You MUST use the search tool for any question about LangChain
- Do NOT answer from your own knowledge
- If the tool returns no relevant info, say: "I could not find the answer in the provided sources"
- Always include sources at the end
- Keep answers concise
"""

        self.agent = create_agent(
            model=self.llm,
            tools=[search],
        )

    def ask(self):

        question = input("you: ").strip()

        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("human", "{input}"),
            ]
        )
        prompt = prompt_template.invoke(question)

        # answers_history = self.agent.invoke(messages=prompt)
        answers_history = self.agent.invoke(prompt)
        answers_history = self._filter_history(answers_history["messages"])

        print(f"agent: {answers_history[-1].content}")

        while True:
            question = input("you: ").strip()

            if question.lower() in ["quit", "exit", "q"]:
                break
            answers_history.append(HumanMessage(question))

            answers_history = self.agent.invoke({"messages": answers_history})

            print(f"agent: {answers_history["messages"][-1].content}")

            answers_history = self._filter_history(answers_history)

    def _filter_history(self, history):
        new_history = []

        for message in history:
            if isinstance(message, ToolMessage):
                continue

            if getattr(message, "tool_calls", None):
                continue

            new_history.append(message)

        return new_history


def main():
    print("Hello from documentation-assistant!")
    agent = DocumentSearchAgent()
    agent.ask()


if __name__ == "__main__":
    main()
