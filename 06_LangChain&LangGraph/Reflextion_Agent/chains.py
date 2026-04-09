from datetime import datetime
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

from schemas import InitialAnswer, ImprovedAnswer

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")


base_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Time now: {time}

Steps:
1. {task}
2. Review your answer and find problems.
3. Suggest search ideas to improve it.
""",
        ),
        MessagesPlaceholder("messages"),
        ("system", "Reply using the required format."),
    ]
).partial(time=datetime.now().isoformat())


first_prompt = base_prompt.partial(
    task="""Give useful general advice even if the question is not specific.
Do not ask the user for more details.
Write about 80 words."""
)

improve_prompt = base_prompt.partial(
    task="""Improve your previous answer:
- Fix missing info
- Remove unnecessary parts
- If the user question is unclear, give a useful answer
- Keep it under 100 words
- Add numbered references at the end"""
)


first_chain = first_prompt | llm.bind_tools(
    [InitialAnswer], tool_choice=InitialAnswer.__name__
)

improve_chain = improve_prompt | llm.bind_tools(
    [ImprovedAnswer], tool_choice=ImprovedAnswer.__name__
)
