from pydantic import BaseModel, Field


class Review(BaseModel):
    missing_parts: str = Field(description="What is missing")
    extra_parts: str = Field(description="What is unnecessary")


class InitialAnswer(BaseModel):
    answer: str = Field(description="Short helpful answer")
    review: Review = Field(description="Self review of the answer")
    queries: list[str] = Field(
        description="Search ideas to improve the answer"
    )


class ImprovedAnswer(InitialAnswer):
    references: list[str] = Field(
        description="List of sources used in the answer"
    )