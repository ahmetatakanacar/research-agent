from typing import List

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class Analyst(BaseModel):
    affiliation: str = Field(description="Primary affiliation of the analyst")
    name: str = Field(description="Name of the analyst")
    role: str = Field(description="Role of the analyst in the context of the topic")
    description: str = Field(description="Description of the analyst focus, concerns and motives")

    @property
    def persona(self) -> str:
        return f"Name: {self.name}\nRole: {self.role}\nAffiliation: {self.affiliation}\nDescription: {self.description}"


class Perpectives(BaseModel):
    analyst: List[Analyst] = Field(description="Comprehensive list of analysts with their roles and affiliations")


class SearchQuery(BaseModel):
    search_query: str = Field(
        description="Search query for web and encyclopedia retrieval, written in the target report language."
    )
    english_query: str = Field(
        default="",
        description="The same query written in English, used to find English-language sources.",
    )


class SourcePicks(BaseModel):
    best: List[int] = Field(
        description="Numbers of the chosen sources, most trustworthy and relevant first. May be fewer than requested."
    )


class ReviewIssue(BaseModel):
    claim: str = Field(description="The claim in the report that is a problem, quoted briefly.")
    problem: str = Field(description="Why it is a problem: unsupported by the cited source, contradicted by it, or invented.")


class ReviewResult(BaseModel):
    issues: List[ReviewIssue] = Field(
        default_factory=list,
        description="Genuine problems only. Leave empty when every checked claim is supported.",
    )


class SourceDoc(TypedDict):
    url: str
    title: str
    content: str
    kind: str
