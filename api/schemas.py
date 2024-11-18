from langchain_core.documents import Document
from pydantic import BaseModel


class InputPrompt(BaseModel):
    input: str


class Answer(BaseModel):
    input: str
    context: list[Document]
    answer: str
