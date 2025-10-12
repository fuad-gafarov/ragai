from pydantic import BaseModel


class LLMQueryDto(BaseModel):
    text: str | None = None
