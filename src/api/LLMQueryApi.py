from fastapi import FastAPI

from src.model.LLMQueryDto import LLMQueryDto
from src.service.LLMService import LLMService

app = FastAPI()

@app.post("/query")
def query(llm_query_dto: LLMQueryDto):
    return LLMService().llm_query(llm_query_dto.text)