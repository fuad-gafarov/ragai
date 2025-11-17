from fastapi import FastAPI

from src.model.LLMQueryDto import LLMQueryDto
from src.service.LLMService import LLMService
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # ← allow all origins
    allow_credentials=False,  # ← must be False when using "*"
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.post("/query")
def query(llm_query_dto: LLMQueryDto):
    return LLMService().llm_query(llm_query_dto.text)