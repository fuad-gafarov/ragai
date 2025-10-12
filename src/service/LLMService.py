from src.repository.BookRepository import BookRepository
from google import genai

class LLMService:
    def __init__(self):
        self.bookRepo = BookRepository()

    def llm_query(self, query: str):
        llm_client = genai.Client()
        repo_result = self.bookRepo.find_book_by_keyword(query)

        adv_query = f"You are a book expert. Answer based on this - {repo_result}. Now question - {query}"
        response = llm_client.models.generate_content(model="gemini-2.5-flash", contents=adv_query)
        return {"key": response.text}