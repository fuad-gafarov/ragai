from src.repository.BookRepository import BookRepository
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM


class LLMService:
    def __init__(self):
        self.bookRepo = BookRepository()

    def llm_query(self, query: str):
        repo_result = self.bookRepo.find_book_by_keyword(query)

        tokenizer = AutoTokenizer.from_pretrained("D:\models\Qwen3-0.6B")
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")

        adv_query = (f"Sen kitab ekspertisen. Cavabi tamamilə Azərbaycan dilində ver və yalnız bu suala uygun ver - "
                     f"{repo_result}. Indi ise sual - {query}")

        messages = [
            {"role": "user", "content": adv_query},
        ]

        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        outputs = model.generate(**inputs, max_new_tokens=40)
        return {"key": tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])}