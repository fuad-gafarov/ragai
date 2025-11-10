from src.repository.BookRepository import BookRepository
from transformers import AutoTokenizer, AutoModelForCausalLM
from groq import Groq
from fastapi.responses import JSONResponse


class LLMService:
    def __init__(self):
        self.bookRepo = BookRepository()


    def llm_query(self, query: str):
        repo_result = self.bookRepo.find_book_by_keyword(query)


        result_str = ""

        for inner_list in repo_result:
            for d in inner_list:
                for v in d.values():
                    result_str += str(v) + " "

        result_str = result_str.strip()

        print(result_str)
        client = Groq()
        completion = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "user",
                    "content": "Sen kitab ekspertisen. Cavabi tamamilə Azərbaycan dilində ver və yalnız bu suala uygun ver - "
                     f"{result_str}. Indi ise sual - {query}"
                }
            ],
            temperature=1,
            max_completion_tokens=8192,
            top_p=1,
            reasoning_effort="medium",
            stream=True,
            stop=None
        )

        for chunk in completion:
            print(chunk.choices[0].delta.content or "", end="")

    def llm_query2(self, query: str):
        repo_result = self.bookRepo.find_book_by_keyword(query)

        tokenizer = AutoTokenizer.from_pretrained("D:\\models\\Qwen3-4B")
        model = AutoModelForCausalLM.from_pretrained("D:\\models\\Qwen3-4B")

        result_str = ""
        count = 0
        for inner_list in repo_result:
            for d in inner_list:
                for v in d.values():
                    result_str += str(v) + " "

        result_str = result_str.strip()
        print(result_str)
        adv_query = (f"Sen kitab ekspertisen. Cavabi tamamilə Azərbaycan dilində ver və yalnız bu suala uygun ver - "
                     f"{result_str}. Indi ise sual - {query}")

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

        outputs = model.generate(**inputs, max_new_tokens=4000)
        #return {"key": tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])}
        return JSONResponse(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))