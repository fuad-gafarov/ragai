from google import genai
from pymilvus import model, MilvusClient
import os

client = MilvusClient(
    uri="http://localhost:19530"
)

docs = [
    "Fuad Qafarov is a good guy.",
]


gemini_ef = model.dense.GeminiEmbeddingFunction(model_name='gemini-embedding-exp-03-07', api_key=os.environ['GEMINI_API_KEY'])
vectors = gemini_ef.encode_documents(docs)

print(vectors)

data = {"vector": vectors[0], "text": docs[0], "metadata": ""}


print(data)
client.insert(collection_name="book", data=data)

#with open("turgic.txt", "r", encoding="utf-8") as f:
#    book_text = f.read()
