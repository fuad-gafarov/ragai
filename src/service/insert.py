import os
import tiktoken
from pymilvus import model, MilvusClient

client = MilvusClient(uri=os.getenv("MILVUS_URL"))

with open("../../book.txt", "r", encoding="utf-8") as f:
    text = f.read()

enc = tiktoken.encoding_for_model("text-embedding-3-large")
tokens = enc.encode(text)

chunk_size = 99      # tokens per chunk
overlap = 10
gemini_ef = model.dense.GeminiEmbeddingFunction(model_name='gemini-embedding-exp-03-07', api_key=os.getenv('GEMINI_API_KEY'))
chunks = []
for i in range(0, len(tokens), chunk_size - overlap):
    chunk = tokens[i:i + chunk_size]
    chunks.append(enc.decode(chunk))

    vector = gemini_ef.encode_documents(list(enc.decode(chunk)))
    data = {"vector": enc.decode(chunk), "text": chunk, "metadata": ""}
    client.insert(collection_name="book", data=data)

