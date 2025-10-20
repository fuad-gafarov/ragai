import os
from pymilvus import MilvusClient, connections, Collection
from sentence_transformers import SentenceTransformer

client = MilvusClient(uri=os.getenv("MILVUS_URL"))

with open("../../book.txt", "r", encoding="utf-8") as f:
    text = f.read()

model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

connections.connect("default", host=os.getenv("MILVUS_IP"), port=os.getenv("MILVUS_PORT"))
collection = Collection("book")
collection.load()

chunk_size = 500
overlap = 100

count = 0
chunks = []
for i in range(0, len(text), chunk_size - overlap):
    chunk = text[i:i + chunk_size]
    count = count + 1
    embeddings = model.encode(chunk)
    data = {"vector": embeddings, "text": chunk, "metadata": ""}
    client.insert(collection_name="book", data=data)

print(count)