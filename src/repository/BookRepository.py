import os
from pymilvus import Collection, MilvusClient, connections
from sentence_transformers import SentenceTransformer


class BookRepository:
    def __init__(self):
        self.client = MilvusClient(uri=os.getenv("MILVUS_URL"))

    def find_book_by_keyword(self, query: str):
        connections.connect("default", host=os.getenv("MILVUS_IP"), port=os.getenv("MILVUS_PORT"))

        model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
        embed_query = model.encode(query).tolist()

        collection = Collection("book")
        collection.load()
        return self.client.search(
            collection_name="book",
            data=[embed_query],
            limit=10,
            search_params={"metric_type": "IP", "params": {}},
            output_fields=["text"],
        )
