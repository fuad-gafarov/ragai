import os
from pymilvus import Collection, MilvusClient, connections
from sentence_transformers import SentenceTransformer, models


class BookRepository:
    def __init__(self):
        self.client = MilvusClient(uri=os.getenv("MILVUS_URL"))

    def find_book_by_keyword(self, query: str):
        connections.connect("default", host=os.getenv("MILVUS_IP"), port=os.getenv("MILVUS_PORT"))

        word_embedding_model = models.Transformer("D:\\models\\qwen3-06B-embedding")
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
      #  model = SentenceTransformer("D:\\models\\qwen3-06B-embedding")
        embed_query = model.encode(query, convert_to_tensor=True).tolist()

        collection = Collection("book")
        collection.load()
        return self.client.search(
            collection_name="book",
            data=[embed_query],
            limit=5,
            search_params={"metric_type": "IP", "params": {}},
            output_fields=["text"],
        )
