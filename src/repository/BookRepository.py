from pymilvus import Collection, MilvusClient, model, connections
import os

class BookRepository:
    def __init__(self):
        self.client = MilvusClient(uri=os.getenv("MILVUS_URL"))

    def find_book_by_keyword(self, query: str):
        connections.connect("default", host=os.getenv("MILVUS_IP"), port=os.getenv("MILVUS_PORT"))

        gemini_ef = model.dense.GeminiEmbeddingFunction(model_name='gemini-embedding-exp-03-07',
                                                        api_key=os.getenv('GEMINI_API_KEY'))

        embed_query = gemini_ef.encode_queries(query)

        collection = Collection("book")
        collection.load()
        return self.client.search(
            collection_name="book",
            data=embed_query,
            limit=10,
            search_params={"metric_type": "IP", "params": {}},
            output_fields=["text"],
        )
