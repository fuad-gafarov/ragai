import os

from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection

connections.connect("default", host=os.getenv("MILVUS_IP"), port=os.getenv("MILVUS_PORT"))


fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=2560),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2000),
    FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=2000)
]

schema = CollectionSchema(fields, "Book chunks")
collection = Collection("book", schema)

index_params = {
        "index_type": "IVF_FLAT",  # Or HNSW, FLAT, etc.
        "metric_type": "IP",       # Or IP, COSINE
        "params": {"nlist": 128}   # Specific parameters for the chosen index type
}

collection.create_index(field_name="vector", index_params=index_params)
