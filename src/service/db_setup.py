from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection

connections.connect("default", host="localhost", port="19530")


fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=3072),
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

    # Create the index on the vector field
collection.create_index(field_name="vector", index_params=index_params)
#collection.release()
#collection.drop_index()
