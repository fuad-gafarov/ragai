from pymilvus import MilvusClient
import os
milvus_uri = os.getenv("MILVUS_URL", "http://194.163.159.195:19530")
MilvusClient(uri=milvus_uri).drop_collection(collection_name="book")
