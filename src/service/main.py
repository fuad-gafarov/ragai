import os

from fastapi import FastAPI
from google import genai
from pymilvus import MilvusClient, model, Collection, connections

llm_client = genai.Client()
app = FastAPI()
client = MilvusClient(uri=os.getenv("MILVUS_URL"))

connections.connect("default", host=os.getenv("MILVUS_IP"), port=os.getenv("MILVUS_PORT"))

docs = [
    "Fuad Qafarov is a good guy.",
]

gemini_ef = model.dense.GeminiEmbeddingFunction(model_name='gemini-embedding-exp-03-07',
                                                api_key=os.environ['GEMINI_API_KEY'])
@app.post("/")
def read_root():
    query_embed = gemini_ef.encode_queries("who is fuad?")
    collection = Collection("book")
    collection.load()
    search_res = client.search(
        collection_name="book",
        data=query_embed,  # Convert the question to an embedding vector
        limit=5,  # Return top 3 results
        search_params={"metric_type": "IP", "params": {}},  # Inner product distance
        output_fields=["text"],  # Return the text field
    )

    adv_query = f"You are a book expert. Answer based on this - {search_res}. Now question - how much good guys we have?"
    response = llm_client.models.generate_content(
        model="gemini-2.5-flash", contents=adv_query
    )
    return {"key": response.text}
