import os, openai

openai.api_key = os.getenv("OPENAI_API_KEY")
print("Key loaded?", bool(openai.api_key))

resp = openai.embeddings.create(
    model="text-embedding-ada-002",
    input="hello world"
)
print("Embedding length:", len(resp["data"][0]["embedding"]))
