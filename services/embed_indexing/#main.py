# main.py
import os
import faiss
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import openai
import os, faiss

app = FastAPI()

@app.get("/", summary="Health check")
async def health():
    return {"status": "embed-indexing up"}

# ---  CONFIGURE YOUR API KEY  ---
openai.api_key = os.getenv("OPENAI_API_KEY")

# ---  MODELS  ---
EMBED_MODEL = "text-embedding-ada-002"

# ---  GLOBALS  ---
app = FastAPI()
@app.get("/", summary="Health check")
async def health():
    return {"status": "embed-indexing up"}

index = None            # FAISS index object
metadata: dict = {}     # maps vector_id -> (document_id, chunk_index, text)
next_id = 0             # incremental ID for each vector

# ---  Pydantic schemas  ---
class Chunk(BaseModel):
    document_id: str
    chunk_index: int
    text: str

class EmbedRequest(BaseModel):
    chunks: list[Chunk]

class EmbedResponse(BaseModel):
    indexed_ids: list[int]

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

class RetrievedChunk(BaseModel):
    document_id: str
    chunk_index: int
    text: str
    score: float

class QueryResponse(BaseModel):
    results: list[RetrievedChunk]

# ---  HELPERS  ---
def get_embedding(text: str) -> list[float]:
    # use the plural "embeddings" in v1
    resp = openai.embeddings.create(
        model=EMBED_MODEL,
        input=[text]
    )
    # the response shape is similar: a "data" list
    return resp["data"][0]["embedding"]


# ---  ROUTES  ---
@app.post("/embed/", response_model=EmbedResponse)
def embed_and_index(req: EmbedRequest):
    global index, metadata, next_id

    # 1) Compute embeddings for each chunk
    embs = [get_embedding(chunk.text) for chunk in req.chunks]
    arr = np.array(embs, dtype="float32")

    # 2) On first call, create FAISS index of correct dim
    dim = arr.shape[1]
    if index is None:
        index = faiss.IndexFlatL2(dim)

    # 3) Add to FAISS and record metadata
    ids = []
    for vec, chunk in zip(arr, req.chunks):
        index.add(vec.reshape(1, -1))
        metadata[next_id] = (chunk.document_id, chunk.chunk_index, chunk.text)
        ids.append(next_id)
        next_id += 1

    return EmbedResponse(indexed_ids=ids)

@app.post("/query/", response_model=QueryResponse)
def query_chunks(req: QueryRequest):
    if index is None or index.ntotal == 0:
        return QueryResponse(results=[])

    # 1) Embed the query
    q_emb = np.array(get_embedding(req.query), dtype="float32").reshape(1, -1)

    # 2) Search top_k
    D, I = index.search(q_emb, req.top_k)  # distances & indices

    results = []
    for dist, idx in zip(D[0], I[0]):
        doc_id, chunk_idx, text = metadata[idx]
        results.append(RetrievedChunk(
            document_id=doc_id,
            chunk_index=chunk_idx,
            text=text,
            score=float(dist),
        ))

    return QueryResponse(results=results)
