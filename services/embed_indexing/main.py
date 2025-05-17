# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

app = FastAPI()

@app.get("/", summary="Health check")
async def health():
    return {"status": "embed-indexing up"}

# --- Pydantic schemas ---
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

# --- Local embedding model setup ---
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBED_MODEL_NAME)

# --- Globals for FAISS index and metadata ---
index = None                      # FAISS index instance
metadata: dict[int, tuple] = {}   # maps vector_id -> (document_id, chunk_index, text)
next_id = 0                       # incremental ID for each vector

def get_embedding(text: str) -> list[float]:
    """
    Encodes text into a float32 vector using a local sentence-transformers model.
    """
    vec = embedder.encode(text, convert_to_numpy=True)
    return vec.astype("float32").tolist()

@app.post("/embed/", response_model=EmbedResponse)
def embed_and_index(req: EmbedRequest):
    global index, metadata, next_id

    # 1) Compute embeddings for each chunk
    embs = [get_embedding(chunk.text) for chunk in req.chunks]
    arr = np.array(embs, dtype="float32")

    # 2) Initialize FAISS index on first call
    dim = arr.shape[1]
    if index is None:
        index = faiss.IndexFlatL2(dim)

    # 3) Add vectors to FAISS and record metadata
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

    # 1) Embed the user query
    q_vec = np.array(get_embedding(req.query), dtype="float32").reshape(1, -1)

    # 2) Search FAISS for top_k nearest neighbors
    distances, indices = index.search(q_vec, req.top_k)

    # 3) Build response from metadata
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        doc_id, chunk_idx, text = metadata[idx]
        results.append(RetrievedChunk(
            document_id=doc_id,
            chunk_index=chunk_idx,
            text=text,
            score=float(dist),
        ))

    return QueryResponse(results=results)
