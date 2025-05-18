# services/query/main.py

from fastapi import FastAPI
from pydantic import BaseModel
import os, httpx, openai

# Configure
EMBED_QUERY_URL = "http://127.0.0.1:8001/query/"
openai.api_key = os.getenv("OPENAI_API_KEY")
LLM_MODEL = "gpt-3.5-turbo"    # or whichever model you prefer

app = FastAPI()

@app.get("/", summary="Health check")
async def health():
    return {"status": "query-service up"}

# --- Schemas ---
class AskRequest(BaseModel):
    query: str
    top_k: int = 3

class SourceChunk(BaseModel):
    document_id: str
    chunk_index: int
    text: str
    score: float

class AnswerResponse(BaseModel):
    answer: str
    source_chunks: list[SourceChunk]

@app.post("/ask/", response_model=AnswerResponse)
async def ask(req: AskRequest):
    # 1) Retrieve top-k chunks from embed/indexing
    async with httpx.AsyncClient() as client:
        embed_resp = await client.post(
            EMBED_QUERY_URL,
            json={"query": req.query, "top_k": req.top_k},
            timeout=30.0
        )
        embed_resp.raise_for_status()
        chunks = embed_resp.json()["results"]  # list of dicts

    # 2) Build a little context string
    context = "\n\n".join(
        f"[{c['document_id']}#{c['chunk_index']}] {c['text']}"
        for c in chunks
    )

    # 3) Construct prompt for the LLM
    prompt = (
        "You are a helpful assistant. Use the following context to answer the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {req.query}\n\nAnswer concisely:"
    )

    # 4) Get completion from OpenAI
    completion = openai.ChatCompletion.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system",  "content": "You are a helpful assistant."},
            {"role": "user",    "content": prompt}
        ],
        temperature=0.2,
    )
    answer = completion.choices[0].message.content.strip()

    # 5) Return the answer + source chunks
    return AnswerResponse(
        answer=answer,
        source_chunks=[SourceChunk(**c) for c in chunks]
    )
