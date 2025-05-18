# services/query/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os, httpx, openai

# 1) URL of your embed/indexing service
EMBED_QUERY_URL = "http://127.0.0.1:8001/query/"

# 2) Load and validate your OpenAI key
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise RuntimeError("Missing OPENAI_API_KEY env var")

# 3) LLM model to use
LLM_MODEL = "gpt-3.5-turbo"

app = FastAPI()

@app.get("/", summary="Health check")
async def health():
    return {"status": "query-service up"}

# --- Request / Response schemas ---
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

# --- The /ask endpoint ---
@app.post("/ask/", response_model=AnswerResponse)
async def ask(req: AskRequest):
    # 1) Retrieve top-k chunks from embed/indexing service
    async with httpx.AsyncClient() as client:
        r = await client.post(
            EMBED_QUERY_URL,
            json={"query": req.query, "top_k": req.top_k},
            timeout=30.0,
        )
        r.raise_for_status()
        chunks = r.json()["results"]

    # 2) Build the prompt with those chunks
    context = "\n\n".join(
        f"[{c['document_id']}#{c['chunk_index']}] {c['text']}"
        for c in chunks
    )
    prompt = (
        "You are a helpful assistant. Use the following context to answer the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {req.query}\n\nAnswer concisely:"
    )

    # 3) Call OpenAI, catching authentication & quota errors
    try:
        resp = openai.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.2,
        )
        answer = resp.choices[0].message.content.strip()
    except openai.AuthenticationError:
        # Invalid or missing API key
        raise HTTPException(status_code=401, detail="Invalid OpenAI API key")
    except openai.RateLimitError:
        # Quota exhausted
        raise HTTPException(status_code=503, detail="OpenAI quota exceeded. Please check your billing.")

    # 4) Return the answer plus which chunks were used
    return AnswerResponse(
        answer=answer,
        source_chunks=[SourceChunk(**c) for c in chunks]
    )
