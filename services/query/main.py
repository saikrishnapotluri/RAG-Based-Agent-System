# services/query/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os, httpx, openai
from transformers import pipeline

# 1) URL of your embed/indexing service
EMBED_QUERY_URL = "http://127.0.0.1:8001/query/"

# 2) Load and validate your OpenAI key
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise RuntimeError("Missing OPENAI_API_KEY env var")

# 3) Which LLM to use for the primary call
LLM_MODEL = "gpt-3.5-turbo"

# 4) Local fallback generator (GPT-2), only return the new text
local_generator = pipeline(
    "text-generation",
    model="gpt2",
    tokenizer="gpt2",
    return_full_text=False,
)

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
    # 1) Retrieve top-k chunks
    async with httpx.AsyncClient() as client:
        r = await client.post(
            EMBED_QUERY_URL,
            json={"query": req.query, "top_k": req.top_k},
            timeout=30.0,
        )
        r.raise_for_status()
        chunks = r.json()["results"]

    # 2) Build the prompt
    context = "\n\n".join(
        f"[{c['document_id']}#{c['chunk_index']}] {c['text']}"
        for c in chunks
    )
    prompt = (
        "You are a helpful assistant. Use the following context to answer the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {req.query}\n\nAnswer concisely:"
    )

    # 3) Try OpenAI, fall back on GPT-2 if auth or quota fails
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

    except (openai.AuthenticationError, openai.RateLimitError):
        # Fallback to local GPT-2 (only the new text!)
        gen = local_generator(
            prompt,
            max_length=150,
            truncation=True,
            num_return_sequences=1
        )[0]["generated_text"]
        answer = gen.strip()

    # 4) Return the answer + source chunks
    return AnswerResponse(
        answer=answer,
        source_chunks=[SourceChunk(**c) for c in chunks]
    )
