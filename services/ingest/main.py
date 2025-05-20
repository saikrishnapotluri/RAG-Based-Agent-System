from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import httpx

EMBED_SERVICE_URL = "http://127.0.0.1:8001/embed/"

app = FastAPI()

@app.get("/", summary="Health check")
async def health():
    return {"status": "ok"}

class Chunk(BaseModel):
    document_id: str
    chunk_index: int
    text: str

class ChunkResponse(BaseModel):
    chunks: list[Chunk]

@app.post("/upload/", response_model=ChunkResponse)
async def upload(document: UploadFile = File(...)):
    # 1) Read & split into fixed-size chunks
    content = (await document.read()).decode("utf-8")
    words = content.split()
    chunk_size = 50
    chunks: list[Chunk] = []
    for i in range(0, len(words), chunk_size):
        chunks.append(Chunk(
            document_id=document.filename,
            chunk_index=i // chunk_size,
            text=" ".join(words[i : i + chunk_size])
        ))

    # 2) Send chunks off to the embed/indexing service
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            EMBED_SERVICE_URL,
            json={"chunks": [c.dict() for c in chunks]},
            timeout=30.0
        )
        resp.raise_for_status()
        # you could grab resp.json()["indexed_ids"] here if you want

    # 3) Return the chunks to the caller
    return ChunkResponse(chunks=chunks)
