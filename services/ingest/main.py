from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Document Ingestion Service")

# Configuration
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    """
    Splits `text` into chunks of approximately `chunk_size` characters
    with `overlap` characters overlap between chunks.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


class Chunk(BaseModel):
    document_id: str
    chunk_index: int
    text: str

@app.post("/upload/", summary="Upload document and split into chunks")
async def upload_document(document: UploadFile = File(...)):
    if document.content_type not in ["text/markdown", "text/plain"]:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    content = (await document.read()).decode("utf-8")
    chunks = chunk_text(content)
    result = []
    for i, txt in enumerate(chunks):
        result.append(Chunk(document_id=document.filename, chunk_index=i, text=txt))
    return {"chunks": [c.dict() for c in result]}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
