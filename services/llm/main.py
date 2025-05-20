from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os, openai

app = FastAPI()

# Load API key
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise RuntimeError("Missing OPENAI_API_KEY")

class GenerateRequest(BaseModel):
    prompt: str
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.2
    max_tokens: int = 150

class GenerateResponse(BaseModel):
    answer: str

@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    try:
        resp = openai.ChatCompletion.create(
            model=req.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user",   "content": req.prompt},
            ],
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )
        return GenerateResponse(answer=resp.choices[0].message.content.strip())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8004)))
