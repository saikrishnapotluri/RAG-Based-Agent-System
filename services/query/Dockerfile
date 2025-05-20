# services/query/Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install Python deps (include transformers & torch)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

EXPOSE 8002
# The OPENAI_API_KEY will come in via docker-compose
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
