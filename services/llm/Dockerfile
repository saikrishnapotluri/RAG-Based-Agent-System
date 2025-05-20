FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy service code
COPY main.py .

# Expose the port your FastAPI app listens on
EXPOSE 8004

# Run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8004"]
