FROM python:3.12-slim

WORKDIR /app

# PyMuPDF and onnxruntime need these at runtime on slim images.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Bake the embedding model into the image at build time instead of
# downloading it on every cold Cloud Run instance's first request.
RUN python scripts/warm_embedding_cache.py

ENV PORT=8080
EXPOSE 8080

CMD exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT} --workers 1
