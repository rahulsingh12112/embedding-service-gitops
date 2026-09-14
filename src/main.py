"""
Embedding Service — FastAPI (EKS pe chalega)
Text bhejo → vector wapas. Ye wahi hai jo tune laptop pe chalaya, ab prod me.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List
import os, time

app = FastAPI(title="Embedding Service", version="1.0.0")

MODEL_NAME = os.getenv("MODEL_NAME", "BAAI/bge-base-en-v1.5")  # base = CPU pe theek

print(f"Loading model {MODEL_NAME} on CPU...")
start = time.time()
model = SentenceTransformer(MODEL_NAME, device="cpu")
DIMENSIONS = model.get_sentence_embedding_dimension()
print(f"Model loaded in {time.time()-start:.1f}s! Dimensions: {DIMENSIONS}")

IS_BGE = "bge" in MODEL_NAME.lower()
QUERY_PREFIX = "Represent this sentence for searching relevant passages: " if IS_BGE else ""


class EmbedRequest(BaseModel):
    texts: List[str]
    is_query: bool = False


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    dimensions: int
    model: str
    count: int


@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest):
    if not request.texts:
        raise HTTPException(400, "texts list cannot be empty")
    if len(request.texts) > 100:
        raise HTTPException(400, "max 100 texts per request")

    texts = request.texts
    if request.is_query and IS_BGE:
        texts = [f"{QUERY_PREFIX}{t}" for t in texts]

    embeddings = model.encode(texts, normalize_embeddings=True, batch_size=16, show_progress_bar=False)
    return EmbedResponse(
        embeddings=embeddings.tolist(), dimensions=DIMENSIONS,
        model=MODEL_NAME, count=len(texts)
    )


@app.get("/health")
async def health():
    return {"status": "healthy", "model": MODEL_NAME, "dimensions": DIMENSIONS}


@app.get("/ready")
async def ready():
    test = model.encode("healthcheck", normalize_embeddings=True)
    if len(test) == DIMENSIONS:
        return {"status": "ready"}
    raise HTTPException(503, "not ready")
