"""REST API. Run: uvicorn app.api:app --reload"""
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import config
from .pipeline import RagPipeline

app = FastAPI(title="Enterprise RAG Agents", version="1.0.0")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)


@lru_cache(maxsize=1)
def get_pipeline() -> RagPipeline:
    return RagPipeline()


@app.get("/health")
def health():
    return {"status": "ok", "mode": "mock" if config.USE_MOCK else "gemini", "llm_model": config.LLM_MODEL}


@app.post("/ask")
def ask(req: AskRequest):
    try:
        return get_pipeline().ask(req.question)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
