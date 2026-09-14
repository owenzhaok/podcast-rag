"""Optional grounded generation and read-only source routes; no RAG caching."""

from time import perf_counter
from fastapi import APIRouter, HTTPException, Depends
from api.rag.config import RagConfig
from api.rag.models import AskRequest, AskResponse, RagSource
from api.rag.retrieval import retrieve, resolve, RetrievalUnavailable
from api.rag.groq import get_generation_provider
from api.rag.generation import generate_answer


def create_router(config: RagConfig, get_clients=lambda: (None, None)) -> APIRouter:
    router = APIRouter()

    @router.post("/ask", response_model=AskResponse)
    async def ask(request: AskRequest, provider=Depends(get_generation_provider)):
        if not config.enabled:
            return AskResponse(
                question=request.question, status="disabled", reason="rag_disabled",
            )
        started = perf_counter()
        try:
            es, pool = get_clients()
            sources = await retrieve(request.question, es, pool, config)
        except RetrievalUnavailable:
            raise HTTPException(status_code=503, detail="retrieval_unavailable") from None
        response = await generate_answer(request.question, sources, config, provider)
        response.took_ms = int((perf_counter() - started) * 1000)
        return response

    @router.get("/sources/{chunk_id}", response_model=RagSource)
    async def source(chunk_id: str):
        if not config.enabled:
            raise HTTPException(status_code=404, detail="source_unavailable")
        try:
            es, pool = get_clients()
            result = await resolve(chunk_id, es, pool, config)
        except RetrievalUnavailable:
            raise HTTPException(status_code=503, detail="retrieval_unavailable") from None
        if result is None:
            raise HTTPException(status_code=404, detail="source_unavailable")
        return result

    return router
