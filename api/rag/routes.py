"""Evidence-only routes; no generation or RAG caching."""

from time import perf_counter
from fastapi import APIRouter, HTTPException
from api.rag.config import RagConfig
from api.rag.models import AskRequest, AskResponse, RagSource
from api.rag.retrieval import retrieve, resolve, RetrievalUnavailable


def create_router(config: RagConfig, get_clients=lambda: (None, None)) -> APIRouter:
    router = APIRouter()

    @router.post("/ask", response_model=AskResponse)
    async def ask(request: AskRequest):
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
        return AskResponse(
            question=request.question,
            status="generation_unavailable" if sources else "insufficient_context",
            reason=("not_implemented" if config.llm_configured else "llm_not_configured")
            if sources else "no_usable_evidence",
            sources=sources, retrieval_mode="bm25",
            degraded=any(not source.metadata_available for source in sources),
            took_ms=int((perf_counter() - started) * 1000),
        )

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
