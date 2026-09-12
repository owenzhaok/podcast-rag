"""RAG routing skeleton: no retrieval, caching, or provider calls."""

from fastapi import APIRouter
from api.rag.config import RagConfig
from api.rag.models import AskRequest, AskResponse


def create_router(config: RagConfig) -> APIRouter:
    router = APIRouter()

    @router.post("/ask", response_model=AskResponse)
    async def ask(request: AskRequest):
        if not config.enabled:
            return AskResponse(
                question=request.question, status="disabled", reason="rag_disabled",
            )
        return AskResponse(
            question=request.question,
            status="generation_unavailable",
            reason="not_implemented" if config.llm_configured else "llm_not_configured",
        )

    return router
