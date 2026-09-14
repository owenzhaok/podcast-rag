"""Provider-independent orchestration after unchanged BM25 retrieval."""

import asyncio

from api.rag.citations import validate_output
from api.rag.cache import GenerationCache, generation_key
from api.rag.config import RagConfig
from api.rag.llm import GenerationRequest, LLMProvider, InvalidGeneration, ProviderFailure
from api.rag.models import Answer, AskResponse, RagSource
from api.rag.prompt import SYSTEM, build_context


async def generate_answer(question: str, sources: list[RagSource], config: RagConfig,
                          provider: LLMProvider | None, redis=None) -> AskResponse:
    response = AskResponse(question=question, sources=sources, retrieval_mode="bm25",
                           status="insufficient_context", reason="no_usable_evidence",
                           degraded=any(not s.metadata_available for s in sources))
    if not sources:
        return response
    if provider is None:
        response.status = "generation_unavailable"
        response.reason = "unsupported_provider" if config.llm_provider not in ("", "groq") else "llm_not_configured"
        response.degraded = True
        return response
    context = build_context(question, sources, config)
    if context is None:
        response.reason = "context_budget_exceeded"
        response.degraded = True
        return response
    try:
        request = GenerationRequest(
            system=SYSTEM, user=context.user, model=config.llm_model,
            max_output_tokens=config.max_output_tokens, timeout_seconds=config.llm_timeout_seconds,
        )
        cache = GenerationCache(redis, config.answer_cache_ttl_seconds)
        key = generation_key(request, config.llm_provider)
        output = await cache.get(key, context.source_ids)
        if output is not None:
            response.cached = True
        else:
            raw = await asyncio.wait_for(provider.generate(request), timeout=config.llm_timeout_seconds)
            output = validate_output(raw, context.source_ids)
            await cache.set(key, output)
        if output.status == "insufficient_context":
            response.reason = "model_abstained"
            return response
        response.status = "answered"
        response.reason = None
        response.answer = Answer(paragraphs=output.paragraphs)
    except InvalidGeneration as exc:
        response.status, response.reason, response.degraded = "invalid_generation", exc.reason, True
    except ProviderFailure as exc:
        response.status, response.reason, response.degraded = "generation_unavailable", exc.reason, True
    except asyncio.TimeoutError:
        response.status, response.reason, response.degraded = "generation_unavailable", "provider_timeout", True
    except Exception:
        # Provider-boundary safeguard: never return raw errors or partial output.
        response.status, response.reason, response.degraded = "generation_unavailable", "provider_unavailable", True
    return response
