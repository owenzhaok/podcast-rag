"""Small Groq chat-completion adapter; no SDK and no startup network activity."""

import os
import json
import logging
import re

import httpx
from fastapi import Request

from api.rag.llm import GenerationRequest, InvalidGeneration, ProviderFailure


logger = logging.getLogger(__name__)

# Groq strict-mode subset: closed objects with every property required.
# Semantic consistency and request-local citation membership remain Python checks.
GENERATION_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["answered", "insufficient_context"]},
        "paragraphs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "source_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["text", "source_ids"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["status", "paragraphs"],
    "additionalProperties": False,
}


def _sanitize_diagnostic(value, api_key: str, limit=512, escaped=False):
    if not isinstance(value, str):
        return None
    # Inspect before truncation. Suppress the entire field if it resembles credentials.
    if (api_key and api_key in value) or re.search(
        r"authorization|bearer|api[ _-]?key|password|secret|credential|token|gsk_|sk-",
        value, re.IGNORECASE,
    ):
        return "[redacted sensitive diagnostic]"
    if escaped:
        return json.dumps(value, ensure_ascii=True)[1:-1][:limit]
    return "".join(c if c.isprintable() else " " for c in value)[:limit]


def _error_diagnostic(status: int, body: bytes, api_key: str, request: GenerationRequest) -> dict:
    """Allowlisted, bounded diagnostics only; never log bodies or request headers."""
    try:
        error = json.loads(body).get("error", {})
        if not isinstance(error, dict):
            error = {}
    except (ValueError, AttributeError, RecursionError):
        error = {}

    # Count only the evidence actually sent after context budgeting; never log it.
    try:
        evidence = json.loads(request.user).get("TRANSCRIPT_EVIDENCE")
        source_count = len(evidence) if isinstance(evidence, list) else None
    except (ValueError, AttributeError, RecursionError):
        source_count = None
    diagnostic = {
        "http_status": status,
        "error_type": _sanitize_diagnostic(error.get("type"), api_key),
        "error_code": _sanitize_diagnostic(error.get("code"), api_key),
        "error_message": _sanitize_diagnostic(error.get("message"), api_key),
        "request": {
            "model": _sanitize_diagnostic(request.model, api_key),
            "response_format_type": "json_schema", "strict": True,
            "max_completion_tokens": request.max_output_tokens,
            "system_message_bytes": len(request.system.encode("utf-8")),
            "user_message_bytes": len(request.user.encode("utf-8")),
            "selected_source_count": source_count,
        },
    }
    if error.get("code") == "json_validate_failed" and "failed_generation" in error:
        failed = error["failed_generation"]
        if not isinstance(failed, str):
            failed = json.dumps(failed, ensure_ascii=True)
        diagnostic["failed_generation"] = _sanitize_diagnostic(failed, api_key, 2000, escaped=True)
    return diagnostic


class GroqProvider:
    URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, api_key: str, transport=None):
        # Normal object repr omits attributes. Never serialize/log this object.
        self._api_key = api_key
        self._transport = transport

    async def generate(self, request: GenerationRequest) -> str:
        try:
            async with httpx.AsyncClient(
                transport=self._transport, timeout=request.timeout_seconds,
                follow_redirects=False, trust_env=False,
            ) as client:
                async with client.stream("POST", self.URL,
                    headers={"Authorization": "Bearer " + self._api_key},
                    json={"model": request.model, "messages": [
                        {"role": "system", "content": request.system},
                        {"role": "user", "content": request.user}],
                        "response_format": {"type": "json_schema", "json_schema": {
                            "name": "grounded_answer", "strict": True, "schema": GENERATION_SCHEMA}},
                        "max_completion_tokens": request.max_output_tokens,
                        "temperature": 0, "stream": False},
                ) as response:
                    body = bytearray()
                    async for chunk in response.aiter_bytes():
                        body.extend(chunk)
                        if len(body) > 131072:
                            raise InvalidGeneration("malformed_provider_response")
                    status = response.status_code
                    if status != 200:
                        logger.warning("Groq provider error: %s", json.dumps(
                            _error_diagnostic(status, body, self._api_key, request), ensure_ascii=True))
                        reason = {401: "authentication_failed", 403: "authentication_failed",
                                  402: "quota_exceeded", 429: "rate_limited"}.get(status)
                        if status == 429:
                            # Inspect only an allowlisted machine code; never expose body/message.
                            try:
                                code = json.loads(body).get("error", {}).get("code")
                                if code in ("insufficient_quota", "quota_exceeded"):
                                    reason = "quota_exceeded"
                            except (ValueError, AttributeError, TypeError, httpx.ResponseNotRead):
                                pass
                        raise ProviderFailure(reason or (
                            "provider_unavailable" if status >= 500 else "provider_request_rejected"))
                    try:
                        data = json.loads(body)
                        choices = data["choices"]
                        if not isinstance(choices, list) or len(choices) != 1:
                            raise ValueError()
                        choice = choices[0]
                        content = choice["message"]["content"]
                        if choice.get("finish_reason") != "stop" or not isinstance(content, str):
                            raise ValueError()
                        if self._api_key and self._api_key in content:
                            raise ValueError()
                        return content
                    except (ValueError, KeyError, TypeError, AttributeError, RecursionError):
                        raise InvalidGeneration("malformed_provider_response") from None
        except httpx.TimeoutException:
            raise ProviderFailure("provider_timeout") from None
        except httpx.HTTPError:
            raise ProviderFailure("provider_unavailable") from None


def get_generation_provider(request: Request):
    config = request.app.state.rag_config
    if not config.enabled or config.llm_provider != "groq" or not config.llm_model:
        return None
    key = os.environ.get("RAG_LLM_API_KEY", "").strip()
    return GroqProvider(key) if key else None
