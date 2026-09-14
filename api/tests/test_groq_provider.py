import json

import httpx
import pytest

from api.rag.groq import GroqProvider
from api.rag.llm import GenerationRequest, InvalidGeneration, ProviderFailure


REQUEST = GenerationRequest("Return JSON.", "Evidence", "runtime-selected-model", 800, 2)


@pytest.mark.asyncio
async def test_groq_wire_format_without_network():
    def handler(request):
        assert str(request.url) == GroqProvider.URL
        assert request.method == "POST"
        assert request.headers["Content-Type"] == "application/json"
        assert request.headers["Authorization"] == "Bearer test-placeholder"
        body = json.loads(request.content)
        assert body["model"] == "runtime-selected-model" and body["max_completion_tokens"] == 800
        assert body["response_format"] == {
            "type": "json_schema",
            "json_schema": {"name": "grounded_answer", "strict": True, "schema": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["answered", "insufficient_context"]},
                    "paragraphs": {"type": "array", "items": {
                        "type": "object",
                        "properties": {"text": {"type": "string"},
                                       "source_ids": {"type": "array", "items": {"type": "string"}}},
                        "required": ["text", "source_ids"], "additionalProperties": False}},
                },
                "required": ["status", "paragraphs"], "additionalProperties": False,
            }},
        }
        assert body["stream"] is False
        assert body["temperature"] == 0
        assert set(body) == {"model", "messages", "response_format", "max_completion_tokens",
                             "temperature", "stream"}
        assert body["messages"] == [{"role": "system", "content": "Return JSON."},
                                    {"role": "user", "content": "Evidence"}]
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": "{}"}}]})
    provider = GroqProvider("test-placeholder", httpx.MockTransport(handler))
    assert await provider.generate(REQUEST) == "{}"
    assert "test-placeholder" not in repr(provider)


@pytest.mark.parametrize("output", [
    {"status": "answered", "paragraphs": [{"text": "Supported fact.", "source_ids": ["S1"]}]},
    {"status": "insufficient_context", "paragraphs": []},
])
@pytest.mark.asyncio
async def test_json_validate_failed_regression_uses_strict_outputs(output):
    # Reproduce the observed rejection if the adapter regresses to JSON Object Mode.
    def handler(request):
        body = json.loads(request.content)
        if body["response_format"]["type"] == "json_object":
            return httpx.Response(400, json={"error": {
                "type": "invalid_request_error", "code": "json_validate_failed",
                "message": "Failed to generate JSON. Please adjust your prompt."}})
        assert body["response_format"]["type"] == "json_schema"
        assert body["response_format"]["json_schema"]["strict"] is True
        assert body["model"] == "openai/gpt-oss-20b"
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop",
                              "message": {"content": json.dumps(output)}}]})

    from api.rag.citations import validate_output
    request = GenerationRequest("Return JSON.", "Evidence S1", "openai/gpt-oss-20b", 800, 2)
    raw = await GroqProvider("test-placeholder", httpx.MockTransport(handler)).generate(request)
    assert validate_output(raw, frozenset({"S1"})).status == output["status"]


@pytest.mark.parametrize("status,code,reason", [
    (401, None, "authentication_failed"), (403, None, "authentication_failed"),
    (402, None, "quota_exceeded"), (429, None, "rate_limited"),
    (429, "insufficient_quota", "quota_exceeded"), (429, "quota_exceeded", "quota_exceeded"),
    (500, None, "provider_unavailable"), (503, None, "provider_unavailable"),
    (400, None, "provider_request_rejected"), (302, None, "provider_request_rejected"),
])
@pytest.mark.asyncio
async def test_http_errors_safe(status, code, reason):
    provider = GroqProvider("test-placeholder", httpx.MockTransport(lambda request: httpx.Response(
        status, json={"error": {"code": code, "message": "private test-placeholder"}})))
    with pytest.raises(ProviderFailure) as caught:
        await provider.generate(REQUEST)
    assert caught.value.reason == reason and "test-placeholder" not in str(caught.value)


@pytest.mark.parametrize("error,reason", [(httpx.ReadTimeout("private"), "provider_timeout"),
                                         (httpx.ConnectError("private"), "provider_unavailable")])
@pytest.mark.asyncio
async def test_transport_errors(error, reason):
    def handler(request):
        raise error
    with pytest.raises(ProviderFailure) as caught:
        await GroqProvider("test-placeholder", httpx.MockTransport(handler)).generate(REQUEST)
    assert str(caught.value) == reason


@pytest.mark.parametrize("body", ["not json", "{}", '{"choices":[]}',
    '{"choices":[{"finish_reason":"length","message":{"content":"{}"}}]}',
    '{"choices":[{"finish_reason":"stop","message":{"content":null}}]}',
    '{"choices":[{"finish_reason":"stop","message":{"content":"test-placeholder"}}]}',
    pytest.param("x" * 131073, id="oversized-envelope"),
])
@pytest.mark.asyncio
async def test_malformed_envelope_and_secret_echo_rejected(body):
    provider = GroqProvider("test-placeholder", httpx.MockTransport(lambda request: httpx.Response(200, text=body)))
    with pytest.raises(InvalidGeneration) as caught:
        await provider.generate(REQUEST)
    assert str(caught.value) == "malformed_provider_response"


@pytest.mark.asyncio
async def test_rejection_records_safe_diagnostic(caplog):
    error = {"type": "invalid_request_error", "code": "invalid_parameter",
             "message": "Parameter response_format is incompatible.\nUse JSON mode."}
    provider = GroqProvider("test-placeholder", httpx.MockTransport(
        lambda request: httpx.Response(400, json={"error": error, "private": "raw-body-marker"})))
    with pytest.raises(ProviderFailure, match="provider_request_rejected"):
        await provider.generate(REQUEST)
    record = next(r for r in caplog.records if r.name == "api.rag.groq")
    diagnostic = json.loads(record.getMessage().split(": ", 1)[1])
    assert diagnostic.pop("request") == {
        "model": "runtime-selected-model", "response_format_type": "json_schema", "strict": True,
        "max_completion_tokens": 800, "system_message_bytes": len(REQUEST.system.encode("utf-8")),
        "user_message_bytes": len(REQUEST.user.encode("utf-8")), "selected_source_count": None,
    }
    assert diagnostic == {"http_status": 400, "error_type": error["type"],
                          "error_code": error["code"],
                          "error_message": error["message"].replace("\n", " ")}
    assert "raw-body-marker" not in caplog.text
    assert record.exc_info is None


@pytest.mark.parametrize("value", [
    "echo test-placeholder", "Authorization: Bearer different-credential",
    "api_key=another-credential", "gsk_othercredential", "password=private",
    "x" * 600 + "test-placeholder", {"Authorization": "private"}, ["test-placeholder"],
])
@pytest.mark.asyncio
async def test_diagnostic_redacts_all_error_fields(value, caplog):
    provider = GroqProvider("test-placeholder", httpx.MockTransport(lambda request: httpx.Response(
        400, json={"error": dict.fromkeys(("type", "code", "message"), value),
                   "headers": {"Authorization": "Bearer test-placeholder"}})))
    with pytest.raises(ProviderFailure, match="provider_request_rejected"):
        await provider.generate(REQUEST)
    diagnostic = json.loads(caplog.records[-1].getMessage().split(": ", 1)[1])
    expected = "[redacted sensitive diagnostic]" if isinstance(value, str) else None
    diagnostic.pop("request")
    assert diagnostic == dict(http_status=400, error_type=expected, error_code=expected,
                              error_message=expected)
    assert "test-placeholder" not in caplog.text and "Authorization" not in caplog.text


@pytest.mark.parametrize("body", [b"Authorization: Bearer test-placeholder", b"[]",
                                       b'{"error":null}', b'{"error":{"message":123}}'])
@pytest.mark.asyncio
async def test_non_structured_error_has_no_raw_fallback(body, caplog):
    provider = GroqProvider("test-placeholder", httpx.MockTransport(
        lambda request: httpx.Response(400, content=body)))
    with pytest.raises(ProviderFailure, match="provider_request_rejected"):
        await provider.generate(REQUEST)
    diagnostic = json.loads(caplog.records[-1].getMessage().split(": ", 1)[1])
    diagnostic.pop("request")
    assert diagnostic == dict(http_status=400, error_type=None, error_code=None, error_message=None)
    assert "test-placeholder" not in caplog.text


@pytest.mark.asyncio
async def test_diagnostic_is_bounded_and_single_line(caplog):
    provider = GroqProvider("test-placeholder", httpx.MockTransport(lambda request: httpx.Response(
        400, json={"error": {"message": "bad\n\r\x1b" + "x" * 1000}})))
    with pytest.raises(ProviderFailure):
        await provider.generate(REQUEST)
    diagnostic = json.loads(caplog.records[-1].getMessage().split(": ", 1)[1])
    assert len(diagnostic["error_message"]) == 512
    assert diagnostic["error_message"].isprintable()


@pytest.mark.asyncio
async def test_ask_never_returns_raw_groq_rejection(monkeypatch, caplog):
    from unittest.mock import AsyncMock
    import api.main as main
    from api.rag.groq import get_generation_provider
    from api.rag.models import RagSource

    monkeypatch.setenv("RAG_ENABLED", "true")
    source = RagSource(source_id="S1", chunk_id="chunk1", podcast_id="show", episode_id="ep",
                       show_name="Show", episode_name="Episode",
                       clip_start_ms=0, clip_end_ms=120000, excerpt="Machine learning.")
    monkeypatch.setattr("api.rag.routes.retrieve", AsyncMock(return_value=[source]))
    provider = GroqProvider("test-placeholder", httpx.MockTransport(lambda request: httpx.Response(
        400, json={"error": {"type": "invalid_request_error", "code": "json_validate_failed",
                               "message": "Authorization: Bearer test-placeholder",
                               "failed_generation": "diagnostic-only-generated-text"},
                   "unsafe": "raw-body-marker"})))
    app = main.create_app()
    app.dependency_overrides[get_generation_provider] = lambda: provider
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/ask", json={"question": "What is machine learning?"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "generation_unavailable"
    assert data["reason"] == "provider_request_rejected"
    assert data["answer"] is None and data["sources"][0]["source_id"] == "S1"
    for private in ("test-placeholder", "Authorization", "json_validate_failed", "raw-body-marker",
                    "failed_generation", "diagnostic-only-generated-text"):
        assert private not in response.text
    assert "test-placeholder" not in caplog.text and "Authorization" not in caplog.text
    assert "diagnostic-only-generated-text" in caplog.text


@pytest.mark.asyncio
async def test_failed_generation_and_request_facts(caplog):
    failed = '{\n"status": "answered",\r\n"paragraphs": []\n}'
    user = json.dumps({"question": "Why?", "TRANSCRIPT_EVIDENCE": [
        {"source_id": "S1", "transcript": "private-transcript-marker"},
        {"source_id": "S2", "transcript": "Unicode: é"}]}, ensure_ascii=False)
    request = GenerationRequest("System é", user, "openai/gpt-oss-20b", 800, 2)
    provider = GroqProvider("test-placeholder", httpx.MockTransport(lambda request: httpx.Response(
        400, json={"error": {"code": "json_validate_failed", "failed_generation": failed,
                               "unrelated": "never-dump-marker"}, "other": "never-dump-marker"})))
    with pytest.raises(ProviderFailure, match="provider_request_rejected"):
        await provider.generate(request)
    line = caplog.records[-1].getMessage()
    diagnostic = json.loads(line.split(": ", 1)[1])
    assert diagnostic["failed_generation"] == json.dumps(failed, ensure_ascii=True)[1:-1]
    assert "\n" not in line and "\r" not in line
    assert diagnostic["request"] == {
        "model": "openai/gpt-oss-20b", "response_format_type": "json_schema", "strict": True,
        "max_completion_tokens": 800, "system_message_bytes": len(request.system.encode("utf-8")),
        "user_message_bytes": len(user.encode("utf-8")), "selected_source_count": 2,
    }
    assert "private-transcript-marker" not in line and "never-dump-marker" not in line


@pytest.mark.parametrize("failed", ["x" * 3000, "\n" * 2000, "é" * 2000])
@pytest.mark.asyncio
async def test_failed_generation_escaped_length_limit(failed, caplog):
    provider = GroqProvider("test-placeholder", httpx.MockTransport(lambda request: httpx.Response(
        400, json={"error": {"code": "json_validate_failed", "failed_generation": failed}})))
    with pytest.raises(ProviderFailure):
        await provider.generate(REQUEST)
    line = caplog.records[-1].getMessage()
    value = json.loads(line.split(": ", 1)[1])["failed_generation"]
    assert len(value) == 2000 and value.isprintable()
    assert "\n" not in line and "\r" not in line


@pytest.mark.parametrize("failed", [
    "test-placeholder", "Authorization: Basic private-value", "Bearer private-value",
    '{"api_key":"private-value"}', "gsk_private-value", "sk-private-value",
    '{"access_token":"private-value"}', '{"credentials":{"value":"private-value"}}',
    {"refresh_token": "private-value"}, "x" * 2500 + "test-placeholder",
])
@pytest.mark.asyncio
async def test_failed_generation_credentials_redacted(failed, caplog):
    provider = GroqProvider("test-placeholder", httpx.MockTransport(lambda request: httpx.Response(
        400, json={"error": {"code": "json_validate_failed", "failed_generation": failed}})))
    with pytest.raises(ProviderFailure):
        await provider.generate(REQUEST)
    line = caplog.records[-1].getMessage()
    assert json.loads(line.split(": ", 1)[1])["failed_generation"] == "[redacted sensitive diagnostic]"
    assert "test-placeholder" not in line and "private-value" not in line
    assert "Authorization" not in line


@pytest.mark.parametrize("error", [
    {"code": "other_error", "failed_generation": "never-dump-marker"},
    {"code": "json_validate_failed"},
])
@pytest.mark.asyncio
async def test_failed_generation_only_for_matching_code_and_present_field(error, caplog):
    provider = GroqProvider("test-placeholder", httpx.MockTransport(lambda request: httpx.Response(
        400, json={"error": error})))
    with pytest.raises(ProviderFailure):
        await provider.generate(REQUEST)
    line = caplog.records[-1].getMessage()
    assert "failed_generation" not in json.loads(line.split(": ", 1)[1])
    assert "never-dump-marker" not in line
