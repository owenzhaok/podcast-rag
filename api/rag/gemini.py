"""Offline Gemini embedding adapter; no SDK, retries, or credential serialization."""

import json
import re

import httpx

from api.rag.embeddings import validate_embeddings


class GeminiEmbeddings:
    def __init__(self, model: str, api_key: str, dimensions: int = 768, timeout: int = 20, transport=None):
        if dimensions != 768 or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", model):
            raise ValueError("Gemini requires 768 dimensions and a valid model identifier")
        if not api_key:
            raise ValueError("Gemini embedding credentials are not configured")
        self.model, self.dimensions, self.timeout = model, dimensions, timeout
        self._api_key, self._transport = api_key, transport

    def embed(self, texts: list[str]) -> list[list[float]]:
        # Conservative 8192-byte bound avoids silent document truncation. No padding,
        # truncation, or preprocessing here: the caller provides canonical input.
        if any(not isinstance(text, str) or not text.strip() or len(text.encode("utf-8")) > 8192 for text in texts):
            raise ValueError("Gemini document input must be nonempty and at most 8192 UTF-8 bytes")
        vectors = []
        try:
            with httpx.Client(transport=self._transport, timeout=self.timeout,
                              follow_redirects=False, trust_env=False) as client:
                for text in texts:
                    with client.stream("POST",
                        f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:embedContent",
                        headers={"x-goog-api-key": self._api_key},
                        json={"content": {"parts": [{"text": text}]}, "outputDimensionality": 768},
                    ) as response:
                        if response.status_code != 200:
                            reason = {401: "authentication", 403: "authentication", 429: "rate limit"}.get(
                                response.status_code, "provider unavailable" if response.status_code >= 500 else "request rejected")
                            raise ValueError("Gemini embedding failed: " + reason)
                        body = bytearray()
                        for chunk in response.iter_bytes():
                            body.extend(chunk)
                            if len(body) > 131072:
                                raise ValueError("Gemini response exceeds size limit")
                        try:
                            values = json.loads(body)["embedding"]["values"]
                            validate_embeddings([values], 1, self.dimensions)
                        except (ValueError, KeyError, TypeError, OverflowError, RecursionError):
                            raise ValueError("Invalid Gemini embedding response") from None
                        vectors.append(values)
        except httpx.TimeoutException:
            raise ValueError("Gemini embedding timed out") from None
        except httpx.HTTPError:
            raise ValueError("Gemini embedding network failure") from None
        validate_embeddings(vectors, len(texts), self.dimensions)
        return vectors
