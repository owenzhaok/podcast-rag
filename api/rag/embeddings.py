"""Offline embedding boundary. Fake vectors test plumbing, not semantic similarity."""

import hashlib
import math
from typing import Protocol


class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per exact text, in input order."""
        ...


class FakeEmbeddings:
    def __init__(self, dimensions: int, model: str, revision: str):
        self.dimensions, self.model, self.revision = dimensions, model, revision

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            seed = f"{self.model}\0{self.revision}\0{text}".encode("utf-8")
            raw = hashlib.shake_256(seed).digest(self.dimensions * 2)
            vector = [(int.from_bytes(raw[i:i + 2], "big") + 1) / 65536
                      for i in range(0, len(raw), 2)]
            norm = math.sqrt(sum(x * x for x in vector))
            vectors.append([x / norm for x in vector])
        return vectors


def validate_embeddings(vectors, count: int, dimensions: int) -> None:
    if not isinstance(vectors, list) or len(vectors) != count:
        raise ValueError("Embedding batch count mismatch")
    for vector in vectors:
        if not isinstance(vector, list) or len(vector) != dimensions:
            raise ValueError("Embedding dimension mismatch")
        if any(type(x) not in (int, float) or not math.isfinite(x) or abs(x) > 3.4e38 for x in vector):
            raise ValueError("Embedding must contain finite float32 values")
        if not any(x != 0 for x in vector):
            raise ValueError("Cosine embedding must be nonzero")
