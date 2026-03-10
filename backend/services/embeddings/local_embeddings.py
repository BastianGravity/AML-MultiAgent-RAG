"""Local deterministic embedding utilities.

This module provides a provider-agnostic fallback embedding generator that
allows the RAG pipeline to run when remote embedding endpoints are unavailable.
"""

import hashlib
from typing import List


def _hash_to_unit_float(seed: str) -> float:
    """Convert a string seed into a deterministic float in [-1.0, 1.0]."""
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    value = int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)
    return (value * 2.0) - 1.0


def deterministic_text_embedding(text: str, dimensions: int = 1536) -> List[float]:
    """Generate a deterministic embedding vector from input text.

    The vector is normalized to unit length so cosine similarity remains stable.
    """
    vector = [
        _hash_to_unit_float(f"{i}:{text}")
        for i in range(dimensions)
    ]

    norm = sum(v * v for v in vector) ** 0.5
    if norm == 0:
        return vector
    return [v / norm for v in vector]
