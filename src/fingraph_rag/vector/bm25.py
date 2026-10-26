"""Deterministic lexical representations and a corpus-aware BM25 baseline."""

import hashlib
import math
import re
from collections import Counter
from collections.abc import Mapping


def tokenize(text: str) -> list[str]:
    return re.findall(r"[\w]+", text.casefold())


def sparse_vector(text: str) -> dict[str, list[int] | list[float]]:
    """Stable hashed TF coordinates; collisions aggregate and never duplicate indices.

    This is a lexical sparse vector, not corpus-calibrated BM25. BM25 scoring is
    provided separately by Bm25Index, where document frequencies are available.
    """
    counts: Counter[int] = Counter()
    for token in tokenize(text):
        index = int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], "big")
        counts[index] += 1
    indices = sorted(counts)
    return {"indices": indices, "values": [1 + math.log(counts[index]) for index in indices]}


class Bm25Index:
    def __init__(self, documents: Mapping[str, str], k1: float = 1.2, b: float = 0.75):
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("invalid BM25 parameters")
        self.documents = {identity: Counter(tokenize(text)) for identity, text in documents.items()}
        self.k1, self.b = k1, b
        self.average_length = sum(
            map(lambda item: sum(item.values()), self.documents.values())
        ) / max(1, len(self.documents))
        self.frequencies: Counter[str] = Counter()
        for terms in self.documents.values():
            self.frequencies.update(terms.keys())

    def search(self, query: str, limit: int = 10) -> list[dict[str, object]]:
        if limit < 1:
            raise ValueError("limit must be positive")
        scores = []
        for identity, terms in self.documents.items():
            score = 0.0
            for term in set(tokenize(query)):
                frequency = terms[term]
                if not frequency:
                    continue
                df = self.frequencies[term]
                idf = math.log(1 + (len(self.documents) - df + 0.5) / (df + 0.5))
                normalization = self.k1 * (
                    1 - self.b + self.b * sum(terms.values()) / max(self.average_length, 1)
                )
                score += idf * frequency * (self.k1 + 1) / (frequency + normalization)
            if score:
                scores.append({"evidence_id": identity, "score": score})
        return sorted(scores, key=lambda item: (-item["score"], item["evidence_id"]))[:limit]
