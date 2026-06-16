"""Pure-Python TF-IDF + cosine similarity document ranker.

No ML libraries: vocabulary, term frequency, inverse document frequency,
TF-IDF vectors and cosine similarity are all computed by hand so every step is
visible. (BM25 would be the more sophisticated — and for a three-document
corpus, over-engineered — alternative.)
"""
import math


def cosine_similarity(a: dict, b: dict) -> float:
    """Cosine similarity between two sparse term -> weight vectors."""
    shared = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in shared)
    norm_a = math.sqrt(sum(w * w for w in a.values()))
    norm_b = math.sqrt(sum(w * w for w in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
