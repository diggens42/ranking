"""Pure-Python TF-IDF + cosine similarity document ranker.

No ML libraries: vocabulary, term frequency, inverse document frequency,
TF-IDF vectors and cosine similarity are all computed by hand so every step is
visible. (BM25 would be the more sophisticated — and for a three-document
corpus, over-engineered — alternative.)
"""
import math
from collections import Counter

from german_text import preprocess


def compute_idf(doc_tokens: list) -> dict:
    """Smoothed IDF over the corpus: log(N / (1 + df)) + 1."""
    n_docs = len(doc_tokens)
    df = Counter()
    for tokens in doc_tokens:
        for term in set(tokens):
            df[term] += 1
    return {term: math.log(n_docs / (1 + d)) + 1 for term, d in df.items()}


def tfidf_vector(tokens: list, idf: dict) -> dict:
    """TF-IDF weights for one token list. TF = raw count / document length."""
    if not tokens:
        return {}
    counts = Counter(tokens)
    length = len(tokens)
    return {t: (c / length) * idf.get(t, 0.0) for t, c in counts.items()}


def cosine_similarity(a: dict, b: dict) -> float:
    """Cosine similarity between two sparse term -> weight vectors."""
    shared = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in shared)
    norm_a = math.sqrt(sum(w * w for w in a.values()))
    norm_b = math.sqrt(sum(w * w for w in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def rank(query: str, documents: dict) -> list:
    """Rank documents ({name: text}) against a query, descending by score."""
    names = list(documents)
    doc_tokens = [preprocess(documents[name]) for name in names]
    idf = compute_idf(doc_tokens)
    doc_vectors = [tfidf_vector(toks, idf) for toks in doc_tokens]
    query_vector = tfidf_vector(preprocess(query), idf)
    scores = [(name, cosine_similarity(query_vector, dv))
              for name, dv in zip(names, doc_vectors)]
    return sorted(scores, key=lambda pair: pair[1], reverse=True)
