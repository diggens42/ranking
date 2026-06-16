"""Pure-Python TF-IDF + cosine similarity document ranker.

No ML libraries: vocabulary, term frequency, inverse document frequency,
TF-IDF vectors and cosine similarity are all computed by hand so every step is
visible. (BM25 would be the more sophisticated — and for a three-document
corpus, over-engineered — alternative.)
"""
import math
from collections import Counter

from german_text import preprocess


def compute_idf(doc_tokens: list[list[str]]) -> dict[str, float]:
    """Smoothed IDF over the corpus: log(N / (1 + df)) + 1."""
    n_docs = len(doc_tokens)
    df = Counter()
    for tokens in doc_tokens:
        for term in set(tokens):
            df[term] += 1
    return {term: math.log(n_docs / (1 + doc_freq)) + 1
            for term, doc_freq in df.items()}


def tfidf_vector(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    """TF-IDF weights for one token list. TF = raw count / document length."""
    if not tokens:
        return {}
    counts = Counter(tokens)
    length = len(tokens)
    return {term: (count / length) * idf.get(term, 0.0)
            for term, count in counts.items()}


def cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    """Cosine similarity between two sparse term -> weight vectors."""
    shared = set(a) & set(b)
    dot = sum(a[term] * b[term] for term in shared)
    norm_a = math.sqrt(sum(w * w for w in a.values()))
    norm_b = math.sqrt(sum(w * w for w in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def rank(query: str, documents: dict[str, str]) -> list[tuple[str, float]]:
    """Rank documents ({name: text}) against a query, descending by score."""
    names = list(documents)
    doc_tokens = [preprocess(documents[name]) for name in names]
    idf = compute_idf(doc_tokens)
    doc_vectors = [tfidf_vector(tokens, idf) for tokens in doc_tokens]
    query_vector = tfidf_vector(preprocess(query), idf)
    scores = [(name, cosine_similarity(query_vector, doc_vector))
              for name, doc_vector in zip(names, doc_vectors)]
    # Sort by score descending; break ties by filename for a stable ordering.
    return sorted(scores, key=lambda pair: (-pair[1], pair[0]))


if __name__ == "__main__":
    import sys
    from corpus import load_documents, load_queries, print_ranking

    sys.stdout.reconfigure(encoding="utf-8")  # print umlauts on the Windows console
    documents = load_documents()
    for q in load_queries():
        print_ranking(q, rank(q, documents))
