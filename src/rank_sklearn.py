"""scikit-learn TF-IDF + cosine ranker — same inputs and preprocessing as the
from-scratch version. Scores differ slightly (sklearn uses a different IDF
smoothing), but it produces the same ranking order, cross-checking the
hand-written version against a trusted library."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from german_text import preprocess


def rank(query: str, documents: dict[str, str]) -> list[tuple[str, float]]:
    """Rank documents ({name: text}) against a query, descending by score."""
    names = list(documents)
    # preprocess already lowercases and tokenizes, so disable sklearn's own.
    vectorizer = TfidfVectorizer(tokenizer=preprocess,
                                 token_pattern=None,
                                 lowercase=False)
    doc_matrix = vectorizer.fit_transform(documents[n] for n in names)
    query_matrix = vectorizer.transform([query])
    scores = cosine_similarity(query_matrix, doc_matrix)[0]
    # Sort by score descending; break ties by filename for a stable ordering.
    ranking = sorted(zip(names, scores), key=lambda pair: (-pair[1], pair[0]))
    return [(name, float(score)) for name, score in ranking]


if __name__ == "__main__":
    import sys
    from corpus import load_documents, load_queries, print_ranking

    sys.stdout.reconfigure(encoding="utf-8")  # print umlauts on the Windows console
    documents = load_documents()
    for q in load_queries():
        print_ranking(q, rank(q, documents))
