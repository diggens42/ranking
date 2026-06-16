"""scikit-learn TF-IDF + cosine ranker — same inputs and preprocessing as the
from-scratch version. Scores differ slightly (sklearn uses a different IDF
smoothing), but it produces the same ranking order, cross-checking the
hand-written version against a trusted library."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from german_text import preprocess


def rank(query: str, documents: dict) -> list:
    """Rank documents ({name: text}) against a query, descending by score."""
    names = list(documents)
    # preprocess already lowercases and tokenizes, so disable sklearn's own.
    vectorizer = TfidfVectorizer(tokenizer=preprocess,
                                 token_pattern=None,
                                 lowercase=False)
    doc_matrix = vectorizer.fit_transform(documents[n] for n in names)
    query_matrix = vectorizer.transform([query])
    scores = cosine_similarity(query_matrix, doc_matrix)[0]
    ranking = sorted(zip(names, scores), key=lambda pair: pair[1], reverse=True)
    return [(name, float(score)) for name, score in ranking]


if __name__ == "__main__":
    from corpus import load_documents, load_queries, print_ranking

    documents = load_documents()
    for q in load_queries():
        print_ranking(q, rank(q, documents))
