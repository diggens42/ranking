from corpus import load_documents, load_queries
from rank_scratch import cosine_similarity, compute_idf, rank, tfidf_vector


def test_cosine_identical_vectors_is_one():
    v = {"a": 1.0, "b": 2.0}
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-9


def test_cosine_disjoint_vectors_is_zero():
    assert cosine_similarity({"a": 1.0}, {"b": 1.0}) == 0.0


def test_cosine_empty_vector_is_zero():
    assert cosine_similarity({}, {"a": 1.0}) == 0.0


def test_idf_downweights_common_terms():
    # 'a' appears in every doc; 'b' is rare -> 'a' must get a lower idf
    docs = [["a", "b"], ["a", "c"], ["a", "d"]]
    idf = compute_idf(docs)
    assert idf["a"] < idf["b"]


def test_tfidf_vector_weights_terms_by_idf():
    idf = {"x": 2.0, "y": 1.0}
    vec = tfidf_vector(["x", "y", "y"], idf)
    # x: (1/3)*2 ; y: (2/3)*1
    assert abs(vec["x"] - (1 / 3) * 2.0) < 1e-9
    assert abs(vec["y"] - (2 / 3) * 1.0) < 1e-9


def test_tfidf_vector_empty_is_empty():
    assert tfidf_vector([], {"x": 1.0}) == {}


def test_query_with_no_usable_terms_scores_zero():
    # A query of only stopwords leaves an empty vector -> all scores 0, no crash.
    docs = load_documents()
    ranking = rank("und der die", docs)
    assert len(ranking) == len(docs)
    assert all(score == 0.0 for _, score in ranking)


def test_each_query_top_document_is_expected():
    docs = load_documents()
    queries = load_queries()
    expected = ["waermepumpe.txt", "photovoltaik.txt", "smartmeter.txt"]
    assert len(queries) == 3
    for query, exp in zip(queries, expected):
        ranking = rank(query, docs)
        assert ranking[0][0] == exp
        # scores must be sorted descending
        scores = [s for _, s in ranking]
        assert scores == sorted(scores, reverse=True)
