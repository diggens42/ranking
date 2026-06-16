from rank_scratch import cosine_similarity


def test_cosine_identical_vectors_is_one():
    v = {"a": 1.0, "b": 2.0}
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-9


def test_cosine_disjoint_vectors_is_zero():
    assert cosine_similarity({"a": 1.0}, {"b": 1.0}) == 0.0


def test_cosine_empty_vector_is_zero():
    assert cosine_similarity({}, {"a": 1.0}) == 0.0
