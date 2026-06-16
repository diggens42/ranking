import rank_scratch
import rank_sklearn
from corpus import load_documents, load_queries


def test_sklearn_top_documents_match_expected():
    docs = load_documents()
    expected = ["waermepumpe.txt", "photovoltaik.txt", "smartmeter.txt"]
    for query, exp in zip(load_queries(), expected):
        assert rank_sklearn.rank(query, docs)[0][0] == exp


def test_scratch_and_sklearn_agree_on_order():
    docs = load_documents()
    for query in load_queries():
        scratch_order = [n for n, _ in rank_scratch.rank(query, docs)]
        sklearn_order = [n for n, _ in rank_sklearn.rank(query, docs)]
        assert scratch_order == sklearn_order
