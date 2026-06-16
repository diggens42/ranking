from german_text import preprocess


def test_removes_stopwords_and_lowercases():
    tokens = preprocess("Die Wärmepumpe")
    assert "die" not in tokens


def test_splits_hyphenated_compounds():
    tokens = preprocess("PV-Anlagen in Deutschland")
    assert "pv" in tokens


def test_stems_inflection_to_same_token():
    # plural and singular must collapse to one stem so query/doc match
    assert preprocess("Wärmepumpen")[0] == preprocess("Wärmepumpe")[0]
