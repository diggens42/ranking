from corpus import load_documents, load_queries


def test_load_queries_tolerates_blank_lines(tmp_path):
    queries_file = tmp_path / "queries.txt"
    queries_file.write_text(
        "Query 1\n\nFirst question?\n\nQuery 2\nSecond question?\n",
        encoding="utf-8",
    )
    assert load_queries(queries_file) == ["First question?", "Second question?"]


def test_load_documents_reads_txt_folder(tmp_path):
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "b.txt").write_text("beta", encoding="utf-8")
    (tmp_path / "ignore.md").write_text("nope", encoding="utf-8")
    assert load_documents(tmp_path) == {"a.txt": "alpha", "b.txt": "beta"}
