"""Load the corpus and queries, and pretty-print rankings."""
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _ROOT / "data"
QUERIES_FILE = _ROOT / "0_queries.txt"


def load_documents(folder: Path = DATA_DIR) -> dict[str, str]:
    """Read every .txt file in *folder* into a {filename: text} dict."""
    return {p.name: p.read_text(encoding="utf-8")
            for p in sorted(folder.glob("*.txt"))}


def load_queries(path: Path = QUERIES_FILE) -> list[str]:
    """Parse 'Query N' blocks: the next non-empty line after each header."""
    lines = path.read_text(encoding="utf-8").splitlines()
    queries = []
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("query"):
            for nxt in lines[i + 1:]:
                if nxt.strip():
                    queries.append(nxt.strip())
                    break
    return queries


def print_ranking(query: str, ranking: list[tuple[str, float]]) -> None:
    """Print one query's ranking as an aligned table."""
    print(f"\nQuery: {query}")
    print(f"  {'Dokument':<22}{'Score':>10}")
    print(f"  {'-' * 22}{'-' * 10}")
    for name, score in ranking:
        print(f"  {name:<22}{score:>10.4f}")
