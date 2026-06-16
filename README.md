# Query↔Document Relevance Ranker

Ranks three German `.txt` documents by relevance to a query using **TF-IDF +
cosine similarity** — the classic, explainable retrieval baseline. No LLM, no
external APIs, no vector database.

## How it works
1. **Preprocess** (`src/german_text.py`): lowercase, split punctuation and
   hyphenated compounds, drop German stopwords, Snowball-DE stemming.
2. **TF-IDF**: term frequency weighted by inverse document frequency, so words
   common to all documents ("Deutschland", "Jahr") are down-weighted and topic
   words ("Wärmepumpe", "Smart", "Solarstrom") dominate.
3. **Cosine similarity**: query and document become weighted vectors; the cosine
   of the angle between them is the relevance score (length-normalized).

Two implementations share the same preprocessing:
- `src/rank_scratch.py` — pure Python, every step by hand (primary, for the walkthrough).
- `src/rank_sklearn.py` — scikit-learn equivalent; a test asserts both agree on order.

BM25 would be the more sophisticated alternative but is over-engineered for a
three-document corpus.

## Project layout
- `data/` — the three topic documents, read automatically.
- `src/german_text.py` — shared German preprocessing.
- `src/corpus.py` — loading documents/queries and printing rankings.
- `src/rank_scratch.py` — pure-Python TF-IDF + cosine ranker (+ CLI).
- `src/rank_sklearn.py` — scikit-learn ranker (+ CLI).
- `tests/` — unit tests and the end-to-end ranking checks.

## Setup
```
py -m pip install -r requirements.txt
```

## Run
```
py -X utf8 src/rank_scratch.py     # pure-Python version
py -X utf8 src/rank_sklearn.py     # scikit-learn version
```
(`-X utf8` lets the Windows console print umlauts; on macOS/Linux plain `python src/rank_scratch.py` is fine.)

## Test
```
py -m pytest -v
```

## Expected result
| Query | Top document |
|-------|--------------|
| Wärmepumpen / Heizungsmarkt | `waermepumpe.txt` |
| Photovoltaik / Solarstrom | `photovoltaik.txt` |
| Smart Meter | `smartmeter.txt` |

Note: the query word "Photovoltaik" never literally appears in its document
(which says "PV"); the match comes from the shared word "Solarstrom". This
illustrates why normalization matters and is a good talking point.
