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
py src/rank_scratch.py     # pure-Python version
py src/rank_sklearn.py     # scikit-learn version
```
(On macOS/Linux use `python` instead of `py`. The scripts set UTF-8 output
themselves, so umlauts print correctly on the Windows console.)

## Demo UI (optional)
A small Streamlit app (`app/app.py`) wraps the ranker for live demos: pick the
method (pure-Python or scikit-learn), click one of the three example queries or
type your own, and see the ranked table and score chart.
```
py -m streamlit run app/app.py
```
It only loads the corpus and calls the existing `rank` functions — no scoring
logic is duplicated.

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
