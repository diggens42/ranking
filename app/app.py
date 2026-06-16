"""Minimal Streamlit UI for the document ranker — for live demonstration.

Kept separate from `src/`: it only loads the corpus and calls the existing
ranking functions, so the scoring logic stays the focus and is not duplicated.

Run with:  py -m streamlit run app/app.py
"""
import sys
from pathlib import Path

# Make the rankers in ../src importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import streamlit as st

import rank_scratch
import rank_sklearn
from corpus import load_documents, load_queries

RANKERS = {
    "Pure Python (from scratch)": rank_scratch.rank,
    "scikit-learn": rank_sklearn.rank,
}


@st.cache_data
def get_documents() -> dict[str, str]:
    return load_documents()


@st.cache_data
def get_suggested_queries() -> list[str]:
    return load_queries()


st.set_page_config(page_title="Dokumenten-Ranking", page_icon="🔎")
st.title("🔎 Dokumenten-Relevanz-Ranking")
st.caption(
    "TF-IDF + Kosinus-Ähnlichkeit: die Query wird mit jedem Dokument verglichen "
    "und nach Relevanz-Score absteigend sortiert."
)

documents = get_documents()

ranker_name = st.radio("Methode", list(RANKERS), horizontal=True)

st.write("**Beispiel-Queries** (zum Ausprobieren anklicken):")
for suggestion in get_suggested_queries():
    if st.button(suggestion, use_container_width=True):
        st.session_state["query"] = suggestion

query = st.text_input("Query", key="query", placeholder="Frage eingeben …")

if query:
    ranking = RANKERS[ranker_name](query, documents)
    table = pd.DataFrame(
        [(i + 1, name, round(score, 4)) for i, (name, score) in enumerate(ranking)],
        columns=["Rang", "Dokument", "Score"],
    )
    top_name, top_score = ranking[0]
    if top_score > 0:
        st.success(f"Relevantestes Dokument: **{top_name}** (Score {top_score:.4f})")
    else:
        st.warning("Keine inhaltliche Übereinstimmung gefunden (alle Scores 0).")
    st.table(table.set_index("Rang"))
    st.bar_chart(table.set_index("Dokument")["Score"])
