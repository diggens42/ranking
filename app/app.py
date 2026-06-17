"""Minimal Streamlit UI for the document ranker.

Run with:  py -m streamlit run app/app.py
"""
import sys
from pathlib import Path

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

with st.sidebar:
    st.header("Methode")
    ranker_name = st.radio(
        "Ranking-Verfahren", list(RANKERS), label_visibility="collapsed"
    )
    st.markdown(
        "**So funktioniert's**\n\n"
        "1. Text normalisieren (Kleinschreibung, Stoppwörter, Stemming)\n"
        "2. **TF-IDF** gewichten – seltene Themenwörter zählen mehr\n"
        "3. **Kosinus-Ähnlichkeit** zwischen Query und Dokument"
    )

st.title("🔎 Dokumenten-Relevanz-Ranking")
st.caption(
    "Die Query wird mit jedem Dokument verglichen und nach Relevanz-Score "
    "absteigend sortiert."
)

documents = get_documents()

st.markdown("**Beispiel-Queries** – zum Ausprobieren anklicken:")
for suggestion in get_suggested_queries():
    if st.button(suggestion, use_container_width=True):
        st.session_state["query"] = suggestion

query = st.text_input("Query", key="query", placeholder="Frage eingeben …")

if query:
    ranking = RANKERS[ranker_name](query, documents)
    top_name, top_score = ranking[0]

    with st.container(border=True):
        if top_score > 0:
            st.success(
                f"Relevantestes Dokument: **{top_name}**  ·  Score {top_score:.4f}"
            )
        else:
            st.warning("Keine inhaltliche Übereinstimmung gefunden (alle Scores 0).")

        table = pd.DataFrame(
            [(i + 1, name, score) for i, (name, score) in enumerate(ranking)],
            columns=["Rang", "Dokument", "Score"],
        )
        st.dataframe(
            table,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Rang": st.column_config.NumberColumn(width="small"),
                "Dokument": st.column_config.TextColumn(),
                "Score": st.column_config.ProgressColumn(
                    "Relevanz",
                    format="%.4f",
                    min_value=0.0,
                    max_value=max(top_score, 1e-9),
                ),
            },
        )
