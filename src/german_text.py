"""German text preprocessing shared by both rankers.

``preprocess`` turns raw German text into normalized, stemmed tokens, so the
from-scratch and scikit-learn rankers differ only in their scoring math.
"""
import re

from snowballstemmer import stemmer

# Small inline German stopword list — kept here (not a dependency) so the
# preprocessing stays fully transparent for the walkthrough.
STOPWORDS = {
    "der", "die", "das", "den", "dem", "des", "ein", "eine", "einen", "einem",
    "einer", "eines", "und", "oder", "aber", "in", "im", "an", "am", "auf",
    "von", "vom", "zu", "zur", "zum", "mit", "bei", "für", "ist", "sind",
    "war", "waren", "wird", "werden", "wurde", "wurden", "es", "sich", "auch",
    "als", "wie", "so", "noch", "nur", "schon", "über", "unter", "durch",
    "gibt", "sowie", "dass", "nicht", "kein", "keine", "welche", "welcher",
    "welches",
}

_STEMMER = stemmer("german")
# Split on any run of characters that is not a German letter or digit;
# this also breaks hyphenated compounds like "PV-Anlagen".
_SPLIT = re.compile(r"[^a-zäöüß0-9]+")


def preprocess(text: str) -> list[str]:
    """Lowercase, split (incl. hyphens), drop stopwords, then stem."""
    raw = _SPLIT.split(text.lower())
    tokens = [t for t in raw if t and t not in STOPWORDS]
    return _STEMMER.stemWords(tokens)
