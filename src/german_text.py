"""German text preprocessing shared by both rankers."""
import re

from snowballstemmer import stemmer

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
_SPLIT = re.compile(r"[^a-zäöüß0-9]+")


def preprocess(text: str) -> list[str]:
    """Lowercase, split (incl. hyphens), drop stopwords, then stem."""
    raw = _SPLIT.split(text.lower())
    tokens = [t for t in raw if t and t not in STOPWORDS]
    return _STEMMER.stemWords(tokens)
