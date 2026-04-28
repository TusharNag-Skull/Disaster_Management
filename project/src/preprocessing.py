import re

import pandas as pd
from nltk.stem import WordNetLemmatizer

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@[A-Za-z0-9_]+")
EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "]+",
    flags=re.UNICODE,
)
NON_ALPHA_RE = re.compile(r"[^a-zA-Z\s]")
LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    x = text.lower().strip()
    x = URL_RE.sub(" ", x)
    x = MENTION_RE.sub(" ", x)
    x = EMOJI_RE.sub(" ", x)
    x = NON_ALPHA_RE.sub(" ", x)
    tokens = " ".join(x.split()).split()
    if not tokens:
        return ""
    # Lemmatization for normalized lexical features.
    lemmas = []
    for tok in tokens:
        try:
            lemmas.append(LEMMATIZER.lemmatize(tok))
        except LookupError:
            lemmas.append(tok)
    return " ".join(lemmas)


def ensure_dataframe_schema(df: pd.DataFrame) -> pd.DataFrame:
    if "text" not in df.columns:
        df["text"] = ""
    if "crisis_type" not in df.columns:
        df["crisis_type"] = "unknown"
    if "severity_score" not in df.columns:
        df["severity_score"] = 0
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def parse_text_input(text: str) -> pd.DataFrame:
    return pd.DataFrame({"text": [text]})
