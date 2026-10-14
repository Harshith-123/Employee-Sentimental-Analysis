"""Loading and cleaning of the raw employee-message dataset.

The raw bodies are Enron-style e-mails exported with *quoted-printable*
artifacts (``=20`` for a space, ``=\\n`` soft line breaks, ``=01)`` etc.).
These artifacts are noise for any NLP model, so we decode / strip them
before sentiment analysis while keeping the original text untouched in
the ``body`` column for traceability.
"""
from __future__ import annotations

import re

import pandas as pd

from . import config as C

_SOFT_BREAK = re.compile(r"=\n")                 # QP soft line break
_QP_HEX = re.compile(r"=([0-9A-Fa-f]{2})")       # =20, =3D, =01 ...
_URL = re.compile(r"https?://\S+|www\.\S+")
_EMAIL = re.compile(r"\S+@\S+\.\S+")
_FORWARD_HDR = re.compile(r"-{3,}\s*(Original Message|Forwarded by).*", re.I | re.S)
# quoted reply history starts at a header line such as "From: ...", "To: ...",
# "Enron North America Corp." or "Jane Doe@ENRON 11/28/2000 08:25 AM"
_QUOTED_HISTORY = re.compile(
    r"\n\s*(From:|To:|Sent:|Enron North America Corp\.|[\w .'-]+@\s*\w+\s+\d{1,2}/\d{1,2}/\d{2,4}).*",
    re.S,
)
_WS = re.compile(r"\s+")


def _decode_qp_char(m: re.Match) -> str:
    code = int(m.group(1), 16)
    # keep printable ASCII (=20 -> ' ', =3D -> '='); drop control bytes
    return chr(code) if 32 <= code < 127 else " "


def clean_text(text: str) -> str:
    """Return a normalised version of an e-mail body for NLP."""
    if not isinstance(text, str):
        return ""
    t = _SOFT_BREAK.sub("", text)
    t = _QP_HEX.sub(_decode_qp_char, t)
    t = _FORWARD_HDR.sub(" ", t)       # drop forwarded history
    t = _QUOTED_HISTORY.sub(" ", t)    # drop quoted reply history (other people's words)
    t = _URL.sub(" ", t)
    t = _EMAIL.sub(" ", t)
    t = _WS.sub(" ", t)
    return t.strip()


def load_raw(path=C.RAW_DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV exactly as delivered."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place test.csv in data/raw/ "
            "or set RAW_DATA_PATH in your .env file."
        )
    return pd.read_csv(path)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Type-cast, clean and enrich the raw frame.

    Adds: ``date`` (datetime), ``month`` (period YYYY-MM), ``clean_text``
    (subject + cleaned body), ``char_length``, ``word_count``.
    """
    out = df.copy()
    out[C.COL_DATE] = pd.to_datetime(out[C.COL_DATE], errors="coerce")
    out[C.COL_SUBJECT] = out[C.COL_SUBJECT].fillna("").astype(str)
    out[C.COL_BODY] = out[C.COL_BODY].fillna("").astype(str)
    out[C.COL_EMPLOYEE] = out[C.COL_EMPLOYEE].astype(str).str.strip().str.lower()

    subj = out[C.COL_SUBJECT].where(out[C.COL_SUBJECT] != "(No Subject)", "")
    out["clean_body"] = out[C.COL_BODY].map(clean_text)
    out["clean_text"] = (subj.map(clean_text) + ". " + out["clean_body"]).str.strip(". ").str.strip()

    out["month"] = out[C.COL_DATE].dt.to_period("M").astype(str)
    out["char_length"] = out["clean_body"].str.len()
    out["word_count"] = out["clean_body"].str.split().str.len().fillna(0).astype(int)
    out = out.dropna(subset=[C.COL_DATE]).sort_values(C.COL_DATE).reset_index(drop=True)
    return out
