"""Task 1 - Sentiment labeling (Positive / Negative / Neutral).

Three interchangeable back-ends, selected with SENTIMENT_METHOD:

* ``vader`` (default) - rule/lexicon based model tuned for short, informal
  text. Deterministic, offline, no GPU, so results are 100 % reproducible.
  Label rule (standard VADER convention):
      compound >=  0.05 -> Positive
      compound <= -0.05 -> Negative
      otherwise         -> Neutral
* ``transformer`` - pretrained RoBERTa sentiment classifier from Hugging
  Face (cardiffnlp/twitter-roberta-base-sentiment-latest). Needs
  ``pip install transformers torch``.
* ``llm`` - zero-shot classification with a large language model
  (Anthropic or OpenAI) using a fixed prompt at temperature 0. Responses
  are cached in data/processed/llm_cache.csv so reruns are free and
  reproducible.

Whatever the back-end, the output is the same: a ``sentiment`` column
(and a ``sentiment_score_raw`` confidence/polarity column).
"""
from __future__ import annotations

import hashlib
import json
from typing import Iterable

import pandas as pd

from . import config as C


# ------------------------------------------------------------------ VADER
def _vader_label(compound: float) -> str:
    if compound >= C.VADER_POS_THRESHOLD:
        return "Positive"
    if compound <= C.VADER_NEG_THRESHOLD:
        return "Negative"
    return "Neutral"


def label_vader(texts: Iterable[str]) -> pd.DataFrame:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    sia = SentimentIntensityAnalyzer()
    compounds = [sia.polarity_scores(t or "")["compound"] for t in texts]
    return pd.DataFrame(
        {"sentiment_score_raw": compounds, "sentiment": [_vader_label(c) for c in compounds]}
    )


# ------------------------------------------------------------ transformer
def label_transformer(texts: Iterable[str], batch_size: int = 32) -> pd.DataFrame:
    from transformers import pipeline  # optional dependency

    clf = pipeline("sentiment-analysis", model=C.TRANSFORMER_MODEL, truncation=True, max_length=512)
    preds = clf([t or "" for t in texts], batch_size=batch_size)
    mapping = {"positive": "Positive", "negative": "Negative", "neutral": "Neutral"}
    return pd.DataFrame(
        {
            "sentiment_score_raw": [p["score"] for p in preds],
            "sentiment": [mapping.get(p["label"].lower(), "Neutral") for p in preds],
        }
    )


# -------------------------------------------------------------------- LLM
LLM_PROMPT = (
    "You are labeling workplace e-mails for an employee-engagement study.\n"
    "Classify the overall sentiment the SENDER expresses in the message as exactly "
    "one word: Positive, Negative, or Neutral.\n"
    "- Positive: appreciation, enthusiasm, satisfaction, good news, friendly tone.\n"
    "- Negative: frustration, complaints, criticism, worry, bad news, conflict.\n"
    "- Neutral: purely informational, logistical or factual content.\n"
    "Answer with the single word only.\n\nMessage:\n\"\"\"{text}\"\"\""
)


def _llm_call(text: str) -> str:
    prompt = LLM_PROMPT.format(text=text[:4000])
    if C.LLM_PROVIDER == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=C.OPENAI_API_KEY)
        r = client.chat.completions.create(
            model=C.LLM_MODEL, temperature=0, max_tokens=3,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = r.choices[0].message.content
    else:
        import anthropic

        client = anthropic.Anthropic(api_key=C.ANTHROPIC_API_KEY)
        r = client.messages.create(
            model=C.LLM_MODEL, max_tokens=5, temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = r.content[0].text
    answer = (answer or "").strip().capitalize().rstrip(".")
    return answer if answer in C.LABELS else "Neutral"


def label_llm(texts: Iterable[str]) -> pd.DataFrame:
    cache_path = C.PROCESSED_DIR / "llm_cache.csv"
    cache = (
        dict(pd.read_csv(cache_path).values) if cache_path.exists() else {}
    )
    labels = []
    for t in texts:
        key = hashlib.sha1((t or "").encode()).hexdigest()
        if key not in cache:
            cache[key] = _llm_call(t or "")
        labels.append(cache[key])
    pd.DataFrame(list(cache.items()), columns=["hash", "label"]).to_csv(cache_path, index=False)
    return pd.DataFrame({"sentiment_score_raw": [None] * len(labels), "sentiment": labels})


# --------------------------------------------------------------- facade
BACKENDS = {"vader": label_vader, "transformer": label_transformer, "llm": label_llm}


def label_messages(df: pd.DataFrame, method: str | None = None, text_col: str = "clean_text") -> pd.DataFrame:
    """Return ``df`` with ``sentiment`` and ``sentiment_score_raw`` columns appended."""
    method = (method or C.SENTIMENT_METHOD).lower()
    if method not in BACKENDS:
        raise ValueError(f"Unknown SENTIMENT_METHOD '{method}'. Choose from {list(BACKENDS)}")
    res = BACKENDS[method](df[text_col].tolist())
    out = df.reset_index(drop=True).copy()
    out["sentiment_score_raw"] = res["sentiment_score_raw"].values
    out["sentiment"] = res["sentiment"].values
    out["sentiment_method"] = method
    return out


def describe_method(method: str | None = None) -> str:
    method = (method or C.SENTIMENT_METHOD).lower()
    return json.dumps(
        {
            "method": method,
            "vader_thresholds": [C.VADER_NEG_THRESHOLD, C.VADER_POS_THRESHOLD],
            "transformer_model": C.TRANSFORMER_MODEL,
            "llm": f"{C.LLM_PROVIDER}:{C.LLM_MODEL}",
        }
    )
