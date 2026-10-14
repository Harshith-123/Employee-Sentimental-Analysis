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
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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


LLM_CACHE_PATH = C.DATA_DIR / "llm_cache.csv"   # committed, so reviewers can reproduce without an API key
_LABEL_RE = re.compile(r"\b(positive|negative|neutral)\b", re.I)
_client = None


def _get_client():
    """Create the API client once (thread-safe enough for read-only reuse)."""
    global _client
    if _client is None:
        if C.LLM_PROVIDER == "openai":
            from openai import OpenAI

            if not C.OPENAI_API_KEY or C.OPENAI_API_KEY.startswith("your-"):
                raise RuntimeError("OPENAI_API_KEY is not set in .env")
            _client = OpenAI(api_key=C.OPENAI_API_KEY)
        else:
            import anthropic

            if not C.ANTHROPIC_API_KEY or C.ANTHROPIC_API_KEY.startswith("your-"):
                raise RuntimeError("ANTHROPIC_API_KEY is not set in .env")
            _client = anthropic.Anthropic(api_key=C.ANTHROPIC_API_KEY)
    return _client


def parse_label(answer: str) -> str:
    """Map the model's reply to a label; anything unparseable -> Neutral."""
    m = _LABEL_RE.search(answer or "")
    return m.group(1).capitalize() if m else "Neutral"


def _llm_call(text: str, retries: int = 6) -> str:
    """Classify one message, retrying with exponential backoff on API errors / rate limits."""
    prompt = LLM_PROMPT.format(text=(text or "")[:4000])
    client = _get_client()
    for attempt in range(retries):
        try:
            if C.LLM_PROVIDER == "openai":
                r = client.chat.completions.create(
                    model=C.LLM_MODEL, temperature=0,
                    messages=[{"role": "user", "content": prompt}],
                )
                return parse_label(r.choices[0].message.content)
            r = client.messages.create(
                model=C.LLM_MODEL, max_tokens=10, temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            return parse_label(r.content[0].text)
        except Exception as exc:  # rate limit, timeout, overloaded ...
            if attempt == retries - 1:
                raise RuntimeError(f"LLM call failed after {retries} attempts: {exc}") from exc
            time.sleep(min(60, 2 ** attempt))
    return "Neutral"  # pragma: no cover


def _save_cache(cache: dict) -> None:
    pd.DataFrame(sorted(cache.items()), columns=["hash", "label"]).to_csv(LLM_CACHE_PATH, index=False)


def label_llm(texts: Iterable[str], workers: int | None = None) -> pd.DataFrame:
    """Label with an LLM. Results are cached per message hash in data/llm_cache.csv:
    reruns cost nothing, an interrupted run resumes where it stopped, and anyone
    with the cache file reproduces the exact labels without an API key."""
    texts = [t or "" for t in texts]
    cache = dict(pd.read_csv(LLM_CACHE_PATH).values) if LLM_CACHE_PATH.exists() else {}
    keys = [hashlib.sha1(t.encode()).hexdigest() for t in texts]
    todo = {k: t for k, t in zip(keys, texts) if k not in cache}

    if todo:
        workers = workers or C.LLM_WORKERS
        print(f"LLM labeling: {len(todo)} new messages via {C.LLM_PROVIDER}:{C.LLM_MODEL} "
              f"({len(cache)} already cached, {workers} parallel requests)")
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_llm_call, t): k for k, t in todo.items()}
            for fut in as_completed(futures):
                cache[futures[fut]] = fut.result()
                done += 1
                if done % 100 == 0 or done == len(todo):
                    _save_cache(cache)   # checkpoint so a crash loses at most 100 calls
                    print(f"  {done}/{len(todo)} labeled")
    else:
        print(f"LLM labeling: all {len(texts)} messages found in {LLM_CACHE_PATH.name} (no API calls)")

    labels = [cache[k] for k in keys]
    return pd.DataFrame({"sentiment_score_raw": [float("nan")] * len(labels), "sentiment": labels})


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
