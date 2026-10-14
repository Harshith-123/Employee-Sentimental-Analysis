"""Task 2 - Exploratory Data Analysis helpers (tables + numbers).

Charts live in visualization.py; this module returns the tabular facts
that the notebook prints and the report quotes.
"""
from __future__ import annotations

import pandas as pd

from . import config as C


def structure(raw: pd.DataFrame) -> pd.DataFrame:
    """Column-level overview: dtype, missing, unique values."""
    return pd.DataFrame({
        "dtype": raw.dtypes.astype(str),
        "non_null": raw.notna().sum(),
        "missing": raw.isna().sum(),
        "unique": raw.nunique(),
    })


def overview(df: pd.DataFrame, raw: pd.DataFrame) -> dict:
    return {
        "records": int(len(raw)),
        "columns": list(raw.columns),
        "duplicate_rows": int(raw.duplicated().sum()),
        "employees": int(df[C.COL_EMPLOYEE].nunique()),
        "date_min": str(df[C.COL_DATE].min().date()),
        "date_max": str(df[C.COL_DATE].max().date()),
        "months": int(df["month"].nunique()),
        "median_words": float(df["word_count"].median()),
        "empty_bodies": int((df["clean_body"].str.len() == 0).sum()),
    }


def sentiment_counts(df: pd.DataFrame) -> pd.DataFrame:
    c = df["sentiment"].value_counts().reindex(list(C.LABELS), fill_value=0)
    return pd.DataFrame({"count": c, "share": (c / c.sum()).round(3)})


def per_employee(df: pd.DataFrame) -> pd.DataFrame:
    t = df.groupby([C.COL_EMPLOYEE, "sentiment"]).size().unstack(fill_value=0)
    t = t.reindex(columns=list(C.LABELS), fill_value=0)
    t["total"] = t.sum(axis=1)
    t["negative_share"] = (t["Negative"] / t["total"]).round(3)
    return t.sort_values("total", ascending=False)


def yearly_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    y = df.assign(year=df[C.COL_DATE].dt.year)
    t = y.groupby(["year", "sentiment"]).size().unstack(fill_value=0).reindex(columns=list(C.LABELS), fill_value=0)
    t["avg_score"] = (t["Positive"] - t["Negative"]) / t.sum(axis=1)
    return t.round(3)


def length_stats(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("sentiment")[["word_count", "char_length"]].median().reindex(list(C.LABELS))
