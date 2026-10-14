"""Task 3 (monthly employee score) and Task 4 (employee ranking).

Scoring rule
------------
Each message contributes   Positive = +1, Negative = -1, Neutral = 0.
Messages are grouped by (employee, calendar month) using the message date
truncated to YYYY-MM; the monthly score is the sum of contributions within
that month only. Because each month is its own group, the score
automatically "resets" to 0 at the beginning of every month.

Ranking rule
------------
For every month:
* Top-3 positive  = highest monthly score; ties broken alphabetically by
  employee e-mail (sort by score DESC, then name ASC).
* Top-3 negative  = lowest monthly score; ties broken alphabetically
  (sort by score ASC i.e. most negative first, then name ASC).
"""
from __future__ import annotations

import pandas as pd

from . import config as C


def add_message_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["message_score"] = out["sentiment"].map(C.SCORE_MAP).astype(int)
    return out


def monthly_scores(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (employee, month) with score and supporting counts."""
    if "message_score" not in df:
        df = add_message_scores(df)
    g = df.groupby([C.COL_EMPLOYEE, "month"])
    out = g.agg(
        monthly_score=("message_score", "sum"),
        message_count=("message_score", "size"),
        positive_count=("sentiment", lambda s: (s == "Positive").sum()),
        negative_count=("sentiment", lambda s: (s == "Negative").sum()),
        neutral_count=("sentiment", lambda s: (s == "Neutral").sum()),
    ).reset_index()
    return out.rename(columns={C.COL_EMPLOYEE: "employee"}).sort_values(["month", "employee"]).reset_index(drop=True)


def rank_month(scores: pd.DataFrame, month: str, n: int = C.TOP_N) -> tuple[pd.DataFrame, pd.DataFrame]:
    m = scores[scores["month"] == month]
    top_pos = m.sort_values(["monthly_score", "employee"], ascending=[False, True]).head(n)
    top_neg = m.sort_values(["monthly_score", "employee"], ascending=[True, True]).head(n)
    return top_pos, top_neg


def monthly_rankings(scores: pd.DataFrame, n: int = C.TOP_N) -> pd.DataFrame:
    """Long table: month, list_type (Top Positive / Top Negative), rank, employee, score."""
    rows = []
    for month in sorted(scores["month"].unique()):
        top_pos, top_neg = rank_month(scores, month, n)
        for label, frame in (("Top Positive", top_pos), ("Top Negative", top_neg)):
            for rank, (_, r) in enumerate(frame.iterrows(), start=1):
                rows.append(
                    {"month": month, "list_type": label, "rank": rank,
                     "employee": r["employee"], "monthly_score": int(r["monthly_score"])}
                )
    return pd.DataFrame(rows)


def overall_ranking(scores: pd.DataFrame) -> pd.DataFrame:
    """Whole-period view (sum of monthly scores) - used for the README summary."""
    tot = scores.groupby("employee")["monthly_score"].sum().reset_index(name="total_score")
    return tot.sort_values(["total_score", "employee"], ascending=[False, True]).reset_index(drop=True)
