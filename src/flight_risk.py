"""Task 5 - Flight-risk identification.

Definition (from the brief): an employee is a flight risk if they sent
**4 or more Negative messages within any rolling 30-day period**,
irrespective of calendar months and irrespective of their score.

Implementation
--------------
For each employee we take the sorted dates of their Negative messages
and slide a window over them: if the 1st and the 4th message of any
consecutive group of four are at most 29 days apart (i.e. all four fall
inside one 30-day span: day d .. day d+29), the employee is flagged.
This two-pointer scan is O(n) per employee and catches windows that
straddle month boundaries, which a month-based groupby would miss.
"""
from __future__ import annotations

import pandas as pd

from . import config as C


def find_risk_windows(
    df: pd.DataFrame,
    min_neg: int = C.FLIGHT_RISK_MIN_NEGATIVES,
    window_days: int = C.FLIGHT_RISK_WINDOW_DAYS,
) -> pd.DataFrame:
    """Every qualifying window (employee, window_start, window_end, negatives_in_window)."""
    neg = df[df["sentiment"] == "Negative"][[C.COL_EMPLOYEE, C.COL_DATE]]
    rows = []
    span = pd.Timedelta(days=window_days - 1)
    for emp, g in neg.groupby(C.COL_EMPLOYEE):
        dates = g[C.COL_DATE].dt.normalize().sort_values().tolist()
        j = 0
        for i in range(len(dates)):
            # advance j to the last message within [dates[i], dates[i] + 29d]
            if j < i:
                j = i
            while j + 1 < len(dates) and dates[j + 1] - dates[i] <= span:
                j += 1
            count = j - i + 1
            if count >= min_neg:
                rows.append(
                    {"employee": emp, "window_start": dates[i].date(),
                     "window_end": (dates[i] + span).date(), "negatives_in_window": count}
                )
    return pd.DataFrame(rows, columns=["employee", "window_start", "window_end", "negatives_in_window"])


def flight_risk_employees(df: pd.DataFrame, windows: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per flagged employee with summary statistics."""
    windows = find_risk_windows(df) if windows is None else windows
    if windows.empty:
        return pd.DataFrame(columns=["employee", "max_negatives_in_30d", "first_flagged_window_start",
                                     "qualifying_windows", "total_negative_messages"])
    neg_tot = df[df["sentiment"] == "Negative"].groupby(C.COL_EMPLOYEE).size()
    out = windows.groupby("employee").agg(
        max_negatives_in_30d=("negatives_in_window", "max"),
        first_flagged_window_start=("window_start", "min"),
        qualifying_windows=("window_start", "size"),
    ).reset_index()
    out["total_negative_messages"] = out["employee"].map(neg_tot).astype(int)
    return out.sort_values(["max_negatives_in_30d", "employee"], ascending=[False, True]).reset_index(drop=True)
