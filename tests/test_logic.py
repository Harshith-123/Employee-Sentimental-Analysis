"""Unit tests for the business rules (scoring, ranking, flight risk, cleaning).

Run with:  pytest -q
"""
import pandas as pd

from src import config as C
from src.flight_risk import find_risk_windows, flight_risk_employees
from src.preprocessing import clean_text
from src.scoring import add_message_scores, monthly_scores, rank_month


def _frame(rows):
    df = pd.DataFrame(rows, columns=[C.COL_EMPLOYEE, C.COL_DATE, "sentiment"])
    df[C.COL_DATE] = pd.to_datetime(df[C.COL_DATE])
    df["month"] = df[C.COL_DATE].dt.to_period("M").astype(str)
    return df


def test_clean_text_decodes_quoted_printable():
    assert clean_text("hello=20\nworld=\nwide") == "hello worldwide"
    assert "http" not in clean_text("see http://x.com/a?b=1 now")


def test_monthly_score_and_reset():
    df = add_message_scores(_frame([
        ("a", "2011-01-05", "Positive"), ("a", "2011-01-20", "Positive"), ("a", "2011-01-21", "Negative"),
        ("a", "2011-02-01", "Neutral"),
    ]))
    s = monthly_scores(df).set_index("month")["monthly_score"]
    assert s["2011-01"] == 1           # +1 +1 -1
    assert s["2011-02"] == 0           # resets: January does not carry over


def test_ranking_ties_are_alphabetical():
    scores = pd.DataFrame({"employee": ["c", "b", "a", "d"], "month": ["2011-01"] * 4,
                           "monthly_score": [5, 5, 5, -2]})
    pos, neg = rank_month(scores, "2011-01")
    assert list(pos["employee"]) == ["a", "b", "c"]
    assert list(neg["employee"]) == ["d", "a", "b"]


def test_flight_risk_rolling_window_crosses_months():
    # 4 negatives from Jan 20 to Feb 15 (27 days) -> flagged even though split across months
    df = _frame([("x", d, "Negative") for d in ["2011-01-20", "2011-01-28", "2011-02-05", "2011-02-15"]])
    assert set(flight_risk_employees(df)["employee"]) == {"x"}


def test_flight_risk_window_boundary():
    # 30-day window = day d .. d+29. 4th negative on day 30 -> NOT flagged
    df = _frame([("y", d, "Negative") for d in ["2011-01-01", "2011-01-10", "2011-01-20", "2011-01-31"]])
    assert find_risk_windows(df).empty
    # 4th negative on day 29 -> flagged
    df2 = _frame([("y", d, "Negative") for d in ["2011-01-01", "2011-01-10", "2011-01-20", "2011-01-30"]])
    assert not find_risk_windows(df2).empty


def test_three_negatives_not_flagged():
    df = _frame([("z", d, "Negative") for d in ["2011-01-01", "2011-01-02", "2011-01-03"]]
                + [("z", "2011-01-04", "Positive")])
    assert flight_risk_employees(df).empty
