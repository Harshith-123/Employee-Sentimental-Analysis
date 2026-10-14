"""Task 6 - Predictive modeling with linear regression (scikit-learn).

Unit of analysis: one row per (employee, month) - the same grain as the
monthly score from Task 3, which is the regression TARGET.

Features (deliberately *not* derived from the sentiment labels, to avoid
target leakage):
    message_count          messages sent that month
    avg_char_length        mean cleaned body length (characters)
    avg_word_count         mean words per message
    total_words            words written that month
    active_days            distinct days with >= 1 message
    avg_exclamations       mean '!' per message (emotional intensity)
    avg_questions          mean '?' per message (inquiry / uncertainty)
    prev_month_score       employee's score in the previous month (0 if none)
    month_index            months since the start of the data (time trend)

Model: StandardScaler -> LinearRegression, 80/20 train/test split
(random_state=42), plus 5-fold cross-validation and a mean-predictor
baseline. Standardised coefficients make feature importance comparable.

A second, simpler regression fits the company-wide average score per
month against time to quantify the overall sentiment trend (slope).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C

FEATURES = [
    "message_count", "avg_char_length", "avg_word_count", "total_words", "active_days",
    "avg_exclamations", "avg_questions", "prev_month_score", "month_index",
]
TARGET = "monthly_score"


def build_features(df: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["exclamations"] = d[C.COL_BODY].str.count("!")
    d["questions"] = d[C.COL_BODY].str.count(r"\?")
    d["day"] = d[C.COL_DATE].dt.normalize()
    feats = d.groupby([C.COL_EMPLOYEE, "month"]).agg(
        avg_char_length=("char_length", "mean"),
        avg_word_count=("word_count", "mean"),
        total_words=("word_count", "sum"),
        active_days=("day", "nunique"),
        avg_exclamations=("exclamations", "mean"),
        avg_questions=("questions", "mean"),
    ).reset_index().rename(columns={C.COL_EMPLOYEE: "employee"})

    data = scores.merge(feats, on=["employee", "month"], how="left")
    months = sorted(data["month"].unique())
    data["month_index"] = data["month"].map({m: i for i, m in enumerate(months)})

    # previous *calendar* month score (0 if the employee was silent that month)
    prev = data[["employee", "month", TARGET]].copy()
    prev["month"] = (pd.PeriodIndex(prev["month"], freq="M") + 1).astype(str)
    prev = prev.rename(columns={TARGET: "prev_month_score"})
    data = data.merge(prev, on=["employee", "month"], how="left")
    data["prev_month_score"] = data["prev_month_score"].fillna(0)
    return data


def train_and_evaluate(data: pd.DataFrame) -> dict:
    X, y = data[FEATURES], data[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=C.TEST_SIZE, random_state=C.RANDOM_STATE)

    model = make_pipeline(StandardScaler(), LinearRegression()).fit(X_tr, y_tr)
    baseline = DummyRegressor(strategy="mean").fit(X_tr, y_tr)
    pred_tr, pred_te = model.predict(X_tr), model.predict(X_te)
    base_te = baseline.predict(X_te)

    cv = cross_val_score(make_pipeline(StandardScaler(), LinearRegression()), X, y,
                         cv=KFold(5, shuffle=True, random_state=C.RANDOM_STATE), scoring="r2")

    lr = model.named_steps["linearregression"]
    coefs = pd.DataFrame({"feature": FEATURES, "std_coefficient": lr.coef_}) \
        .assign(abs_coef=lambda f: f["std_coefficient"].abs()) \
        .sort_values("abs_coef", ascending=False).drop(columns="abs_coef").reset_index(drop=True)

    metrics = {
        "n_rows": int(len(data)), "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
        "train_r2": float(r2_score(y_tr, pred_tr)),
        "test_r2": float(r2_score(y_te, pred_te)),
        "test_mae": float(mean_absolute_error(y_te, pred_te)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_te, pred_te))),
        "baseline_test_mae": float(mean_absolute_error(y_te, base_te)),
        "baseline_test_rmse": float(np.sqrt(mean_squared_error(y_te, base_te))),
        "cv5_r2_mean": float(cv.mean()), "cv5_r2_std": float(cv.std()),
        "intercept": float(lr.intercept_),
    }
    return {"model": model, "metrics": metrics, "coefficients": coefs,
            "y_test": y_te.to_numpy(), "y_pred": pred_te}


def company_trend(scores: pd.DataFrame) -> dict:
    """Linear trend of the average employee score per month."""
    m = scores.groupby("month")["monthly_score"].mean().reset_index()
    m["t"] = np.arange(len(m))
    lr = LinearRegression().fit(m[["t"]], m["monthly_score"])
    m["trend"] = lr.predict(m[["t"]])
    return {"table": m, "slope_per_month": float(lr.coef_[0]), "intercept": float(lr.intercept_),
            "r2": float(r2_score(m["monthly_score"], m["trend"]))}


def save_outputs(result: dict, trend: dict) -> None:
    payload = dict(result["metrics"])
    payload["company_trend_slope_per_month"] = trend["slope_per_month"]
    payload["company_trend_r2"] = trend["r2"]
    C.MODEL_METRICS_PATH.write_text(json.dumps(payload, indent=2))
    result["coefficients"].to_csv(C.MODEL_COEFS_PATH, index=False)
