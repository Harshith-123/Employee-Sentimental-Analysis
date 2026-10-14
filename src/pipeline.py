"""End-to-end pipeline: raw CSV -> labeled data -> scores -> rankings ->
flight risk -> regression -> charts -> summary.json.

The notebook calls the same functions step by step; this module lets
everything be reproduced with one command:  python run_pipeline.py
"""
from __future__ import annotations

import json

from . import config as C
from . import eda, flight_risk, modeling, preprocessing, scoring, sentiment
from . import visualization as viz


def run(method: str | None = None, verbose: bool = True) -> dict:
    log = print if verbose else (lambda *a, **k: None)

    raw = preprocessing.load_raw()
    df = preprocessing.prepare(raw)
    log(f"[1/6] loaded {len(raw):,} messages from {df[C.COL_EMPLOYEE].nunique()} employees")

    df = sentiment.label_messages(df, method=method)
    df = scoring.add_message_scores(df)
    df.drop(columns=["clean_body"]).to_csv(C.LABELED_PATH, index=False)
    log(f"[2/6] sentiment labeled with '{df['sentiment_method'].iat[0]}': "
        + ", ".join(f"{k}={v}" for k, v in df["sentiment"].value_counts().items()))

    scores = scoring.monthly_scores(df)
    scores.to_csv(C.MONTHLY_SCORES_PATH, index=False)
    rankings = scoring.monthly_rankings(scores)
    rankings.to_csv(C.RANKINGS_PATH, index=False)
    log(f"[3/6] monthly scores for {len(scores)} employee-months; rankings for {scores['month'].nunique()} months")

    windows = flight_risk.find_risk_windows(df)
    risk = flight_risk.flight_risk_employees(df, windows)
    windows.to_csv(C.FLIGHT_RISK_WINDOWS_PATH, index=False)
    risk.to_csv(C.FLIGHT_RISK_PATH, index=False)
    log(f"[4/6] flight risks: {len(risk)} employee(s)")

    data = modeling.build_features(df, scores)
    result = modeling.train_and_evaluate(data)
    trend = modeling.company_trend(scores)
    modeling.save_outputs(result, trend)
    m = result["metrics"]
    log(f"[5/6] regression test R2={m['test_r2']:.3f} MAE={m['test_mae']:.2f} RMSE={m['test_rmse']:.2f}")

    latest = sorted(scores["month"].unique())[-1]
    charts = [
        viz.sentiment_distribution(df), viz.messages_per_month(df), viz.sentiment_over_time(df),
        viz.messages_per_employee(df), viz.length_by_sentiment(df), viz.weekday_pattern(df),
        viz.score_heatmap(scores), viz.ranking_frequency(rankings), viz.latest_month_ranking(scores, latest),
        viz.flight_risk_chart(df, risk),
        viz.model_pred_vs_actual(result["y_test"], result["y_pred"], m),
        viz.model_residuals(result["y_test"], result["y_pred"]),
        viz.model_coefficients(result["coefficients"]), viz.company_trend_chart(trend),
    ]
    log(f"[6/6] {len(charts)} charts written to {C.VIS_DIR}")

    summary = {
        "overview": eda.overview(df, raw),
        "sentiment_counts": eda.sentiment_counts(df)["count"].to_dict(),
        "latest_month": latest,
        "latest_top_positive": scoring.rank_month(scores, latest)[0][["employee", "monthly_score"]].to_dict("records"),
        "latest_top_negative": scoring.rank_month(scores, latest)[1][["employee", "monthly_score"]].to_dict("records"),
        "overall_ranking": scoring.overall_ranking(scores).to_dict("records"),
        "flight_risk": risk.astype(str).to_dict("records"),
        "model": m,
        "company_trend_slope_per_month": trend["slope_per_month"],
        "sentiment_method": sentiment.describe_method(method),
    }
    C.SUMMARY_PATH.write_text(json.dumps(summary, indent=2, default=str))
    return {"raw": raw, "df": df, "scores": scores, "rankings": rankings, "windows": windows,
            "risk": risk, "model_data": data, "model": result, "trend": trend, "summary": summary}
