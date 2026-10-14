"""Data-driven observations and summary text.

Every number, name and insight in the notebook observations, the README
summary and the Final Report is generated here from the actual results.
So if the labeling method changes (e.g. VADER -> LLM), all of the written
findings update automatically and never contradict the data.
"""
from __future__ import annotations

import pandas as pd

from . import config as C

METHOD_NAMES = {
    "vader": "VADER (lexicon + rule-based NLP model)",
    "transformer": f"a RoBERTa transformer ({C.TRANSFORMER_MODEL})",
    "llm": f"a large language model ({C.LLM_PROVIDER}: {C.LLM_MODEL}, zero-shot, temperature 0)",
}


def _pct(x: float) -> str:
    return f"{x:.1%}"


def _names(items) -> str:
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def method_of(df: pd.DataFrame) -> str:
    return str(df["sentiment_method"].iat[0]) if "sentiment_method" in df else C.SENTIMENT_METHOD


# ------------------------------------------------------------- Task 1
def labeling_md(df: pd.DataFrame) -> str:
    n = len(df)
    c = df["sentiment"].value_counts().reindex(list(C.LABELS), fill_value=0)
    method = method_of(df)
    empty = int((df["clean_body"].str.len() == 0).sum()) if "clean_body" in df else 0
    lines = [
        f"* Labeling method: **{METHOD_NAMES.get(method, method)}**.",
        f"* {n:,} messages labeled: **{c['Positive']:,} Positive ({_pct(c['Positive'] / n)})**, "
        f"**{c['Neutral']:,} Neutral ({_pct(c['Neutral'] / n)})**, **{c['Negative']:,} Negative ({_pct(c['Negative'] / n)})**.",
    ]
    dominant = c.idxmax()
    lines.append(f"* The dominant class is **{dominant}**. The corpus is mostly routine business correspondence, "
                 "so polite and informational messages far outnumber complaints.")
    if method == "vader":
        lines.append("* Known limitation of a lexicon model: it reads words, not intent. Some purely factual messages that "
                     "contain words such as *risk*, *problem* or *warning* (e.g. automatic mailbox warnings) are labeled "
                     "Negative. Setting `SENTIMENT_METHOD=llm` switches to an LLM labeler with no other code changes.")
    elif method == "llm":
        lines.append("* The LLM judges the sender's overall intent rather than individual words, so factual messages that "
                     "merely contain words like *risk* or *warning* stay Neutral. Every label is cached in "
                     "`data/llm_cache.csv`, so the exact labels are reproducible without an API key.")
    if empty:
        lines.append(f"* {empty} messages have an empty body. They are labeled from their subject line, which is why the "
                     "subject is part of the analysed text.")
    return "\n".join(lines)


# ------------------------------------------------------------- Task 2
def eda_facts(df: pd.DataFrame, raw: pd.DataFrame) -> dict:
    per_month = df.groupby("month").size()
    wd = df[C.COL_DATE].dt.dayofweek
    per_emp = df.groupby(C.COL_EMPLOYEE)["sentiment"].value_counts(normalize=True).unstack(fill_value=0)
    per_emp = per_emp.reindex(columns=list(C.LABELS), fill_value=0)
    counts = df[C.COL_EMPLOYEE].value_counts()
    year = df.assign(y=df[C.COL_DATE].dt.year).groupby("y")["message_score"].mean()
    med = df.groupby("sentiment")["word_count"].median().reindex(list(C.LABELS))
    return {
        "rows": len(raw), "cols": list(raw.columns), "missing": int(raw.isna().sum().sum()),
        "dup_rows": int(raw.duplicated().sum()),
        "repeated": int(raw.duplicated(subset=[C.COL_SUBJECT, C.COL_BODY]).sum()),
        "employees": df[C.COL_EMPLOYEE].nunique(), "months": df["month"].nunique(),
        "date_min": df[C.COL_DATE].min().date(), "date_max": df[C.COL_DATE].max().date(),
        "vol_min": int(per_month.min()), "vol_max": int(per_month.max()),
        "vol_cv": float(per_month.std() / per_month.mean()),
        "weekend_share": float((wd >= 5).mean()),
        "busiest": counts.idxmax(), "busiest_n": int(counts.max()),
        "quietest": counts.idxmin(), "quietest_n": int(counts.min()),
        "neg_share": per_emp["Negative"].sort_values(ascending=False),
        "year_score": year, "median_words": med,
        "all_months_active": bool((df.groupby(C.COL_EMPLOYEE)["month"].nunique() == df["month"].nunique()).all()),
    }


def eda_md(df: pd.DataFrame, raw: pd.DataFrame) -> str:
    f = eda_facts(df, raw)
    ns = f["neg_share"]
    top_neg = ", ".join(f"{e} ({_pct(v)})" for e, v in ns.head(3).items())
    med = f["median_words"]
    longest, shortest = med.idxmax(), med.idxmin()
    lines = [
        f"* **Structure:** {f['rows']:,} rows × {len(f['cols'])} columns ({', '.join(f'`{c}`' for c in f['cols'])}); "
        f"{'**no missing values**' if f['missing'] == 0 else str(f['missing']) + ' missing values'}; "
        f"{f['dup_rows']} fully duplicated rows. `date` is parsed to datetime.",
        f"* **Coverage:** {f['employees']} employees, {f['months']} months ({f['date_min']} to {f['date_max']})"
        + (", and every employee is active in every month." if f["all_months_active"] else "."),
    ]
    anomalies = []
    if f["vol_cv"] < 0.05:
        anomalies.append(f"Monthly volume is almost perfectly flat ({f['vol_min']}–{f['vol_max']} messages per month)")
    if f["weekend_share"] > 0.2:
        anomalies.append(f"{_pct(f['weekend_share'])} of messages fall on weekends, about the same rate as weekdays")
    if anomalies:
        lines.append("* **Anomaly:** " + "; ".join(anomalies) + ". Real e-mail traffic is rarely this uniform, so the "
                     "dates appear to have been **assigned synthetically**. Time-based conclusions should be read with caution.")
    if f["repeated"]:
        lines.append(f"* **Anomaly:** {f['repeated']} messages repeat an earlier subject + body on a different date. They are "
                     "kept because each row is a distinct sent message, but they inflate some employees' counts.")
    ys = f["year_score"]
    if len(ys) > 1:
        lines.append(f"* **Trend:** the mean message score moves from {ys.iloc[0]:.2f} ({ys.index[0]}) to "
                     f"{ys.iloc[-1]:.2f} ({ys.index[-1]}).")
    lines += [
        f"* **Employees:** activity ranges from {f['quietest_n']} ({f['quietest']}) to {f['busiest_n']} messages "
        f"({f['busiest']}). Highest negative share: {top_neg}. Lowest: {ns.index[-1]} ({_pct(ns.iloc[-1])}).",
        f"* **Length:** {longest} messages are the longest (median {med[longest]:.0f} words) and {shortest} the shortest "
        f"(median {med[shortest]:.0f} words).",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------- Task 3
def scoring_md(scores: pd.DataFrame) -> str:
    s = scores["monthly_score"]
    nonpos = scores[s <= 0]
    lines = [
        f"* Monthly scores range from **{s.min()} to {s.max()}** (mean {s.mean():.1f}, median {s.median():.1f}).",
        f"* **{len(nonpos)} of {len(scores)} employee-months are ≤ 0.**",
    ]
    if len(nonpos):
        worst = nonpos["employee"].value_counts()
        lines[-1] += (f" The most frequent is {worst.index[0]} with {worst.iloc[0]} "
                      f"({', '.join(nonpos.loc[nonpos.employee == worst.index[0], 'month'])}).")
    lines.append("* The validation asserts confirm that no message was lost or double-counted in the aggregation, "
                 "and that score = positives − negatives in every month.")
    return "\n".join(lines)


# ------------------------------------------------------------- Task 4
def ranking_facts(scores: pd.DataFrame, rankings: pd.DataFrame) -> dict:
    from .scoring import overall_ranking, rank_month

    latest = sorted(scores["month"].unique())[-1]
    pos, neg = rank_month(scores, latest)
    freq = rankings.groupby(["list_type", "employee"]).size()
    fp, fn = freq.get("Top Positive", pd.Series(dtype=int)), freq.get("Top Negative", pd.Series(dtype=int))
    return {"latest": latest, "pos": pos, "neg": neg,
            "most_pos": sorted(fp[fp == fp.max()].index), "most_pos_n": int(fp.max()),
            "most_neg": sorted(fn[fn == fn.max()].index), "most_neg_n": int(fn.max()),
            "overall": overall_ranking(scores), "months": scores["month"].nunique()}


def ranking_md(scores: pd.DataFrame, rankings: pd.DataFrame) -> str:
    f = ranking_facts(scores, rankings)
    fmt = lambda fr: ", ".join(f"{r.employee} ({r.monthly_score})" for r in fr.itertuples())
    tie = f["pos"]["monthly_score"].duplicated().any() or f["neg"]["monthly_score"].duplicated().any()
    ov = f["overall"]
    return "\n".join([
        f"* **Latest month ({f['latest']}):** top positive: {fmt(f['pos'])}. Top negative: {fmt(f['neg'])}."
        + (" Ties are broken alphabetically." if tie else ""),
        f"* **Consistency across {f['months']} months:** {_names(f['most_pos'])} "
        f"{'is' if len(f['most_pos']) == 1 else 'are'} in the monthly top-3 positive most often ({f['most_pos_n']} months). "
        f"{_names(f['most_neg'])} {'appears' if len(f['most_neg']) == 1 else 'appear'} in the top-3 negative most often "
        f"({f['most_neg_n']} months).",
        f"* Rankings partly reflect volume: the score is a sum, so prolific writers rank higher. The whole-period leader is "
        f"{ov.employee.iat[0]} ({ov.total_score.iat[0]:+d}).",
    ])


# ------------------------------------------------------------- Task 5
def flight_facts(scores: pd.DataFrame, risk: pd.DataFrame) -> dict:
    naive = set(scores.loc[scores["negative_count"] >= C.FLIGHT_RISK_MIN_NEGATIVES, "employee"])
    flagged = list(risk["employee"]) if len(risk) else []
    strongest = risk.sort_values(["max_negatives_in_30d", "total_negative_messages"], ascending=False).head(1)
    return {"flagged": flagged, "naive": naive, "extra": sorted(set(flagged) - naive),
            "strongest": strongest.employee.iat[0] if len(strongest) else None,
            "strongest_n": int(strongest.max_negatives_in_30d.iat[0]) if len(strongest) else 0,
            "strongest_start": str(strongest.first_flagged_window_start.iat[0]) if len(strongest) else ""}


def flight_md(scores: pd.DataFrame, risk: pd.DataFrame, n_employees: int) -> str:
    f = flight_facts(scores, risk)
    if not f["flagged"]:
        return (f"* **No employee is flagged:** nobody sent {C.FLIGHT_RISK_MIN_NEGATIVES} or more negative messages "
                f"within any rolling {C.FLIGHT_RISK_WINDOW_DAYS}-day window.")
    detail = ", ".join(f"{r.employee} ({r.max_negatives_in_30d})" for r in risk.itertuples())
    lines = [f"* **{len(f['flagged'])} of {n_employees} employees are flagged** (max negatives in one 30-day window): {detail}."]
    if f["extra"]:
        lines.append(f"* A calendar-month count would flag only **{len(f['naive'] & set(f['flagged']))}**. The rolling window "
                     f"adds **{_names(f['extra'])}**, whose negative bursts straddle a month boundary. This is why the "
                     "brief specifies a rolling window.")
    else:
        lines.append("* In this data the rolling window and a calendar-month count flag the same employees.")
    lines.append("* The flag is deliberately independent of the score: an employee with a high long-run score can still "
                 "have a short burst of negativity. Every qualifying window is listed in "
                 "`data/processed/flight_risk_windows.csv` so that each flag can be reviewed manually.")
    return "\n".join(lines)


# ------------------------------------------------------------- Task 6
def model_md(metrics: dict, coefs: pd.DataFrame, trend: dict, df: pd.DataFrame) -> str:
    m = metrics
    top = coefs.iloc[0]
    avg_msg = df["message_score"].mean()
    c = coefs.set_index("feature")["std_coefficient"]
    overfit = abs(m["train_r2"] - m["test_r2"]) < 0.05
    lines = [
        f"* **Performance:** test R² = **{m['test_r2']:.2f}**, MAE = **{m['test_mae']:.2f}**, RMSE = **{m['test_rmse']:.2f}**, "
        f"against a mean-only baseline with MAE {m['baseline_test_mae']:.2f} / RMSE {m['baseline_test_rmse']:.2f}. "
        f"5-fold cross-validation R² = {m['cv5_r2_mean']:.2f} ± {m['cv5_r2_std']:.2f}."
        + (" Train R² ≈ test R², so there is no sign of overfitting." if overfit else
           f" Train R² is {m['train_r2']:.2f}, so some overfitting is present."),
        f"* **Strongest driver:** `{top.feature}` (standardised coefficient {top.std_coefficient:+.2f}).",
    ]
    if top.feature == "message_count":
        lines.append(f"  The average message scores {avg_msg:+.2f}, so every extra message moves the monthly score by about "
                     "that much. The monthly score is largely a function of how much an employee communicates.")
    pm = c.get("prev_month_score", 0)
    lines.append(f"* `prev_month_score` is {'slightly negative (mild mean reversion)' if pm < 0 else 'positive (sentiment momentum)'}; "
                 f"the company-wide average score trends {trend['slope_per_month']:+.3f} points per month.")
    lines.append("* **Significance:** the model describes and forecasts the *level* of the score well. Because volume drives "
                 "much of it, a volume-normalised target (average score per message) would better isolate tone, and is the "
                 "natural next iteration.")
    return "\n".join(lines)


# ---------------------------------------------------- summary + README
def recommendations(df, scores, rankings, risk) -> list[str]:
    rf, ff = ranking_facts(scores, rankings), flight_facts(scores, risk)
    pos_share = (df["sentiment"] == "Positive").mean()
    recs = [f"Overall sentiment is {'healthy' if pos_share >= 0.5 else 'mixed'} ({_pct(pos_share)} positive). "
            + ("The risk signal comes from short **bursts** of negativity rather than persistently negative employees."
               if ff["flagged"] else
               f"No employee reached {C.FLIGHT_RISK_MIN_NEGATIVES} negative messages within any rolling "
               f"{C.FLIGHT_RISK_WINDOW_DAYS}-day window, so there is no acute flight-risk signal.")]
    who = []
    if ff["strongest"]:
        who.append(f"**{ff['strongest']}** (strongest burst: {ff['strongest_n']} negatives in 30 days from "
                   f"{ff['strongest_start']})")
    persistent = [e for e in rf["most_neg"] if e != ff["strongest"]]
    if persistent:
        who.append(f"**{_names(persistent)}** (top-3 negative in {rf['most_neg_n']} of {rf['months']} months)")
    if who:
        recs.append("Prioritise check-ins with " + " and ".join(who) + ".")
    if ff["extra"]:
        recs.append(f"Use the **rolling-window** flag operationally. It catches {len(ff['extra'])} employee(s) a monthly "
                    "report would miss.")
    recs.append("Monthly scores are volume-driven. Report the average score per message alongside the sum, so quiet "
                "employees are not mislabelled as disengaged.")
    recs.append("Review each flagged window's messages before acting: automated sentiment is a screening tool, not a verdict."
                if ff["flagged"] else
                "Keep running the rolling-window check each month: automated sentiment is a screening tool, so any future "
                "flag should be confirmed by reading the messages.")
    return recs


def readme_summary_md(df, raw, scores, rankings, risk, windows, metrics, trend) -> str:
    rf = ranking_facts(scores, rankings)
    c = df["sentiment"].value_counts().reindex(list(C.LABELS), fill_value=0)
    n = len(df)
    method = method_of(df)
    out = [
        f"> Dataset: {n:,} messages · {df[C.COL_EMPLOYEE].nunique()} employees · "
        f"{df[C.COL_DATE].min():%b %Y} – {df[C.COL_DATE].max():%b %Y} ({df['month'].nunique()} months). "
        f"Labeling: {METHOD_NAMES.get(method, method)}.",
        "",
        f"**Sentiment:** {c['Positive']:,} Positive ({_pct(c['Positive'] / n)}) · {c['Neutral']:,} Neutral "
        f"({_pct(c['Neutral'] / n)}) · {c['Negative']:,} Negative ({_pct(c['Negative'] / n)})",
        "",
        f"### Top three positive and negative employees (latest month, {rf['latest']})",
        "",
        "| Rank | Top positive | Score | Top negative | Score |",
        "|---|---|---|---|---|",
    ]
    for i in range(3):
        p = rf["pos"].iloc[i] if i < len(rf["pos"]) else None
        q = rf["neg"].iloc[i] if i < len(rf["neg"]) else None
        out.append(f"| {i + 1} | {p.employee if p is not None else ''} | {p.monthly_score if p is not None else ''} | "
                   f"{q.employee if q is not None else ''} | {q.monthly_score if q is not None else ''} |")
    out += [
        "",
        "Ties are broken alphabetically. Rankings for all months are in `data/processed/monthly_rankings.csv` and in the "
        f"notebook. Across all {rf['months']} months, **{_names(rf['most_pos'])}** made the monthly top-3 positive most "
        f"often ({rf['most_pos_n']} months), and **{_names(rf['most_neg'])}** the top-3 negative most often "
        f"({rf['most_neg_n']} months).",
        "",
        f"### Employees flagged as flight risk (≥ {C.FLIGHT_RISK_MIN_NEGATIVES} negative messages within any rolling "
        f"{C.FLIGHT_RISK_WINDOW_DAYS} days)",
        "",
    ]
    if len(risk):
        first = windows.sort_values("window_start").groupby("employee").first()
        out += ["| Employee | Max negatives in 30 days | First flagged window |", "|---|---|---|"]
        for r in risk.itertuples():
            w = first.loc[r.employee]
            out.append(f"| {r.employee} | {r.max_negatives_in_30d} | {w.window_start} → {w.window_end} |")
        ff = flight_facts(scores, risk)
        if ff["extra"]:
            out += ["", f"A calendar-month count would catch only {len(ff['naive'] & set(ff['flagged']))} of these "
                        f"{len(ff['flagged'])}. The rolling window also finds {_names(ff['extra'])}, whose bursts of "
                        "negativity cross a month boundary."]
    else:
        out.append("No employee meets the flight-risk criterion.")
    m = metrics
    out += [
        "",
        "### Predictive model",
        "",
        f"Linear regression of the monthly score achieves **test R² = {m['test_r2']:.2f}**, MAE = {m['test_mae']:.2f} and "
        f"RMSE = {m['test_rmse']:.2f}. A mean-only baseline scores MAE = {m['baseline_test_mae']:.2f} and RMSE = "
        f"{m['baseline_test_rmse']:.2f}. 5-fold cross-validation gives R² = {m['cv5_r2_mean']:.2f} ± {m['cv5_r2_std']:.2f}. "
        f"The company-wide average score trends {trend['slope_per_month']:+.3f} points per month.",
        "",
        "### Key insights and recommendations",
        "",
    ]
    out += [f"{i}. {r}" for i, r in enumerate(recommendations(df, scores, rankings, risk), 1)]
    return "\n".join(out)


README_START, README_END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def update_readme(summary_md: str, path=C.ROOT_DIR / "README.md") -> None:
    """Replace the generated results block in README.md (between the markers)."""
    text = path.read_text(encoding="utf-8")
    if README_START not in text or README_END not in text:
        return
    head, rest = text.split(README_START, 1)
    _, tail = rest.split(README_END, 1)
    path.write_text(f"{head}{README_START}\n{summary_md}\n{README_END}{tail}", encoding="utf-8")
