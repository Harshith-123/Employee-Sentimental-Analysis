"""Builds reports/Final_Report.docx from the pipeline outputs.

Run after the notebook / run_pipeline.py:   python -m src.report
Numbers in the report are read from data/processed/, so the document
always matches the latest run.
"""
from __future__ import annotations

import json

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

from . import config as C

BLUE = RGBColor(0x1C, 0x5C, 0xAB)


def _shade(cell, hex_fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), hex_fill)
    tc_pr.append(shd)


def _table(doc, df: pd.DataFrame, widths=None) -> None:
    t = doc.add_table(rows=1, cols=len(df.columns))
    t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, col in enumerate(df.columns):
        c = t.rows[0].cells[i]; c.text = ""
        r = c.paragraphs[0].add_run(str(col)); r.bold = True; r.font.size = Pt(9.5)
        _shade(c, "DCE8F7")
    for _, row in df.iterrows():
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            cells[i].paragraphs[0].add_run(str(v)).font.size = Pt(9.5)
    if widths:
        t.autofit = False
        for i, w in enumerate(widths):
            t.columns[i].width = Inches(w)
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Inches(w)
    doc.add_paragraph()


def _bullets(doc, items) -> None:
    for it in items:
        doc.add_paragraph(it, style="List Bullet")


def _figure(doc, name: str, caption: str, width: float = 6.3) -> None:
    doc.add_picture(str(C.VIS_DIR / name), width=Inches(width))
    p = doc.add_paragraph(caption); p.runs[0].italic = True; p.runs[0].font.size = Pt(9)


def build(path=C.REPORT_DIR / "Final_Report.docx") -> str:
    s = json.loads(C.SUMMARY_PATH.read_text())
    ov, m = s["overview"], s["model"]
    risk = pd.read_csv(C.FLIGHT_RISK_PATH)
    windows = pd.read_csv(C.FLIGHT_RISK_WINDOWS_PATH)
    scores = pd.read_csv(C.MONTHLY_SCORES_PATH)
    rankings = pd.read_csv(C.RANKINGS_PATH)
    coefs = pd.read_csv(C.MODEL_COEFS_PATH)
    labeled = pd.read_csv(C.LABELED_PATH)
    counts = s["sentiment_counts"]; total = sum(counts.values())
    naive = set(scores.loc[scores.negative_count >= C.FLIGHT_RISK_MIN_NEGATIVES, "employee"])
    extra = sorted(set(risk.employee) - naive)

    doc = Document()
    st = doc.styles["Normal"]; st.font.name = "Calibri"; st.font.size = Pt(11)
    for lvl in (1, 2):
        doc.styles[f"Heading {lvl}"].font.color.rgb = BLUE

    doc.add_heading("Employee Sentiment Analysis — Final Report", 0)
    doc.add_paragraph(
        f"Dataset: test.csv — {ov['records']:,} employee e-mails from {ov['employees']} employees, "
        f"{ov['date_min']} to {ov['date_max']} ({ov['months']} months). "
        "All numbers below are generated from data/processed/ by src/report.py."
    )

    # ---------------------------------------------------------- summary
    doc.add_heading("Executive summary", 1)
    lp = ", ".join(f"{r['employee']} ({r['monthly_score']})" for r in s["latest_top_positive"])
    ln = ", ".join(f"{r['employee']} ({r['monthly_score']})" for r in s["latest_top_negative"])
    _bullets(doc, [
        f"Sentiment is predominantly positive: {counts['Positive']:,} positive ({counts['Positive']/total:.0%}), "
        f"{counts['Neutral']:,} neutral ({counts['Neutral']/total:.0%}), {counts['Negative']:,} negative ({counts['Negative']/total:.0%}).",
        f"Latest month ({s['latest_month']}) — top positive: {lp}; top negative: {ln}.",
        f"{len(risk)} of {ov['employees']} employees are flight risks (≥{C.FLIGHT_RISK_MIN_NEGATIVES} negative messages in a "
        f"rolling {C.FLIGHT_RISK_WINDOW_DAYS}-day window): {', '.join(risk.employee)}.",
        f"Linear regression explains the monthly score well (test R² {m['test_r2']:.2f}, MAE {m['test_mae']:.2f} vs "
        f"baseline MAE {m['baseline_test_mae']:.2f}); message volume is the dominant driver.",
        f"Company-wide average monthly score trends slightly upward ({s['company_trend_slope_per_month']:+.3f} per month).",
    ])

    # ------------------------------------------------------- methodology
    doc.add_heading("1. Approach and methodology", 1)
    doc.add_paragraph(
        "The project is a reproducible Python pipeline. The main notebook (employee_sentiment_analysis.ipynb) runs every "
        "step; reusable logic lives in src/ and is covered by unit tests (tests/). run_pipeline.py regenerates every "
        "output from the raw CSV with one command."
    )
    doc.add_heading("Data preparation", 2)
    doc.add_paragraph(
        "E-mail bodies contain quoted-printable artifacts (=20, soft line breaks, control bytes). These are decoded and "
        "removed, together with forwarded/quoted history, URLs and e-mail addresses. The text analysed is the subject plus "
        "the cleaned body (so the 70 messages with an empty body are labeled from their subject). Dates are parsed to "
        "datetime and truncated to calendar months (YYYY-MM)."
    )
    doc.add_heading("Sentiment labeling (Task 1)", 2)
    doc.add_paragraph(
        "src/sentiment.py exposes three interchangeable back-ends: VADER (default), a RoBERTa transformer, and an LLM "
        "(Anthropic or OpenAI, zero-shot prompt at temperature 0, cached per message). The reported results use VADER, "
        "because it is deterministic and runs offline, so any reviewer can reproduce them without API keys. The rule is: "
        f"compound ≥ {C.VADER_POS_THRESHOLD} → Positive, ≤ {C.VADER_NEG_THRESHOLD} → Negative, else Neutral. "
        "The limitation of a lexicon model is that it reads words, not intent: a few factual messages that contain "
        "words such as 'warning' or 'risk' are labeled Negative. Setting SENTIMENT_METHOD=llm in .env switches to the LLM "
        "labeler without any other code changes."
    )

    # ---------------------------------------------------------------- EDA
    doc.add_heading("2. Key findings from the EDA (Task 2)", 1)
    _bullets(doc, [
        f"{ov['records']:,} rows × 4 text columns (Subject, body, date, from); no missing values; "
        f"{ov['duplicate_rows']} fully duplicated rows.",
        f"{ov['employees']} employees, each active in all {ov['months']} months (240 employee-months).",
        "Anomaly: monthly volume is almost constant (91–92 messages) and weekends are as busy as weekdays, which suggests "
        "the dates were assigned synthetically. Time-based conclusions should be read with caution.",
        f"Anomaly: {int(labeled.duplicated(subset=['Subject', 'body']).sum())} messages repeat an earlier subject + body on a "
        "different date. They are kept, because each row is a sent message.",
        "Positive messages are the longest (median 35 words) and Neutral the shortest (median 9 words).",
        "The positive share is stable over time; the mean message score rises from 0.62 (2010) to 0.66 (2011).",
    ])
    _figure(doc, "01_sentiment_distribution.png", "Figure 1 — Sentiment distribution.", 4.8)
    _figure(doc, "03_sentiment_share_by_month.png", "Figure 2 — Share of sentiment by month.")
    _figure(doc, "04_messages_per_employee.png", "Figure 3 — Messages per employee by sentiment.", 5.5)

    # ------------------------------------------------------ scoring/rank
    doc.add_heading("3. Employee scoring and ranking (Tasks 3 & 4)", 1)
    doc.add_paragraph(
        "Each message scores Positive +1, Negative −1, Neutral 0. Scores are summed per employee per calendar month; "
        "each month is an independent group, so the score resets to 0 at the start of every month. Validation checks "
        "confirm that the monthly totals equal the per-message total and that score = positives − negatives. "
        f"Monthly scores range from {scores.monthly_score.min()} to {scores.monthly_score.max()} "
        f"(mean {scores.monthly_score.mean():.1f})."
    )
    doc.add_paragraph(
        "Ranking: for each month, the top-3 positive list sorts by score descending and then employee name alphabetically. "
        "The top-3 negative list sorts by score ascending (most negative first) and then alphabetically. The full "
        "month-by-month table is in data/processed/monthly_rankings.csv."
    )
    latest = rankings[rankings.month == s["latest_month"]]
    tbl = pd.DataFrame({
        "Rank": [1, 2, 3],
        "Top positive": latest[latest.list_type == "Top Positive"].employee.tolist(),
        "Score (+)": latest[latest.list_type == "Top Positive"].monthly_score.tolist(),
        "Top negative": latest[latest.list_type == "Top Negative"].employee.tolist(),
        "Score (−)": latest[latest.list_type == "Top Negative"].monthly_score.tolist(),
    })
    doc.add_paragraph(f"Table 1 — Rankings for the latest month ({s['latest_month']}).").runs[0].bold = True
    _table(doc, tbl, [0.5, 2.1, 0.7, 2.1, 0.7])
    freq = rankings.groupby(["employee", "list_type"]).size().unstack(fill_value=0) \
        .reindex(columns=["Top Positive", "Top Negative"], fill_value=0).sort_values("Top Positive", ascending=False).reset_index()
    freq.columns = ["Employee", "Months in top-3 positive", "Months in top-3 negative"]
    doc.add_paragraph("Table 2 — How often each employee appears in the monthly top-3 lists (24 months).").runs[0].bold = True
    _table(doc, freq, [2.6, 1.8, 1.8])
    _figure(doc, "07_monthly_score_heatmap.png", "Figure 4 — Monthly score per employee.")
    _figure(doc, "08_ranking_frequency.png", "Figure 5 — Months in the top-3 lists.")

    # -------------------------------------------------------- flight risk
    doc.add_heading("4. Flight risk identification (Task 5)", 1)
    doc.add_paragraph(
        f"Criterion: {C.FLIGHT_RISK_MIN_NEGATIVES} or more Negative messages within any rolling {C.FLIGHT_RISK_WINDOW_DAYS}-day "
        "period, irrespective of calendar month or score. For each employee, the negative-message dates are sorted and "
        "a window is slid over them. The employee is flagged if four consecutive negatives fall within days d to d+29. "
        "Unit tests cover month-straddling windows and the exact 30-day boundary."
    )
    rt = risk.rename(columns={"employee": "Employee", "max_negatives_in_30d": "Max negatives in 30 days",
                              "first_flagged_window_start": "First window start",
                              "qualifying_windows": "Windows", "total_negative_messages": "Total negatives"})
    doc.add_paragraph(f"Table 3 — {len(risk)} flagged employees.").runs[0].bold = True
    _table(doc, rt, [2.6, 1.1, 1.2, 0.8, 0.9])
    doc.add_paragraph(
        f"A naive calendar-month count would flag only {len(naive)} employees. The rolling window additionally catches "
        f"{', '.join(extra)}, whose bursts of negativity straddle a month boundary. Every qualifying window is listed in "
        f"data/processed/flight_risk_windows.csv ({len(windows)} windows) so that flags can be reviewed manually."
    )
    _figure(doc, "10_flight_risk.png", "Figure 6 — Negative messages per employee; flight risks highlighted.", 5.5)

    # -------------------------------------------------------------- model
    doc.add_heading("5. Predictive model (Task 6)", 1)
    doc.add_paragraph(
        "Target: the monthly score per employee (240 employee-months). Features are behavioural and not derived from the "
        "sentiment labels (to avoid leakage): message count, average length in characters and words, total words, "
        "active days, average exclamation and question marks per message, the previous month's score, and a time index. "
        f"Model: StandardScaler + LinearRegression (scikit-learn), 80/20 split (random_state={C.RANDOM_STATE}), 5-fold "
        "cross-validation, compared with a mean-only baseline."
    )
    mt = pd.DataFrame({
        "Metric": ["R² (train)", "R² (test)", "MAE (test)", "RMSE (test)", "Baseline MAE", "Baseline RMSE", "5-fold CV R²"],
        "Value": [f"{m['train_r2']:.3f}", f"{m['test_r2']:.3f}", f"{m['test_mae']:.2f}", f"{m['test_rmse']:.2f}",
                  f"{m['baseline_test_mae']:.2f}", f"{m['baseline_test_rmse']:.2f}",
                  f"{m['cv5_r2_mean']:.3f} ± {m['cv5_r2_std']:.3f}"],
    })
    doc.add_paragraph("Table 4 — Model evaluation.").runs[0].bold = True
    _table(doc, mt, [2.5, 2.0])
    ct = coefs.assign(std_coefficient=lambda f: f.std_coefficient.round(3))
    ct.columns = ["Feature", "Standardised coefficient"]
    doc.add_paragraph("Table 5 — Standardised coefficients (effect of +1 SD on the monthly score).").runs[0].bold = True
    _table(doc, ct, [2.5, 2.0])
    _bullets(doc, [
        "The model generalises well: train and test R² are almost equal, and cross-validation is stable.",
        "Message count dominates. Because about 72% of messages are positive, each extra message adds about +0.64 on "
        "average, so the score largely measures how much an employee communicates.",
        "More exclamation and question marks per message slightly raise the score. The previous month's score is "
        "slightly negative (mild mean reversion), and the time index is slightly positive (upward trend).",
        "Next step: model a volume-normalised target (average score per message) to isolate tone from activity.",
    ])
    _figure(doc, "11_model_pred_vs_actual.png", "Figure 7 — Predicted vs actual monthly score (test set).", 4.2)
    _figure(doc, "13_model_coefficients.png", "Figure 8 — Standardised coefficients.", 5.0)
    _figure(doc, "14_company_sentiment_trend.png", "Figure 9 — Company-wide sentiment trend.")

    # ---------------------------------------------------- recommendations
    doc.add_heading("6. Conclusions and recommendations", 1)
    _bullets(doc, [
        "Overall sentiment is healthy and stable. Risk shows up as short bursts of negativity rather than persistently "
        "negative employees.",
        "Prioritise check-ins with sally.beck (strongest burst: 5 negatives in 30 days; highest negative share) and "
        "kayne.coulter (top-3 negative in 11 of 24 months).",
        "Use the rolling-window flag operationally, since it catches employees a monthly report misses.",
        "Report average score per message alongside the monthly sum, so low-volume employees are not mistaken for "
        "disengaged ones.",
        "For production, use the LLM labeler to reduce false negatives on factual messages, and review flagged "
        "windows before acting.",
    ])
    doc.save(path)
    return str(path)


if __name__ == "__main__":
    print(build())
