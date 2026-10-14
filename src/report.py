"""Builds reports/Final_Report.docx from the pipeline outputs.

All narrative text comes from src/findings.py, so the report always matches
the latest run, whichever sentiment-labeling method was used.

Called automatically by run_pipeline.py. To rebuild on its own (after a run):
    python -m src.report
"""
from __future__ import annotations

import json
import re

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

from . import config as C
from . import findings as F

BLUE = RGBColor(0x1C, 0x5C, 0xAB)
_INLINE = re.compile(r"(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*)")


# ------------------------------------------------------------- helpers
def _shade(cell, hex_fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), hex_fill)
    tc_pr.append(shd)


def _runs(par, text: str) -> None:
    """Add text to a paragraph, honouring **bold**, *italic* and `code` markdown."""
    for part in _INLINE.split(text):
        if not part:
            continue
        if part.startswith("**"):
            par.add_run(part[2:-2]).bold = True
        elif part.startswith("`"):
            r = par.add_run(part[1:-1]); r.font.name = "Consolas"; r.font.size = Pt(10)
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            par.add_run(part[1:-1]).italic = True
        else:
            par.add_run(part)


def _md(doc, md: str) -> None:
    """Render findings markdown: '* ' -> bullet, indented line -> continuation, 'N. ' -> numbered."""
    last = None
    for line in md.splitlines():
        if not line.strip():
            continue
        if line.startswith("* "):
            last = doc.add_paragraph(style="List Bullet"); _runs(last, line[2:])
        elif re.match(r"^\d+\. ", line):
            last = doc.add_paragraph(style="List Number"); _runs(last, line.split(". ", 1)[1])
        elif line.startswith("  ") and last is not None:
            _runs(last, " " + line.strip())
        else:
            last = doc.add_paragraph(); _runs(last, line)


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


def _caption(doc, text: str) -> None:
    doc.add_paragraph().add_run(text).bold = True


def _figure(doc, name: str, caption: str, width: float = 6.3) -> None:
    path = C.VIS_DIR / name
    if not path.exists():
        return
    doc.add_picture(str(path), width=Inches(width))
    p = doc.add_paragraph(); r = p.add_run(caption); r.italic = True; r.font.size = Pt(9)


def _load():
    """Reload everything from disk (used when run as `python -m src.report`)."""
    from . import modeling, preprocessing

    raw = preprocessing.load_raw()
    df = pd.read_csv(C.LABELED_PATH, parse_dates=[C.COL_DATE])
    df["clean_body"] = preprocessing.prepare(raw)["clean_body"].values
    scores = pd.read_csv(C.MONTHLY_SCORES_PATH)
    trend = modeling.company_trend(scores)
    return dict(df=df, raw=raw, scores=scores, rankings=pd.read_csv(C.RANKINGS_PATH),
                risk=pd.read_csv(C.FLIGHT_RISK_PATH), windows=pd.read_csv(C.FLIGHT_RISK_WINDOWS_PATH),
                coefs=pd.read_csv(C.MODEL_COEFS_PATH), metrics=json.loads(C.MODEL_METRICS_PATH.read_text()),
                trend=trend)


# --------------------------------------------------------------- build
def build(path=C.REPORT_DIR / "Final_Report.docx", **data) -> str:
    d = data or _load()
    df, raw, scores, rankings = d["df"], d["raw"], d["scores"], d["rankings"]
    risk, windows, coefs, m, trend = d["risk"], d["windows"], d["coefs"], d["metrics"], d["trend"]
    method = F.method_of(df)
    ov = F.eda_facts(df, raw)

    doc = Document()
    st = doc.styles["Normal"]; st.font.name = "Calibri"; st.font.size = Pt(11)
    for lvl in (1, 2):
        doc.styles[f"Heading {lvl}"].font.color.rgb = BLUE

    doc.add_heading("Employee Sentiment Analysis — Final Report", 0)
    doc.add_paragraph(
        f"Dataset: test.csv — {ov['rows']:,} employee e-mails from {ov['employees']} employees, {ov['date_min']} to "
        f"{ov['date_max']} ({ov['months']} months). Sentiment labeling: {F.METHOD_NAMES.get(method, method)}. "
        "All numbers and findings in this report are generated from the pipeline outputs (src/report.py)."
    )

    doc.add_heading("Executive summary", 1)
    _md(doc, "\n".join(f"{i}. {r}" for i, r in enumerate(F.recommendations(df, scores, rankings, risk), 1)))

    # ------------------------------------------------------ methodology
    doc.add_heading("1. Approach and methodology", 1)
    doc.add_paragraph(
        "The project is a reproducible Python pipeline. The main notebook (employee_sentiment_analysis.ipynb) runs every "
        "step; reusable logic lives in src/ and is covered by unit tests (tests/). run_pipeline.py regenerates every "
        "output, this report and the README results from the raw CSV with one command."
    )
    doc.add_heading("Data preparation", 2)
    doc.add_paragraph(
        "E-mail bodies contain quoted-printable artifacts (=20, soft line breaks, control bytes). These are decoded and "
        "removed, together with forwarded and quoted reply history, URLs and e-mail addresses. The analysed text is the "
        "subject plus the cleaned body, so messages with an empty body are labeled from their subject. Dates are parsed "
        "to datetime and truncated to calendar months (YYYY-MM)."
    )
    doc.add_heading("Sentiment labeling (Task 1)", 2)
    if method == "llm":
        doc.add_paragraph(
            f"Each message was classified by {F.METHOD_NAMES['llm']} with a fixed prompt that defines the three classes: "
            "Positive (appreciation, enthusiasm, satisfaction, good news), Negative (frustration, complaints, criticism, "
            "worry, bad news, conflict) and Neutral (purely informational or logistical content). The reply is parsed "
            "to one label, and anything unparseable defaults to Neutral. Temperature 0 makes the output as deterministic "
            "as the API allows, and every label is cached in data/llm_cache.csv, so the exact labels can be reproduced "
            "without an API key. VADER and a RoBERTa transformer are available as alternative back-ends (SENTIMENT_METHOD)."
        )
    else:
        doc.add_paragraph(
            "src/sentiment.py exposes three interchangeable back-ends: VADER, a RoBERTa transformer, and an LLM. The "
            f"reported results use {F.METHOD_NAMES.get(method, method)}, which is deterministic and runs offline. Rule: "
            f"compound ≥ {C.VADER_POS_THRESHOLD} → Positive, ≤ {C.VADER_NEG_THRESHOLD} → Negative, else Neutral. "
            "Setting SENTIMENT_METHOD=llm switches to the LLM labeler with no other code changes."
        )
    _md(doc, F.labeling_md(df))

    # -------------------------------------------------------------- EDA
    doc.add_heading("2. Key findings from the EDA (Task 2)", 1)
    _md(doc, F.eda_md(df, raw))
    _figure(doc, "01_sentiment_distribution.png", "Figure 1 — Sentiment distribution.", 4.8)
    _figure(doc, "03_sentiment_share_by_month.png", "Figure 2 — Share of sentiment by month.")
    _figure(doc, "04_messages_per_employee.png", "Figure 3 — Messages per employee by sentiment.", 5.5)

    # ------------------------------------------------- scoring/ranking
    doc.add_heading("3. Employee scoring and ranking (Tasks 3 & 4)", 1)
    doc.add_paragraph(
        "Each message scores Positive +1, Negative −1, Neutral 0. Scores are summed per employee per calendar month. "
        "Each month is an independent group, so the score resets to 0 at the start of every month."
    )
    _md(doc, F.scoring_md(scores))
    doc.add_paragraph(
        "Ranking: for each month, the top-3 positive list sorts by score descending and then employee name "
        "alphabetically. The top-3 negative list sorts by score ascending (most negative first) and then alphabetically. "
        "The full month-by-month table is in data/processed/monthly_rankings.csv."
    )
    rf = F.ranking_facts(scores, rankings)
    pad = lambda s: list(s) + [""] * (3 - len(s))
    tbl = pd.DataFrame({
        "Rank": [1, 2, 3],
        "Top positive": pad(rf["pos"].employee), "Score (+)": pad(rf["pos"].monthly_score),
        "Top negative": pad(rf["neg"].employee), "Score (−)": pad(rf["neg"].monthly_score),
    })
    _caption(doc, f"Table 1 — Rankings for the latest month ({rf['latest']}).")
    _table(doc, tbl, [0.5, 2.1, 0.7, 2.1, 0.7])
    freq = rankings.groupby(["employee", "list_type"]).size().unstack(fill_value=0) \
        .reindex(columns=["Top Positive", "Top Negative"], fill_value=0).sort_values("Top Positive", ascending=False).reset_index()
    freq.columns = ["Employee", "Months in top-3 positive", "Months in top-3 negative"]
    _caption(doc, f"Table 2 — How often each employee appears in the monthly top-3 lists ({rf['months']} months).")
    _table(doc, freq, [2.6, 1.8, 1.8])
    _md(doc, F.ranking_md(scores, rankings))
    _figure(doc, "07_monthly_score_heatmap.png", "Figure 4 — Monthly score per employee.")
    _figure(doc, "08_ranking_frequency.png", "Figure 5 — Months in the top-3 lists.")

    # ------------------------------------------------------ flight risk
    doc.add_heading("4. Flight risk identification (Task 5)", 1)
    doc.add_paragraph(
        f"Criterion: {C.FLIGHT_RISK_MIN_NEGATIVES} or more Negative messages within any rolling {C.FLIGHT_RISK_WINDOW_DAYS}-day "
        "period, irrespective of calendar month or score. For each employee, the negative-message dates are sorted and "
        "a window is slid over them. The employee is flagged if four negatives fall within days d to d+29. Unit tests "
        "cover month-straddling windows and the exact 30-day boundary."
    )
    if len(risk):
        rt = risk.rename(columns={"employee": "Employee", "max_negatives_in_30d": "Max negatives in 30 days",
                                  "first_flagged_window_start": "First window start",
                                  "qualifying_windows": "Windows", "total_negative_messages": "Total negatives"})
        _caption(doc, f"Table 3 — {len(risk)} flagged employees.")
        _table(doc, rt, [2.6, 1.1, 1.2, 0.8, 0.9])
    _md(doc, F.flight_md(scores, risk, ov["employees"]))
    _figure(doc, "10_flight_risk.png", "Figure 6 — Negative messages per employee; flight risks highlighted.", 5.5)

    # ------------------------------------------------------------ model
    doc.add_heading("5. Predictive model (Task 6)", 1)
    doc.add_paragraph(
        "Target: the monthly score per employee-month. Features are behavioural and not derived from the sentiment labels "
        "(to avoid leakage): message count, average length in characters and words, total words, active days, average "
        "exclamation and question marks per message, the previous month's score, and a time index. Model: StandardScaler "
        f"+ LinearRegression (scikit-learn), 80/20 split (random_state={C.RANDOM_STATE}), 5-fold cross-validation, compared "
        "with a mean-only baseline."
    )
    mt = pd.DataFrame({
        "Metric": ["R² (train)", "R² (test)", "MAE (test)", "RMSE (test)", "Baseline MAE", "Baseline RMSE", "5-fold CV R²"],
        "Value": [f"{m['train_r2']:.3f}", f"{m['test_r2']:.3f}", f"{m['test_mae']:.2f}", f"{m['test_rmse']:.2f}",
                  f"{m['baseline_test_mae']:.2f}", f"{m['baseline_test_rmse']:.2f}",
                  f"{m['cv5_r2_mean']:.3f} ± {m['cv5_r2_std']:.3f}"],
    })
    _caption(doc, "Table 4 — Model evaluation.")
    _table(doc, mt, [2.5, 2.0])
    ct = coefs.assign(std_coefficient=lambda f: f.std_coefficient.round(3))
    ct.columns = ["Feature", "Standardised coefficient"]
    _caption(doc, "Table 5 — Standardised coefficients (effect of +1 SD on the monthly score).")
    _table(doc, ct, [2.5, 2.0])
    _md(doc, F.model_md(m, coefs, trend, df))
    _figure(doc, "11_model_pred_vs_actual.png", "Figure 7 — Predicted vs actual monthly score (test set).", 4.2)
    _figure(doc, "13_model_coefficients.png", "Figure 8 — Standardised coefficients.", 5.0)
    _figure(doc, "14_company_sentiment_trend.png", "Figure 9 — Company-wide sentiment trend.")

    doc.add_heading("6. Conclusions and recommendations", 1)
    _md(doc, "\n".join(f"{i}. {r}" for i, r in enumerate(F.recommendations(df, scores, rankings, risk), 1)))
    doc.save(path)
    return str(path)


if __name__ == "__main__":
    print(build())
