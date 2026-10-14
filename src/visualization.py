"""All charts for the project, saved as PNGs in ./visualization.

Colour policy (kept consistent across every figure):
* Sentiment is a *polarity*, so it uses a diverging pair with a neutral
  grey midpoint: Positive = blue, Neutral = grey, Negative = red.
* Single-series charts use one blue; flight-risk highlights use the
  reserved "critical" red and are always labelled in text as well.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config as C  # noqa: E402

SURFACE = "#fcfcfb"
INK, INK_2, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
BLUE, RED, GREY, CRITICAL = "#2a78d6", "#e34948", "#a8a7a1", "#d03b3b"
SENT_COLORS = {"Positive": BLUE, "Neutral": GREY, "Negative": RED}
ORDER = ["Positive", "Neutral", "Negative"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.titleweight": "bold", "axes.titlesize": 13, "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": GRID, "grid.linewidth": 0.8, "font.size": 10, "legend.frameon": False,
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
})


def _save(fig, name: str) -> str:
    path = C.VIS_DIR / name
    fig.savefig(path)
    plt.close(fig)
    return str(path)


# ----------------------------------------------------------------- EDA
def sentiment_distribution(df: pd.DataFrame) -> str:
    counts = df["sentiment"].value_counts().reindex(ORDER, fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(counts.index, counts.values, color=[SENT_COLORS[s] for s in counts.index], width=0.55)
    total = counts.sum()
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}  ({v / total:.0%})", ha="center", va="bottom", color=INK_2)
    ax.set_title("Sentiment distribution across all messages")
    ax.set_ylabel("Messages")
    return _save(fig, "01_sentiment_distribution.png")


def messages_per_month(df: pd.DataFrame) -> str:
    m = df.groupby("month").size()
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(m.index, m.values, color=BLUE, lw=2, marker="o", ms=4)
    ax.set_title("Message volume per month")
    ax.set_ylabel("Messages")
    ax.tick_params(axis="x", rotation=60)
    return _save(fig, "02_messages_per_month.png")


def sentiment_over_time(df: pd.DataFrame) -> str:
    t = df.groupby(["month", "sentiment"]).size().unstack(fill_value=0).reindex(columns=ORDER, fill_value=0)
    share = t.div(t.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    bottom = np.zeros(len(share))
    for s in ORDER:
        ax.bar(share.index, share[s], bottom=bottom, color=SENT_COLORS[s], label=s,
               width=0.85, edgecolor=SURFACE, linewidth=1)
        bottom += share[s].values
    ax.set_title("Share of sentiment by month")
    ax.set_ylabel("Share of messages")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.tick_params(axis="x", rotation=60)
    ax.legend(ncol=3, loc="lower right", bbox_to_anchor=(1, 1.0))
    return _save(fig, "03_sentiment_share_by_month.png")


def messages_per_employee(df: pd.DataFrame) -> str:
    t = df.groupby([C.COL_EMPLOYEE, "sentiment"]).size().unstack(fill_value=0).reindex(columns=ORDER, fill_value=0)
    t = t.loc[t.sum(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(t) + 1.5))
    left = np.zeros(len(t))
    for s in ORDER:
        ax.barh(t.index, t[s], left=left, color=SENT_COLORS[s], label=s, height=0.7, edgecolor=SURFACE, linewidth=1)
        left += t[s].values
    ax.grid(axis="x"); ax.grid(axis="y", visible=False)
    ax.set_title("Messages per employee by sentiment")
    ax.set_xlabel("Messages")
    ax.legend(ncol=3, loc="lower right")
    return _save(fig, "04_messages_per_employee.png")


def length_by_sentiment(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(7, 4))
    data = [df.loc[df["sentiment"] == s, "word_count"].clip(upper=df["word_count"].quantile(0.99)) for s in ORDER]
    bp = ax.boxplot(data, tick_labels=ORDER, patch_artist=True, widths=0.5, showfliers=False,
                    medianprops={"color": INK})
    for patch, s in zip(bp["boxes"], ORDER):
        patch.set_facecolor(SENT_COLORS[s]); patch.set_alpha(0.85); patch.set_edgecolor(SURFACE)
    ax.set_title("Message length (words) by sentiment")
    ax.set_ylabel("Words per message (outliers hidden)")
    return _save(fig, "05_length_by_sentiment.png")


def weekday_pattern(df: pd.DataFrame) -> str:
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    d = df.assign(wd=df[C.COL_DATE].dt.dayofweek.map(dict(enumerate(days))))
    t = d.groupby(["wd", "sentiment"]).size().unstack(fill_value=0).reindex(index=days, columns=ORDER, fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(days)); w = 0.26
    for i, s in enumerate(ORDER):
        ax.bar(x + (i - 1) * w, t[s], width=w - 0.03, color=SENT_COLORS[s], label=s)
    ax.set_xticks(x, days)
    ax.set_title("Messages by weekday and sentiment")
    ax.set_ylabel("Messages")
    ax.legend(ncol=3, loc="upper right")
    return _save(fig, "06_weekday_sentiment.png")


# ------------------------------------------------------ scores & ranks
def score_heatmap(scores: pd.DataFrame) -> str:
    p = scores.pivot(index="employee", columns="month", values="monthly_score").fillna(0)
    p = p.loc[p.sum(axis=1).sort_values(ascending=False).index]
    lim = max(1, np.abs(p.values).max())
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("div", [RED, "#f0efec", BLUE])
    fig, ax = plt.subplots(figsize=(13, 0.38 * len(p) + 2))
    im = ax.imshow(p.values, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(p.shape[1]), p.columns, rotation=70, fontsize=8)
    ax.set_yticks(range(p.shape[0]), p.index, fontsize=8)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("Monthly score (+1 pos / −1 neg)")
    ax.set_title("Monthly sentiment score per employee")
    return _save(fig, "07_monthly_score_heatmap.png")


def ranking_frequency(rankings: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (label, color) in zip(axes, (("Top Positive", BLUE), ("Top Negative", RED))):
        f = rankings[rankings["list_type"] == label]["employee"].value_counts().sort_values().tail(10)
        ax.barh(f.index, f.values, color=color, height=0.6)
        for i, v in enumerate(f.values):
            ax.text(v, i, f" {v}", va="center", color=INK_2)
        ax.grid(axis="x"); ax.grid(axis="y", visible=False)
        ax.set_title(f"Months appearing in '{label} 3'")
        ax.set_xlabel("Number of months")
    fig.tight_layout()
    return _save(fig, "08_ranking_frequency.png")


def latest_month_ranking(scores: pd.DataFrame, month: str) -> str:
    from .scoring import rank_month

    pos, neg = rank_month(scores, month)
    fig, ax = plt.subplots(figsize=(8, 4))
    lab = list(pos["employee"]) + list(neg["employee"])
    val = list(pos["monthly_score"]) + list(neg["monthly_score"])
    col = [BLUE] * len(pos) + [RED] * len(neg)
    y = np.arange(len(lab))[::-1]
    ax.barh(y, val, color=col, height=0.6)
    ax.set_yticks(y, [f"{'+' if i < len(pos) else '−'} {n}" for i, n in enumerate(lab)])
    for yi, v in zip(y, val):
        ax.text(v, yi, f" {v}", va="center", ha="left" if v >= 0 else "right", color=INK_2)
    ax.axvline(0, color=INK_MUTED, lw=1)
    ax.grid(axis="x"); ax.grid(axis="y", visible=False)
    ax.set_title(f"Top-3 positive (+) and negative (−) employees, {month}")
    ax.set_xlabel("Monthly score")
    return _save(fig, "09_latest_month_ranking.png")


# --------------------------------------------------------- flight risk
def flight_risk_chart(df: pd.DataFrame, risk: pd.DataFrame) -> str:
    neg = df[df["sentiment"] == "Negative"].groupby(C.COL_EMPLOYEE).size().sort_values()
    flagged = set(risk["employee"]) if not risk.empty else set()
    fig, ax = plt.subplots(figsize=(9, 0.4 * len(neg) + 1.5))
    colors = [CRITICAL if e in flagged else GREY for e in neg.index]
    ax.barh(neg.index, neg.values, color=colors, height=0.65)
    for i, (e, v) in enumerate(neg.items()):
        ax.text(v, i, f" {v}" + ("  ⚠ flight risk" if e in flagged else ""), va="center",
                color=CRITICAL if e in flagged else INK_2, fontsize=9)
    ax.grid(axis="x"); ax.grid(axis="y", visible=False)
    ax.set_title(f"Negative messages per employee — {len(flagged)} flagged as flight risk")
    ax.set_xlabel("Total negative messages (flag = ≥4 within any rolling 30 days)")
    return _save(fig, "10_flight_risk.png")


# --------------------------------------------------------------- model
def model_pred_vs_actual(y_true, y_pred, metrics: dict) -> str:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=36, color=BLUE, alpha=0.75, edgecolor=SURFACE, linewidth=0.8)
    lo, hi = min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))
    ax.plot([lo, hi], [lo, hi], color=INK_MUTED, lw=1.2, ls="--", label="Perfect prediction")
    ax.set_xlabel("Actual monthly score"); ax.set_ylabel("Predicted monthly score")
    ax.set_title("Linear regression — test set")
    ax.grid(axis="both")
    ax.text(0.02, 0.97, f"R² = {metrics['test_r2']:.3f}\nMAE = {metrics['test_mae']:.2f}\nRMSE = {metrics['test_rmse']:.2f}",
            transform=ax.transAxes, va="top", color=INK_2)
    ax.legend(loc="lower right")
    return _save(fig, "11_model_pred_vs_actual.png")


def model_residuals(y_true, y_pred) -> str:
    res = np.asarray(y_true) - np.asarray(y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(y_pred, res, s=30, color=BLUE, alpha=0.75, edgecolor=SURFACE, linewidth=0.8)
    axes[0].axhline(0, color=INK_MUTED, lw=1)
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Residual (actual − predicted)")
    axes[0].set_title("Residuals vs predicted"); axes[0].grid(axis="both")
    axes[1].hist(res, bins=20, color=BLUE, edgecolor=SURFACE)
    axes[1].set_title("Residual distribution"); axes[1].set_xlabel("Residual")
    fig.tight_layout()
    return _save(fig, "12_model_residuals.png")


def model_coefficients(coefs: pd.DataFrame) -> str:
    c = coefs.sort_values("std_coefficient")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(c["feature"], c["std_coefficient"], color=[BLUE if v >= 0 else RED for v in c["std_coefficient"]], height=0.6)
    ax.axvline(0, color=INK_MUTED, lw=1)
    ax.grid(axis="x"); ax.grid(axis="y", visible=False)
    ax.set_title("Standardised regression coefficients")
    ax.set_xlabel("Effect on monthly score per 1 SD of feature")
    return _save(fig, "13_model_coefficients.png")


def company_trend_chart(trend: dict) -> str:
    t = trend["table"]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t["month"], t["monthly_score"], color=BLUE, lw=2, marker="o", ms=4, label="Avg employee score")
    ax.plot(t["month"], t["trend"], color=INK_MUTED, lw=1.5, ls="--",
            label=f"Linear trend (slope {trend['slope_per_month']:+.3f}/month)")
    ax.set_title("Average monthly sentiment score — company-wide trend")
    ax.set_ylabel("Mean monthly score")
    ax.tick_params(axis="x", rotation=60)
    ax.legend(loc="upper left")
    return _save(fig, "14_company_sentiment_trend.png")
