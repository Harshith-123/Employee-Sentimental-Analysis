"""Central configuration: paths, constants and tunable thresholds.

Every other module imports from here so the whole pipeline can be
re-pointed (e.g. a different dataset) by editing one file or by setting
environment variables (see .env.example).
"""
from __future__ import annotations

import os
from pathlib import Path

try:  # optional: load a local .env if python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

# ---------------------------------------------------------------- paths
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = Path(os.getenv("RAW_DATA_PATH", DATA_DIR / "raw" / "test.csv"))
PROCESSED_DIR = DATA_DIR / "processed"
VIS_DIR = ROOT_DIR / "visualization"
REPORT_DIR = ROOT_DIR / "reports"

for _d in (PROCESSED_DIR, VIS_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

LABELED_PATH = PROCESSED_DIR / "labeled_messages.csv"
MONTHLY_SCORES_PATH = PROCESSED_DIR / "monthly_scores.csv"
RANKINGS_PATH = PROCESSED_DIR / "monthly_rankings.csv"
FLIGHT_RISK_PATH = PROCESSED_DIR / "flight_risk_employees.csv"
FLIGHT_RISK_WINDOWS_PATH = PROCESSED_DIR / "flight_risk_windows.csv"
MODEL_METRICS_PATH = PROCESSED_DIR / "model_metrics.json"
MODEL_COEFS_PATH = PROCESSED_DIR / "model_coefficients.csv"
SUMMARY_PATH = PROCESSED_DIR / "summary.json"

# ------------------------------------------------------------- columns
COL_SUBJECT = "Subject"
COL_BODY = "body"
COL_DATE = "date"
COL_EMPLOYEE = "from"

# ---------------------------------------------------- sentiment config
# Labeling backend: "vader" (default, offline, deterministic),
# "transformer" (Hugging Face model, needs `transformers` + `torch`),
# or "llm" (OpenAI-compatible / Anthropic chat model, needs an API key).
SENTIMENT_METHOD = os.getenv("SENTIMENT_METHOD", "vader").lower()

# VADER compound-score thresholds (standard values from Hutto & Gilbert 2014)
VADER_POS_THRESHOLD = float(os.getenv("VADER_POS_THRESHOLD", 0.05))
VADER_NEG_THRESHOLD = float(os.getenv("VADER_NEG_THRESHOLD", -0.05))

TRANSFORMER_MODEL = os.getenv(
    "TRANSFORMER_MODEL", "cardiffnlp/twitter-roberta-base-sentiment-latest"
)

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic").lower()  # anthropic | openai
LLM_MODEL = os.getenv("LLM_MODEL", "claude-haiku-5-5")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_WORKERS = int(os.getenv("LLM_WORKERS", 8))   # parallel API requests (lower it if you hit rate limits)

LABELS = ("Positive", "Neutral", "Negative")
SCORE_MAP = {"Positive": 1, "Negative": -1, "Neutral": 0}

# --------------------------------------------------- business rules
FLIGHT_RISK_MIN_NEGATIVES = 4      # >= 4 negative messages ...
FLIGHT_RISK_WINDOW_DAYS = 30       # ... within any rolling 30-day span
TOP_N = 3                          # top-3 positive / negative per month

RANDOM_STATE = 42
TEST_SIZE = 0.2
