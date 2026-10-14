# Employee Sentiment Analysis

NLP and statistical analysis of an unlabeled dataset of employee e-mails (`test.csv`). The project labels sentiment, explores the data, scores and ranks employees monthly, flags flight risks, and models sentiment with linear regression.

**Main file:** [`employee_sentiment_analysis.ipynb`](employee_sentiment_analysis.ipynb). Every step runs there, with a title, commentary and observations for each section. The `src/` modules hold the supporting code.

---

## Summary of results

<!-- RESULTS:START -->
> Dataset: 2,191 messages · 10 employees · Jan 2010 – Dec 2011 (24 months). Labeling: VADER (lexicon + rule-based NLP model).

**Sentiment:** 1,573 Positive (71.8%) · 445 Neutral (20.3%) · 173 Negative (7.9%)

### Top three positive and negative employees (latest month, 2011-12)

| Rank | Top positive | Score | Top negative | Score |
|---|---|---|---|---|
| 1 | eric.bass@enron.com | 12 | johnny.palmer@enron.com | 1 |
| 2 | patti.thompson@enron.com | 12 | bobette.riner@ipgdirect.com | 3 |
| 3 | lydia.delgado@enron.com | 10 | john.arnold@enron.com | 4 |

Ties are broken alphabetically. Rankings for all months are in `data/processed/monthly_rankings.csv` and in the notebook. Across all 24 months, **john.arnold@enron.com** made the monthly top-3 positive most often (13 months), and **don.baughman@enron.com and kayne.coulter@enron.com** the top-3 negative most often (11 months).

### Employees flagged as flight risk (≥ 4 negative messages within any rolling 30 days)

| Employee | Max negatives in 30 days | First flagged window |
|---|---|---|
| sally.beck@enron.com | 5 | 2011-08-06 → 2011-09-04 |
| bobette.riner@ipgdirect.com | 4 | 2011-03-26 → 2011-04-24 |
| eric.bass@enron.com | 4 | 2011-04-22 → 2011-05-21 |
| john.arnold@enron.com | 4 | 2010-05-25 → 2010-06-23 |
| johnny.palmer@enron.com | 4 | 2010-02-09 → 2010-03-10 |
| kayne.coulter@enron.com | 4 | 2010-04-06 → 2010-05-05 |
| patti.thompson@enron.com | 4 | 2011-07-02 → 2011-07-31 |

A calendar-month count would catch only 4 of these 7. The rolling window also finds bobette.riner@ipgdirect.com, john.arnold@enron.com and johnny.palmer@enron.com, whose bursts of negativity cross a month boundary.

### Predictive model

Linear regression of the monthly score achieves **test R² = 0.82**, MAE = 1.37 and RMSE = 1.79. A mean-only baseline scores MAE = 3.57 and RMSE = 4.25. 5-fold cross-validation gives R² = 0.80 ± 0.03. The company-wide average score trends +0.045 points per month.

### Key insights and recommendations

1. Overall sentiment is healthy (71.8% positive). The risk signal comes from short **bursts** of negativity rather than persistently negative employees.
2. Prioritise check-ins with **sally.beck@enron.com** (strongest burst: 5 negatives in 30 days from 2011-08-06) and **don.baughman@enron.com and kayne.coulter@enron.com** (top-3 negative in 11 of 24 months).
3. Use the **rolling-window** flag operationally. It catches 3 employee(s) a monthly report would miss.
4. Monthly scores are volume-driven. Report the average score per message alongside the sum, so quiet employees are not mislabelled as disengaged.
5. Review each flagged window's messages before acting: automated sentiment is a screening tool, not a verdict.
<!-- RESULTS:END -->

---

## Project structure

```
employee-sentiment-analysis/
├── employee_sentiment_analysis.ipynb   # MAIN deliverable: all tasks, commentary, outputs
├── run_pipeline.py                     # one-command reproduction of every output
├── src/
│   ├── config.py          # paths, thresholds, business rules (env-overridable)
│   ├── preprocessing.py   # load + clean (quoted-printable decoding, quoted history removal)
│   ├── sentiment.py       # Task 1: VADER / transformer / LLM labeling back-ends
│   ├── eda.py             # Task 2: EDA tables
│   ├── visualization.py   # all charts -> visualization/
│   ├── scoring.py         # Task 3 monthly scores + Task 4 rankings
│   ├── flight_risk.py     # Task 5 rolling 30-day flight-risk detection
│   ├── modeling.py        # Task 6 linear regression + company trend
│   ├── pipeline.py        # orchestrates all steps (used by run_pipeline.py)
│   ├── findings.py        # data-driven observations, README summary and report text
│   └── report.py          # builds reports/Final_Report.docx from outputs
├── tests/test_logic.py    # unit tests for scoring, ranking, flight-risk boundaries, cleaning
├── data/
│   ├── raw/test.csv       # input dataset
│   ├── llm_cache.csv      # cached LLM labels (present after an LLM run)
│   └── processed/         # generated outputs (labeled data, scores, rankings, risks, metrics)
├── visualization/         # 14 PNG charts (EDA, rankings, flight risk, model performance)
├── reports/Final_Report.docx
├── requirements.txt
└── .env.example
```

---

## Setup

Requires **Python 3.10+**.

```bash
# 1. Clone and enter the repo
git clone <your-repo-url>.git
cd employee-sentiment-analysis

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) configure environment variables
cp .env.example .env               # Windows: copy .env.example .env
```

No API keys are needed for the default run. `.env` is only required to change the data path or thresholds, or to use the transformer or LLM labelers.

The dataset ships in `data/raw/test.csv`. To use a different copy, place it there or set `RAW_DATA_PATH` in `.env`.

## Usage

**Option A: notebook (recommended for review)**

```bash
jupyter notebook employee_sentiment_analysis.ipynb
# then: Kernel -> Restart & Run All
```

**Option B: command line (reproduces every output from raw data)**

```bash
python run_pipeline.py                  # uses SENTIMENT_METHOD from .env (default: vader)
pytest -q                               # run the unit tests
```

`run_pipeline.py` writes the outputs to `data/processed/` (CSV/JSON) and `visualization/` (PNG). It also regenerates the **results section of this README** and **`reports/Final_Report.docx`**, so the written findings always match the data. The notebook's observations are computed from the data in the same way.

**Choosing the labeling back-end**

The method is read from `SENTIMENT_METHOD` in `.env`, which both the notebook and `run_pipeline.py` use. You can also override it on the command line with `--method`.

```bash
# LLM (Anthropic or OpenAI): set SENTIMENT_METHOD=llm, LLM_PROVIDER, LLM_MODEL and the API key in .env
pip install anthropic                   # or: pip install openai
python run_pipeline.py --method llm

# Hugging Face transformer
pip install transformers torch
python run_pipeline.py --method transformer
```

LLM labels are cached in `data/llm_cache.csv`, one row per unique message. This has three effects:
- Reruns make no API calls.
- An interrupted run resumes where it stopped.
- **Anyone with the cache reproduces the exact labels without an API key.** Set `SENTIMENT_METHOD=llm`, leave the key blank, and run.

---

## Methodology

### 1. Data preparation

- **Text cleaning.** E-mail bodies contain quoted-printable artifacts: `=20` for a space, `=`+newline for a soft break, and stray control bytes. These are decoded. Forwarded and quoted reply history, URLs and e-mail addresses are removed, and whitespace is normalised.
- **Analysed text.** The model reads **subject + cleaned body**. This way the 70 messages with an empty body are still labeled from their subject.
- **Dates.** Dates are parsed to datetime. Months are the date truncated to `YYYY-MM`.

### 2. Sentiment labeling (Task 1)

`src/sentiment.py` has three interchangeable back-ends behind one function, `label_messages()`:

| Back-end | Description | Reproducibility |
|---|---|---|
| `llm` | Claude or GPT, zero-shot prompt, temperature 0 | Cached per message in `data/llm_cache.csv` |
| `vader` *(code default)* | Lexicon + rule model built for informal text. Handles negation, intensifiers, punctuation and capitalisation. | Deterministic, offline |
| `transformer` | `cardiffnlp/twitter-roberta-base-sentiment-latest` | Deterministic |

The method used for the reported results is stated at the top of the results summary above.

**Labeling rule (LLM).** A fixed prompt (`LLM_PROMPT` in `src/sentiment.py`) asks the model for the overall sentiment the *sender* expresses, as exactly one word. It defines the three classes:
- **Positive:** appreciation, enthusiasm, satisfaction, good news, friendly tone.
- **Negative:** frustration, complaints, criticism, worry, bad news, conflict.
- **Neutral:** purely informational, logistical or factual content.

The reply is parsed to a single label; anything unparseable becomes Neutral. Temperature 0 keeps the output as stable as the API allows, and the cache makes the labels exactly reproducible. An LLM judges intent rather than individual words, so factual messages that merely contain words like "risk" or "warning" are not marked Negative.

**Labeling rule (VADER).** compound ≥ 0.05 → **Positive**; compound ≤ −0.05 → **Negative**; otherwise **Neutral**. These are the standard thresholds from Hutto & Gilbert (2014) and are configurable in `.env`. VADER runs offline with no API key, so it is the fallback when no key or cache is available.

### 3. EDA (Task 2)

- **Structure:** record count, data types, missing values, duplicates.
- **Label distribution** across the dataset.
- **Trends:** monthly volume and sentiment share over time.
- **Per-employee activity and sentiment mix.**
- **Message length by sentiment** and weekday patterns.
- **Anomaly checks:** flat monthly volume, weekend activity, repeated messages.

### 4. Monthly employee score (Task 3)

Each message is scored Positive = **+1**, Negative = **−1**, Neutral = **0**. Scores are **summed per (employee, calendar month)**. Each month is an independent group, so the score resets to 0 at the start of every month.

Validation asserts check that no message is lost or double-counted, and that score = positives − negatives in every month.

### 5. Ranking (Task 4)

For each month, the top-3 positive list sorts by score **descending**, then by employee name **alphabetically**. The top-3 negative list sorts by score **ascending** (most negative first), then alphabetically.

Outputs include a month-by-month table, the latest-month chart, and how often each employee appears in either list.

### 6. Flight risk (Task 5)

An employee is flagged if they sent **≥ 4 Negative messages within any rolling 30-day window**, regardless of calendar month or score.

For each employee, the sorted negative-message dates are scanned with a sliding window (two pointers, O(n)). The employee is flagged if 4 negatives fall inside days *d* … *d+29*.

Unit tests cover windows that straddle months and the exact 30-day boundary. Every qualifying window is exported to `data/processed/flight_risk_windows.csv` for manual review.

### 7. Linear regression (Task 6)

- **Target:** monthly score per employee (240 employee-months).
- **Features:** behavioural and not derived from the labels, to avoid target leakage:
  - message count
  - average characters and words per message
  - total words
  - active days
  - average `!` and `?` per message
  - previous month's score
  - month index (time trend)
- **Model:** `StandardScaler → LinearRegression` (scikit-learn), with an 80/20 train/test split (`random_state=42`).
- **Evaluation:**
  - R², MAE and RMSE on the test set.
  - 5-fold cross-validation.
  - Comparison against a mean-only baseline.
  - Standardised coefficients for interpretation.
- **Company trend:** a second regression fits the company-wide average monthly score against time to measure the overall trend.

### Assumptions

- The `from` address identifies the employee.
- Each row is one sent message, so repeated content on different dates counts again.
- Months with no messages for an employee have no score row. With this dataset that never happens: every employee is active in all 24 months.
