# Employee Sentiment Analysis

NLP and statistical analysis of an unlabeled dataset of employee e-mails (`test.csv`). The project labels sentiment, explores the data, scores and ranks employees monthly, flags flight risks, and models sentiment with linear regression.

**Main file:** [`employee_sentiment_analysis.ipynb`](employee_sentiment_analysis.ipynb). Every step runs there, with a title, commentary and observations for each section. The `src/` modules hold the supporting code.

---

## Summary of results

> Dataset: 2,191 messages · 10 employees · Jan 2010 – Dec 2011 (24 months). Labeling: VADER (default, reproducible offline).

**Sentiment:** 1,573 Positive (71.8%) · 445 Neutral (20.3%) · 173 Negative (7.9%)

### Top three positive and negative employees (latest month, 2011-12)

| Rank | Top positive | Score | Top negative | Score |
|---|---|---|---|---|
| 1 | eric.bass@enron.com | 12 | johnny.palmer@enron.com | 1 |
| 2 | patti.thompson@enron.com | 12 | bobette.riner@ipgdirect.com | 3 |
| 3 | lydia.delgado@enron.com | 10 | john.arnold@enron.com | 4 |

eric.bass and patti.thompson are tied at 12; the tie is broken alphabetically. Rankings for all 24 months are in `data/processed/monthly_rankings.csv` and in the notebook. Over the whole period, **john.arnold** made the monthly top-3 positive most often (13 of 24 months). **kayne.coulter** and **don.baughman** made the top-3 negative most often (11 months each).

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

A calendar-month count would catch only 4 of these 7. The rolling window also finds bobette.riner, john.arnold and johnny.palmer, whose bursts of negativity cross a month boundary.

### Predictive model

Linear regression of the monthly score achieves **test R² = 0.82**, MAE = 1.37 and RMSE = 1.79. A mean-only baseline scores MAE = 3.57 and RMSE = 4.25. 5-fold cross-validation gives R² = 0.80 ± 0.03, so the model does not overfit. **Message volume is the dominant driver**: more messages lead to a higher score, because most messages are positive.

### Key insights and recommendations

1. **Sentiment is healthy and stable.** About 72% of messages are positive, and the company-wide average score rises slightly (+0.045 points per month). Risk appears as short **bursts** of negativity, not as persistently negative employees.
2. **Prioritise sally.beck and kayne.coulter for check-ins.**
   - sally.beck has the strongest burst (5 negative messages in 30 days) and the highest negative share overall.
   - kayne.coulter is in the top-3 negative in 11 of 24 months and has three non-positive months.
3. **Use the rolling-window flag in reporting, not a monthly count.** It surfaces 3 additional at-risk employees.
4. **Report the average score per message alongside the monthly sum.** The sum rewards volume, so quiet employees can look less engaged than they are.
5. **Data caveats:**
   - Message dates look synthetically assigned: monthly volume is flat and weekends are as busy as weekdays.
   - 625 messages repeat earlier content.
   - The lexicon labeler marks a few factual messages (e.g. automatic warnings) as negative.

   Use the LLM labeler (`SENTIMENT_METHOD=llm`) for production, and review flagged windows before acting.

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
│   └── report.py          # builds reports/Final_Report.docx from outputs
├── tests/test_logic.py    # unit tests for scoring, ranking, flight-risk boundaries, cleaning
├── data/
│   ├── raw/test.csv       # input dataset
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
python run_pipeline.py                  # VADER labeling (default)
python -m src.report                    # rebuild reports/Final_Report.docx
pytest -q                               # run the unit tests
```

**Optional labeling back-ends**

```bash
# LLM (Anthropic or OpenAI) - set the provider, model and API key in .env first
pip install anthropic                   # or: pip install openai
python run_pipeline.py --method llm

# Hugging Face transformer
pip install transformers torch
python run_pipeline.py --method transformer
```

Outputs are written to `data/processed/` (CSV/JSON) and `visualization/` (PNG).

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
| `vader` *(default)* | Lexicon + rule model built for informal text. Handles negation, intensifiers, punctuation and capitalisation. | Deterministic, offline |
| `transformer` | `cardiffnlp/twitter-roberta-base-sentiment-latest` | Deterministic |
| `llm` | Claude or GPT, zero-shot prompt, temperature 0 | Cached per message hash |

**Labeling rule (VADER):** compound ≥ 0.05 → **Positive**; compound ≤ −0.05 → **Negative**; otherwise **Neutral**. These are the standard thresholds from Hutto & Gilbert (2014) and are configurable in `.env`.

VADER is the default so any reviewer can reproduce the exact results without API keys or paid calls. The LLM back-end is the accuracy upgrade and needs no other code changes.

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
