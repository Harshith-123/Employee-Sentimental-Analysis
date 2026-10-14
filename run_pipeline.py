"""Reproduce every output (labeled data, scores, rankings, flight risks,
model metrics, charts) from the raw CSV with a single command.

    python run_pipeline.py                     # default VADER labeling
    python run_pipeline.py --method llm        # LLM labeling (needs API key in .env)
    python run_pipeline.py --method transformer
"""
import argparse

from src.pipeline import run

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Employee sentiment analysis pipeline")
    p.add_argument("--method", choices=["vader", "transformer", "llm"], default=None,
                   help="sentiment labeling backend (default: SENTIMENT_METHOD env var or 'vader')")
    args = p.parse_args()
    run(method=args.method)
