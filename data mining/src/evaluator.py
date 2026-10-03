"""
WebPathMine — Evaluator
Persists and loads pipeline results, builds comparison tables.
"""

import os
import joblib
import pandas as pd
import numpy as np

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

RESULTS_PATH = os.path.join(PROCESSED_DIR, "pipeline_results.pkl")


def save_results(results: dict) -> None:
    """Persist the full pipeline results dict to disk."""
    joblib.dump(results, RESULTS_PATH)
    print(f"\n💾 Results saved to {RESULTS_PATH}")


def load_results() -> dict | None:
    """Load persisted pipeline results if they exist."""
    if os.path.exists(RESULTS_PATH):
        print(f"\n📂 Loading cached results from {RESULTS_PATH}")
        return joblib.load(RESULTS_PATH)
    return None


def results_exist() -> bool:
    return os.path.exists(RESULTS_PATH)


def format_rules_table(rules: pd.DataFrame) -> pd.DataFrame:
    """Pretty-format association rules for display."""
    if rules.empty:
        return pd.DataFrame(columns=["Antecedent", "Consequent", "Support", "Confidence", "Lift"])
    df = rules.copy()
    df["Antecedent"] = df["antecedents"].apply(lambda x: ", ".join(sorted(x)))
    df["Consequent"] = df["consequents"].apply(lambda x: ", ".join(sorted(x)))
    df["Support"] = df["support"].round(4)
    df["Confidence"] = df["confidence"].round(4)
    df["Lift"] = df["lift"].round(4)
    return df[["Antecedent", "Consequent", "Support", "Confidence", "Lift"]].reset_index(drop=True)
