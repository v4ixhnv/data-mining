"""
WebPathMine — Pattern Mining
Implements Apriori and FP-Growth association rule mining using MLxtend.
"""

import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules


def encode_transactions(sessions: list) -> pd.DataFrame:
    """
    One-hot encode session page sequences into a boolean transaction matrix.

    Parameters
    ----------
    sessions : list of lists
        Each inner list is an ordered sequence of page/category visits for one session.

    Returns
    -------
    pd.DataFrame : boolean transaction matrix (rows=sessions, cols=unique pages)
    """
    te = TransactionEncoder()
    te_array = te.fit_transform(sessions)
    return pd.DataFrame(te_array, columns=te.columns_)


def run_apriori(
    te_df: pd.DataFrame,
    min_support: float = 0.05,
    min_confidence: float = 0.3,
    min_lift: float = 1.0,
    max_len: int | None = None,
) -> tuple:
    """
    Run Apriori algorithm and extract association rules.

    Returns
    -------
    (frequent_itemsets, rules) DataFrames
    """
    print(f"\n⛏️  Running Apriori (min_support={min_support}, min_confidence={min_confidence}) ...")
    frequent_itemsets = apriori(
        te_df,
        min_support=min_support,
        use_colnames=True,
        max_len=max_len,
    )
    frequent_itemsets["length"] = frequent_itemsets["itemsets"].apply(len)

    if frequent_itemsets.empty:
        print("   ⚠️  No frequent itemsets found. Try lowering min_support.")
        return frequent_itemsets, pd.DataFrame()

    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=min_confidence,
        num_itemsets=len(frequent_itemsets),
    )
    rules = rules[rules["lift"] >= min_lift].sort_values("lift", ascending=False)

    print(f"   Frequent itemsets: {len(frequent_itemsets):,}")
    print(f"   Association rules: {len(rules):,}")
    return frequent_itemsets, rules


def run_fpgrowth(
    te_df: pd.DataFrame,
    min_support: float = 0.05,
    min_confidence: float = 0.3,
    min_lift: float = 1.0,
    max_len: int | None = None,
) -> tuple:
    """
    Run FP-Growth algorithm and extract association rules.

    Returns
    -------
    (frequent_itemsets, rules) DataFrames
    """
    print(f"\n⛏️  Running FP-Growth (min_support={min_support}, min_confidence={min_confidence}) ...")
    frequent_itemsets = fpgrowth(
        te_df,
        min_support=min_support,
        use_colnames=True,
        max_len=max_len,
    )
    frequent_itemsets["length"] = frequent_itemsets["itemsets"].apply(len)

    if frequent_itemsets.empty:
        print("   ⚠️  No frequent itemsets found. Try lowering min_support.")
        return frequent_itemsets, pd.DataFrame()

    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=min_confidence,
        num_itemsets=len(frequent_itemsets),
    )
    rules = rules[rules["lift"] >= min_lift].sort_values("lift", ascending=False)

    print(f"   Frequent itemsets: {len(frequent_itemsets):,}")
    print(f"   Association rules: {len(rules):,}")
    return frequent_itemsets, rules


def compare_algorithms(
    sessions: list,
    min_support: float = 0.05,
    min_confidence: float = 0.3,
) -> dict:
    """
    Run both algorithms on the same session data and return a comparison dict.
    """
    import time

    te_df = encode_transactions(sessions)

    # Apriori
    t0 = time.perf_counter()
    fi_ap, rules_ap = run_apriori(te_df, min_support, min_confidence)
    t_ap = time.perf_counter() - t0

    # FP-Growth
    t0 = time.perf_counter()
    fi_fp, rules_fp = run_fpgrowth(te_df, min_support, min_confidence)
    t_fp = time.perf_counter() - t0

    return {
        "te_df": te_df,
        "apriori": {
            "frequent_itemsets": fi_ap,
            "rules": rules_ap,
            "runtime_sec": round(t_ap, 4),
        },
        "fpgrowth": {
            "frequent_itemsets": fi_fp,
            "rules": rules_fp,
            "runtime_sec": round(t_fp, 4),
        },
    }


def format_rules_for_display(rules: pd.DataFrame) -> pd.DataFrame:
    """Convert frozenset columns to readable strings for display."""
    if rules.empty:
        return rules
    display = rules.copy()
    display["antecedents"] = display["antecedents"].apply(lambda x: " → ".join(sorted(x)))
    display["consequents"] = display["consequents"].apply(lambda x: " → ".join(sorted(x)))
    cols = ["antecedents", "consequents", "support", "confidence", "lift"]
    if "conviction" in display.columns:
        cols.append("conviction")
    return display[cols].reset_index(drop=True)
