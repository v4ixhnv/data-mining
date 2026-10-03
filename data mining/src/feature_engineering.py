"""
WebPathMine — Feature Engineering
Extracts behavioral session-level features from the Online Shoppers dataset.
"""

import pandas as pd
import numpy as np


# ─── Feature definitions ──────────────────────────────────────────────────────

FEATURE_COLS = [
    # Page view counts
    "Administrative", "Informational", "ProductRelated",
    # Time spent
    "Administrative_Duration", "Informational_Duration", "ProductRelated_Duration",
    # Search engine metrics
    "BounceRates", "ExitRates", "PageValues",
    # Temporal / user-type
    "SpecialDay", "Month", "OperatingSystems", "Browser",
    "Region", "TrafficType", "VisitorType", "Weekend",
    # Derived (added below)
    "TotalPages", "TotalDuration", "Avg_Duration_Per_Page",
    "HighIntent", "Cart_Proxy",
]

TARGET_COL = "Revenue"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived behavioral features to the Online Shoppers DataFrame.

    New columns:
    - TotalPages          : sum of Administrative + Informational + ProductRelated views
    - TotalDuration       : total time spent across all page types (seconds)
    - Avg_Duration_Per_Page: TotalDuration / TotalPages (handles divide-by-zero)
    - HighIntent          : 1 if PageValues > median PageValues (proxy for cart/checkout)
    - Cart_Proxy          : 1 if ProductRelated views > 10 AND BounceRates < 0.05
    """
    df = df.copy()

    df["TotalPages"] = (
        df["Administrative"].fillna(0)
        + df["Informational"].fillna(0)
        + df["ProductRelated"].fillna(0)
    )

    df["TotalDuration"] = (
        df["Administrative_Duration"].fillna(0)
        + df["Informational_Duration"].fillna(0)
        + df["ProductRelated_Duration"].fillna(0)
    )

    df["Avg_Duration_Per_Page"] = np.where(
        df["TotalPages"] > 0,
        df["TotalDuration"] / df["TotalPages"],
        0.0,
    )

    median_pv = df["PageValues"].median()
    df["HighIntent"] = (df["PageValues"] > median_pv).astype(int)

    df["Cart_Proxy"] = (
        (df["ProductRelated"] > 10) & (df["BounceRates"] < 0.05)
    ).astype(int)

    print("\n⚙️  Feature engineering complete.")
    print(f"   Total features (incl. derived): {df.shape[1]}")
    print(f"   HighIntent sessions: {df['HighIntent'].mean():.1%}")
    print(f"   Cart_Proxy sessions: {df['Cart_Proxy'].mean():.1%}")
    return df


def get_feature_matrix(df: pd.DataFrame) -> tuple:
    """
    Return (X, y) where X contains only the defined FEATURE_COLS
    (that are present in df) and y is the Revenue target.
    """
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].copy()
    y = df[TARGET_COL].copy()
    return X, y


def get_cluster_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a scaled feature matrix suitable for K-Means clustering.
    Uses a subset of numeric features most relevant to browsing behavior.
    """
    cluster_cols = [
        "TotalPages", "TotalDuration", "Avg_Duration_Per_Page",
        "BounceRates", "ExitRates", "PageValues", "HighIntent", "Cart_Proxy",
    ]
    available = [c for c in cluster_cols if c in df.columns]
    return df[available].copy()
