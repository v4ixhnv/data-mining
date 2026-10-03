"""
WebPathMine — Preprocessor
Cleans, encodes, and sessionizes raw datasets.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)


# ─── Online Shoppers ──────────────────────────────────────────────────────────

MONTH_ORDER = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "June": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}

def preprocess_shoppers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and encode the Online Shoppers Purchasing Intention dataset.

    Steps:
    1. Drop exact duplicate rows.
    2. Fill numeric NaN with column median.
    3. Ordinal-encode Month (Jan→1 … Dec→12).
    4. Label-encode VisitorType (Returning/New/Other → 0/1/2).
    5. Cast boolean Weekend + Revenue to int.
    6. Return processed DataFrame with 'Revenue' as target column (int).
    """
    print("\n🔧 Preprocessing Online Shoppers dataset ...")
    df = df.copy()

    # 1. Drop duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"   Removed {before - len(df):,} duplicate rows")

    # 2. Fill numeric NaN
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # 3. Month → ordinal
    if "Month" in df.columns:
        df["Month"] = df["Month"].map(MONTH_ORDER).fillna(0).astype(int)

    # 4. VisitorType → label-encoded
    if "VisitorType" in df.columns:
        le = LabelEncoder()
        df["VisitorType"] = le.fit_transform(df["VisitorType"].astype(str))

    # 5. Booleans → int
    for col in ["Weekend", "Revenue"]:
        if col in df.columns:
            df[col] = df[col].astype(int)

    print(f"   Final shape: {df.shape[0]:,} rows × {df.shape[1]} cols")
    print(f"   Purchase rate: {df['Revenue'].mean():.1%}")
    return df


def scale_shoppers(df: pd.DataFrame, feature_cols: list) -> tuple:
    """
    MinMax-scale the specified feature columns.
    Returns (X_scaled_df, scaler).
    """
    scaler = MinMaxScaler()
    X = df[feature_cols].copy()
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=feature_cols, index=df.index), scaler


# ─── Clickstream ──────────────────────────────────────────────────────────────

def preprocess_clickstream(df: pd.DataFrame) -> tuple:
    """
    Clean and sessionize the Clickstream dataset.

    Returns:
        df_clean : cleaned DataFrame
        sessions : list of lists — each inner list is an ordered page sequence
                   for one session, e.g. [['Electronics', 'Shoes'], ...]
    """
    print("\n🔧 Preprocessing Clickstream dataset ...")
    df = df.copy()

    # Standardise column names (lowercase, strip whitespace)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Identify key columns flexibly
    session_col = _find_col(df, ["session_id", "session", "sid"])
    page_col = _find_col(df, ["page_2", "page2", "page", "category", "page_1_main_category"])
    order_col = _find_col(df, ["order", "click_order", "seq"])

    if session_col is None:
        # Fall back: treat every unique row index as its own identifier
        df["session_id"] = df.index
        session_col = "session_id"

    if page_col is None:
        raise ValueError("Cannot find a page/category column in clickstream data.")

    # Drop rows with null page or session values
    df.dropna(subset=[session_col, page_col], inplace=True)

    # Sort within sessions
    if order_col and order_col != session_col:
        df.sort_values([session_col, order_col], inplace=True)

    # Build session sequences
    sessions = (
        df.groupby(session_col)[page_col]
        .apply(lambda x: list(x.astype(str).str.strip()))
        .tolist()
    )
    # Filter out single-page sessions (no association rules possible)
    sessions = [s for s in sessions if len(s) > 1]

    print(f"   Sessions (≥2 pages): {len(sessions):,}")
    avg_len = np.mean([len(s) for s in sessions])
    print(f"   Avg session length: {avg_len:.1f} pages")
    return df, sessions


def _find_col(df: pd.DataFrame, candidates: list) -> str | None:
    """Return the first candidate column name found in df.columns."""
    for c in candidates:
        if c in df.columns:
            return c
    return None
