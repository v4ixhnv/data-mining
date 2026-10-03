"""
WebPathMine — Data Loader
Downloads and caches both UCI datasets locally.
"""

import os
import requests
import pandas as pd
import numpy as np

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

# ─── Dataset URLs ─────────────────────────────────────────────────────────────
SHOPPERS_URL = (
    "https://archive.ics.uci.edu/static/public/468/"
    "online+shoppers+purchasing+intention+dataset.zip"
)
CLICKSTREAM_URL = (
    "https://archive.ics.uci.edu/static/public/553/"
    "clickstream+data+for+online+shopping.zip"
)

SHOPPERS_PATH = os.path.join(RAW_DIR, "online_shoppers.csv")
CLICKSTREAM_PATH = os.path.join(RAW_DIR, "clickstream.csv")


def _download(url: str, dest: str, csv_name_hint: str = "") -> None:
    """Download a file (raw CSV or ZIP archive) with progress indication."""
    import io, zipfile
    print(f"  Downloading {os.path.basename(dest)} ...")
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        # If the response is a ZIP, extract the first matching CSV inside
        if r.headers.get("Content-Type", "").startswith("application/zip") or url.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
                if not csv_names:
                    raise ValueError("No CSV found inside ZIP archive.")
                # Prefer a name that matches the hint, otherwise take the first
                pick = next((n for n in csv_names if csv_name_hint.lower() in n.lower()), csv_names[0])
                with zf.open(pick) as src, open(dest, "wb") as dst:
                    dst.write(src.read())
        else:
            with open(dest, "wb") as f:
                f.write(r.content)
        print(f"  ✔ Saved to {dest}")
    except Exception as e:
        print(f"  ✘ Download failed: {e}")
        raise


def load_shoppers(force_download: bool = False) -> pd.DataFrame:
    """
    Load the Online Shoppers Purchasing Intention dataset.
    17 features, 12,330 sessions, binary target: Revenue (True/False).
    """
    if not os.path.exists(SHOPPERS_PATH) or force_download:
        try:
            _download(SHOPPERS_URL, SHOPPERS_PATH, csv_name_hint="shoppers")
        except Exception:
            # Fallback to legacy direct-CSV URL
            _download(
                "https://archive.ics.uci.edu/ml/machine-learning-databases/00468/online_shoppers_intention.csv",
                SHOPPERS_PATH,
            )
    df = pd.read_csv(SHOPPERS_PATH)
    print(f"  Online Shoppers dataset: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


def load_clickstream(force_download: bool = False) -> pd.DataFrame:
    """
    Load the Clickstream Data for Online Shopping dataset.
    Contains product page sequences with session and order information.
    """
    if not os.path.exists(CLICKSTREAM_PATH) or force_download:
        try:
            _download(CLICKSTREAM_URL, CLICKSTREAM_PATH, csv_name_hint="clickstream")
        except Exception:
            # Fallback to legacy direct-CSV URL
            _download(
                "https://archive.ics.uci.edu/ml/machine-learning-databases/00553/clickstream_data_for_online_shopping.csv",
                CLICKSTREAM_PATH,
            )
    df = pd.read_csv(CLICKSTREAM_PATH, sep=";", encoding="latin-1")
    print(f"  Clickstream dataset: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


def load_all(force_download: bool = False) -> dict:
    """Load both datasets and return as a dictionary."""
    print("\n📥 Loading datasets ...")
    return {
        "shoppers": load_shoppers(force_download),
        "clickstream": load_clickstream(force_download),
    }


def generate_synthetic_clickstream(n_sessions: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic clickstream data for demonstration when the UCI
    clickstream dataset is unavailable or malformed.
    """
    rng = np.random.default_rng(seed)
    categories = [
        "Home", "Electronics", "Clothing", "Books", "Sports",
        "Toys", "Grocery", "Beauty", "Automotive", "Garden",
    ]
    records = []
    for sid in range(1, n_sessions + 1):
        n_pages = rng.integers(2, 10)
        pages = rng.choice(categories, size=n_pages, replace=True)
        for order, page in enumerate(pages, 1):
            records.append({
                "session_id": sid,
                "order": order,
                "page_2": page,
                "country": rng.choice(["PL", "DE", "UK", "US", "FR"]),
                "page_1_main_category": rng.choice(["trousers", "shoes", "bags", "accesories"]),
            })
    return pd.DataFrame(records)
