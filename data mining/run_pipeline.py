"""
WebPathMine — Full Pipeline Runner
Run this script once to download data, train all models, and cache results
so the Streamlit dashboard can load them instantly.

Usage:
    python run_pipeline.py
    python run_pipeline.py --force-download   # re-download datasets
    python run_pipeline.py --help
"""

import sys
import os
import time
import argparse
import warnings

# Force UTF-8 output — prevents UnicodeEncodeError from emoji in print() on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

# Make src importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_loader import load_all, generate_synthetic_clickstream
from preprocessor import preprocess_shoppers, preprocess_clickstream
from feature_engineering import engineer_features, get_feature_matrix, get_cluster_features
from pattern_mining import compare_algorithms, encode_transactions
from clustering import run_full_clustering
from classification import train_and_evaluate, get_comparison_table
from evaluator import save_results


def main(force_download: bool = False):
    t_start = time.perf_counter()
    print("=" * 60)
    print("  [WebPathMine] Full Pipeline")
    print("=" * 60)

    # -- 1. Load data --------------------------------------------------
    datasets = load_all(force_download=force_download)
    df_shoppers_raw = datasets["shoppers"]
    df_click_raw    = datasets["clickstream"]

    # -- 2. Preprocess -----------------------------------------------
    df_shoppers = preprocess_shoppers(df_shoppers_raw)

    try:
        df_click, sessions = preprocess_clickstream(df_click_raw)
        if len(sessions) < 10:
            raise ValueError("Too few sessions; falling back to synthetic data.")
    except Exception as e:
        print(f"   ⚠️  Clickstream preprocessing issue: {e}")
        print("   → Using synthetic clickstream data ...")
        df_click_raw = generate_synthetic_clickstream(n_sessions=600)
        df_click, sessions = preprocess_clickstream(df_click_raw)

    # -- 3. Feature engineering ----------------------------------------
    df_shoppers = engineer_features(df_shoppers)
    X, y = get_feature_matrix(df_shoppers)
    X_cluster = get_cluster_features(df_shoppers)

    # -- 4. Pattern mining --------------------------------------------
    mining_results = compare_algorithms(
        sessions,
        min_support=0.05,
        min_confidence=0.3,
    )

    # -- 5. Clustering -----------------------------------------------
    clustering_results = run_full_clustering(X_cluster, k_range=range(2, 8))

    # -- 6. Classification --------------------------------------------
    clf_output = train_and_evaluate(X, y, test_size=0.2, scale=True)
    comparison_table = get_comparison_table(clf_output["results"])

    print("\n[Model Comparison]")
    print(comparison_table.to_string())

    # -- 7. Save all results ------------------------------------------
    results = {
        "shoppers_df": df_shoppers,
        "shoppers_raw_df": df_shoppers_raw,
        "clickstream_df": df_click,
        "sessions": sessions,
        "mining": mining_results,
        "clustering": clustering_results,
        "classification": clf_output,
        "comparison_table": comparison_table,
    }
    save_results(results)

    elapsed = time.perf_counter() - t_start
    print(f"\n[DONE] Pipeline complete in {elapsed:.1f}s")
    print("   Run the dashboard with:  streamlit run dashboard/app.py")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebPathMine Pipeline Runner")
    parser.add_argument(
        "--force-download", action="store_true",
        help="Force re-download of datasets from UCI."
    )
    args = parser.parse_args()
    main(force_download=args.force_download)
