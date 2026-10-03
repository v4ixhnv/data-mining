"""
WebPathMine — Clustering
K-Means clustering with Elbow method and Silhouette analysis.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score



def run_elbow_silhouette(X_scaled: np.ndarray, k_range: range = range(2, 9)) -> dict:
    """
    Compute inertia (Elbow method) and Silhouette scores for a range of k.

    Returns dict with keys 'k_values', 'inertias', 'silhouettes'.
    """
    print("\n📊 Computing Elbow and Silhouette scores ...")
    inertias = []
    silhouettes = []
    k_values = list(k_range)

    for k in k_values:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil = silhouette_score(X_scaled, labels) if k > 1 else 0.0
        silhouettes.append(sil)
        print(f"   k={k}: inertia={km.inertia_:.1f}, silhouette={sil:.4f}")

    best_k = k_values[int(np.argmax(silhouettes))]
    print(f"   → Best k by silhouette: {best_k}")
    return {
        "k_values": k_values,
        "inertias": inertias,
        "silhouettes": silhouettes,
        "best_k": best_k,
    }


def fit_kmeans(X_scaled: np.ndarray, n_clusters: int = 3) -> tuple:
    """
    Fit K-Means with the given number of clusters.

    Returns (kmeans_model, labels_array).
    """
    print(f"\n🔵 Fitting K-Means with k={n_clusters} ...")
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    print(f"   Silhouette score: {sil:.4f}")
    return km, labels


def pca_project(X_scaled: np.ndarray, n_components: int = 2) -> tuple:
    """
    Project high-dimensional features to 2D via PCA for visualization.

    Returns (pca_coords_2d, explained_variance_ratio).
    """
    pca = PCA(n_components=n_components, random_state=42)
    coords = pca.fit_transform(X_scaled)
    return coords, pca.explained_variance_ratio_


def cluster_profiles(df_features: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """
    Compute per-cluster mean feature profiles and assign persona labels.

    Returns a DataFrame with cluster stats.
    """
    df = df_features.copy()
    df["Cluster"] = labels

    profile = df.groupby("Cluster").mean().round(4)
    profile["Count"] = df.groupby("Cluster").size()
    profile["Pct"] = (profile["Count"] / len(df) * 100).round(1)

    # Assign persona labels heuristically based on PageValues and TotalPages
    personas = []
    for idx, row in profile.iterrows():
        pv = row.get("PageValues", 0)
        tp = row.get("TotalPages", 0)
        br = row.get("BounceRates", 1)

        if pv > profile["PageValues"].median() and tp > profile["TotalPages"].median():
            personas.append("🎯 High-Intent Buyer")
        elif br > profile["BounceRates"].median():
            personas.append("💨 Quick Visitor")
        elif tp > profile["TotalPages"].median():
            personas.append("🔍 Product Explorer")
        else:
            personas.append("📖 Research Browser")

    profile["Persona"] = personas
    return profile.reset_index()


def run_full_clustering(X: pd.DataFrame, k_range: range = range(2, 8)) -> dict:
    """
    Complete clustering pipeline:
    1. Scale features
    2. Elbow + Silhouette analysis
    3. Fit best K-Means
    4. PCA projection
    5. Cluster profiles

    Returns dict with all results.
    """
    print("\n🗂️  Running full clustering pipeline ...")
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    elbow_data = run_elbow_silhouette(X_scaled, k_range)
    best_k = elbow_data["best_k"]

    km, labels = fit_kmeans(X_scaled, n_clusters=best_k)
    pca_coords, var_ratio = pca_project(X_scaled)
    profiles = cluster_profiles(X, labels)

    return {
        "scaler": scaler,
        "kmeans": km,
        "labels": labels,
        "best_k": best_k,
        "elbow": elbow_data,
        "pca_coords": pca_coords,
        "pca_variance_ratio": var_ratio,
        "profiles": profiles,
    }
