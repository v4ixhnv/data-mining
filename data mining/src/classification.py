"""
WebPathMine — Classification
ID3 Decision Tree, Gini Decision Tree, and Neural Network classifiers.
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, roc_curve,
    classification_report,
)
from sklearn.inspection import permutation_importance
import joblib
import os

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def build_models() -> dict:
    """Return the three classifier definitions."""
    return {
        "ID3 Decision Tree": DecisionTreeClassifier(
            criterion="entropy",  # Information Gain / ID3
            max_depth=8,
            min_samples_leaf=5,
            random_state=42,
        ),
        "Gini Decision Tree": DecisionTreeClassifier(
            criterion="gini",
            max_depth=8,
            min_samples_leaf=5,
            random_state=42,
        ),
        "Neural Network": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            learning_rate_init=0.001,
        ),
    }


def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    scale: bool = True,
) -> dict:
    """
    Train all three classifiers and compute comprehensive evaluation metrics.

    Parameters
    ----------
    X : feature DataFrame
    y : target Series (binary: 0/1)
    test_size : fraction held out for evaluation
    scale : whether to MinMax-scale features (recommended for NN)

    Returns
    -------
    dict with per-model results + shared split info
    """
    print("\n🤖 Training classifiers ...")

    # Train / test split (stratified to preserve class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    scaler = None
    if scale:
        scaler = MinMaxScaler()
        X_train_s = pd.DataFrame(
            scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
        )
        X_test_s = pd.DataFrame(
            scaler.transform(X_test), columns=X_test.columns, index=X_test.index
        )
    else:
        X_train_s, X_test_s = X_train, X_test

    results = {}
    models = build_models()

    for name, model in models.items():
        print(f"   Training {name} ...")
        model.fit(X_train_s, y_train)

        y_pred = model.predict(X_test_s)
        y_prob = (
            model.predict_proba(X_test_s)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)
        cm   = confusion_matrix(y_test, y_pred)
        auc  = roc_auc_score(y_test, y_prob) if y_prob is not None else None
        fpr, tpr, _ = roc_curve(y_test, y_prob) if y_prob is not None else (None, None, None)

        # 5-fold CV accuracy
        cv_scores = cross_val_score(model, X_train_s, y_train, cv=5, scoring="accuracy")

        # Feature importance
        if hasattr(model, "feature_importances_"):
            feat_imp = pd.Series(
                model.feature_importances_, index=X_train.columns
            ).sort_values(ascending=False)
        else:
            # Permutation importance for NN (on test set)
            perm = permutation_importance(
                model, X_test_s, y_test, n_repeats=5, random_state=42
            )
            feat_imp = pd.Series(
                perm.importances_mean, index=X_train.columns
            ).sort_values(ascending=False)

        auc_str = f"{auc:.4f}" if auc is not None else "N/A"
        print(f"     Acc={acc:.4f} | Prec={prec:.4f} | Rec={rec:.4f} | F1={f1:.4f} | AUC={auc_str}")

        results[name] = {
            "model": model,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "auc": auc,
            "fpr": fpr,
            "tpr": tpr,
            "confusion_matrix": cm,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "feature_importance": feat_imp,
            "y_pred": y_pred,
            "y_prob": y_prob,
        }

        # Save model
        joblib.dump(model, os.path.join(MODELS_DIR, f"{name.replace(' ', '_')}.pkl"))

    return {
        "results": results,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "feature_cols": list(X.columns),
    }


def get_comparison_table(results: dict) -> pd.DataFrame:
    """Build a summary DataFrame comparing all models."""
    rows = []
    for name, r in results.items():
        rows.append({
            "Model": name,
            "Accuracy": f"{r['accuracy']:.4f}",
            "Precision": f"{r['precision']:.4f}",
            "Recall": f"{r['recall']:.4f}",
            "F1 Score": f"{r['f1']:.4f}",
            "AUC-ROC": f"{r['auc']:.4f}" if r["auc"] else "N/A",
            "CV Acc (mean±std)": f"{r['cv_mean']:.4f} ± {r['cv_std']:.4f}",
        })
    return pd.DataFrame(rows).set_index("Model")
