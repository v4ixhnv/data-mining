# 🌐 WebPathMine

> **Web Usage & Navigation Pattern Mining with User Behaviour Prediction**  
> E-Commerce Analytics Platform | Data Mining Course Project

---

## 👥 Team Members

| Name | Roll No |
|------|---------|
| Johan Gejo | 320 |
| Karthikeyan Biju | 323 |
| Ron John Kurien | 351 |
| Vaishnav S Kumar | 366 |

---

## 📋 Overview

WebPathMine is a unified data-mining system that transforms raw e-commerce clickstream and session logs into actionable business intelligence. The pipeline applies:

- **Association Rule Mining** (Apriori & FP-Growth) to discover frequent navigation patterns
- **K-Means Clustering** to segment users into behavioural groups
- **Classification** (ID3 DT, Gini DT, Neural Network) to predict purchase intent
- **Interactive Streamlit Dashboard** to visualise all discovered patterns and predictions

---

## 🗂️ Project Structure

```
data mining/
├── data/
│   ├── raw/            ← auto-downloaded UCI datasets
│   └── processed/      ← cached pipeline outputs
├── src/
│   ├── data_loader.py
│   ├── preprocessor.py
│   ├── feature_engineering.py
│   ├── pattern_mining.py
│   ├── clustering.py
│   ├── classification.py
│   └── evaluator.py
├── dashboard/
│   └── app.py          ← Streamlit dashboard
├── run_pipeline.py     ← CLI entry point
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline (downloads data + trains models)

```bash
python run_pipeline.py
```

> Optional: `python run_pipeline.py --force-download` to re-download datasets.

### 3. Launch the dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard will open at `http://localhost:8501`.

---

## 📊 Datasets

| Dataset | Source | Rows | Task |
|---------|--------|------|------|
| Online Shoppers Purchasing Intention | [UCI #468](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset) | 12,330 | Classification + Clustering |
| Clickstream Data for Online Shopping | [UCI #553](https://archive.ics.uci.edu/ml/datasets/Clickstream+Data+for+Online+Shopping) | ~165K events | Pattern Mining |

---

## 🔬 Algorithm Pipeline

### Stage 1 — Data Preprocessing
- Drop duplicates, fill missing values with column median
- Ordinal-encode `Month`, label-encode `VisitorType`
- Sessionize clickstream events by `session_id`

### Stage 2 — Feature Engineering
New derived features:
- `TotalPages` — sum of all page category views
- `TotalDuration` — total session time across all page types
- `Avg_Duration_Per_Page` — average dwell time per page
- `HighIntent` — binary flag: `PageValues > median`
- `Cart_Proxy` — binary flag: `ProductRelated > 10 AND BounceRate < 5%`

### Stage 3 — Association Rule Mining
| Parameter | Default |
|-----------|---------|
| Algorithm | Apriori, FP-Growth |
| Min Support | 0.05 |
| Min Confidence | 0.30 |
| Min Lift | 1.0 |

Metrics reported: **support, confidence, lift, conviction**

### Stage 4 — K-Means Clustering
- Feature scaling (MinMax)
- Elbow method (k = 2–8) + Silhouette analysis
- PCA 2D projection for visualisation
- Behavioural personas: High-Intent Buyer, Product Explorer, Quick Visitor, Research Browser

### Stage 5 — Classification
| Model | Criterion | Layers |
|-------|-----------|--------|
| ID3 Decision Tree | Entropy | max_depth=8 |
| Gini Decision Tree | Gini | max_depth=8 |
| Neural Network | — | (128, 64), ReLU, Adam |

Metrics: **Accuracy, Precision, Recall, F1, AUC-ROC**, 5-fold Cross-Validation

---

## 📈 Dashboard Pages

| Page | Description |
|------|-------------|
| 🏠 Overview | KPIs, pipeline architecture, dataset info |
| 🔍 Data Explorer | Data preview, distributions, correlation heatmap, session stats |
| ⛏️ Pattern Mining | Apriori vs FP-Growth rules, support/confidence scatter, speed comparison |
| 🤖 Classification | Radar chart, confusion matrices, ROC curves, feature importance |
| 🗂️ Clustering | Elbow + silhouette plots, PCA cluster scatter, persona cards |
| 📊 Evaluation | Full metric comparison, CV results, downloadable CSV |

---

## 📚 References

1. UCI Machine Learning Repository. *Clickstream Data for Online Shopping Dataset* (2019). Dataset #553.
2. UCI Machine Learning Repository. *Online Shoppers Purchasing Intention Dataset* (2018). Dataset #468.
3. Agrawal, R., & Srikant, R. (1994). Fast algorithms for mining association rules. *VLDB*.
4. Han, J., Pei, J., & Yin, Y. (2000). Mining frequent patterns without candidate generation. *SIGMOD*.
5. Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. *JMLR*.
