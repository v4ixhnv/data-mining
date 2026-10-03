"""
WebPathMine — Interactive Streamlit Dashboard
Multi-page data mining visualization for e-commerce behaviour analysis.
"""

import sys
import os
import warnings
import time

# Force UTF-8 output — prevents UnicodeEncodeError from emoji in print() on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WebPathMine | E-Commerce Analytics",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Styling ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Import Inter font */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
  }

  /* Dark sidebar */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
  }
  [data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
  }
  [data-testid="stSidebar"] .stSelectbox label,
  [data-testid="stSidebar"] .stSlider label,
  [data-testid="stSidebar"] .stRadio label {
    color: #94a3b8 !important;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  /* Main background */
  .main .block-container {
    background: #0d1117;
    padding-top: 1.5rem;
  }
  body { background-color: #0d1117; }

  /* Metric cards */
  .metric-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
  }
  .metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 25px rgba(99, 102, 241, 0.3);
  }
  .metric-value {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  .metric-label {
    font-size: 0.8rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.3rem;
  }
  .metric-delta {
    font-size: 0.85rem;
    color: #22c55e;
    margin-top: 0.2rem;
  }

  /* Section headers */
  .section-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: #f1f5f9;
    border-left: 4px solid #6366f1;
    padding-left: 0.8rem;
    margin-bottom: 1rem;
  }

  /* Rule cards */
  .rule-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.5rem;
  }
  .rule-arrow {
    color: #6366f1;
    font-weight: 700;
  }

  /* Cluster persona cards */
  .persona-card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid #475569;
    border-radius: 14px;
    padding: 1.2rem;
    margin-bottom: 0.8rem;
  }
  .persona-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #e2e8f0;
  }
  .persona-stat {
    font-size: 0.85rem;
    color: #94a3b8;
  }

  /* Model comparison table */
  .model-table th {
    background: #1e293b;
    color: #94a3b8;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
  }

  /* Pipeline badge */
  .pipeline-badge {
    display: inline-block;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 0.15rem;
  }

  /* Glow button */
  .stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.2s ease;
  }
  .stButton > button:hover {
    box-shadow: 0 0 20px rgba(99, 102, 241, 0.5);
    transform: translateY(-1px);
  }

  /* Dataframe styling */
  .stDataFrame { border-radius: 10px; overflow: hidden; }

  /* Tab styling */
  .stTabs [data-baseweb="tab-list"] {
    background: #1e293b;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
  }
  .stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #94a3b8;
    font-weight: 500;
  }
  .stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
  }

  /* Alert boxes */
  .info-box {
    background: rgba(99, 102, 241, 0.1);
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    color: #c7d2fe;
    font-size: 0.9rem;
  }

  /* Header gradient */
  .hero-header {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    color: white;
  }
  .hero-title {
    font-size: 2.5rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin: 0;
  }
  .hero-subtitle {
    font-size: 1rem;
    opacity: 0.85;
    margin-top: 0.5rem;
  }
</style>
""", unsafe_allow_html=True)

# ─── Plotly dark template ─────────────────────────────────────────────────────
PLOT_TEMPLATE = "plotly_dark"
COLOR_SEQ = px.colors.qualitative.Vivid
ACCENT = "#6366f1"
ACCENT2 = "#8b5cf6"
GREEN = "#22c55e"
ORANGE = "#f97316"
RED = "#ef4444"

# ─── Helpers ──────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_cached_results():
    """Load pre-computed pipeline results from disk."""
    from evaluator import load_results
    return load_results()


def run_pipeline_inline(min_support, min_confidence, n_clusters_override=None):
    """Run the full pipeline live with given parameters (with caching)."""
    from data_loader import load_all, generate_synthetic_clickstream
    from preprocessor import preprocess_shoppers, preprocess_clickstream
    from feature_engineering import engineer_features, get_feature_matrix, get_cluster_features
    from pattern_mining import compare_algorithms
    from clustering import run_full_clustering
    from classification import train_and_evaluate, get_comparison_table
    from evaluator import save_results

    with st.spinner("📥 Loading datasets ..."):
        try:
            datasets = load_all()
            df_shoppers_raw = datasets["shoppers"]
            df_click_raw    = datasets["clickstream"]
        except Exception:
            st.warning("Could not download UCI datasets. Using built-in demonstration data.")
            df_shoppers_raw = _generate_synthetic_shoppers()
            df_click_raw    = generate_synthetic_clickstream(600)

    with st.spinner("🔧 Preprocessing ..."):
        df_shoppers = preprocess_shoppers(df_shoppers_raw)
        try:
            df_click, sessions = preprocess_clickstream(df_click_raw)
            if len(sessions) < 10:
                raise ValueError("Too few sessions.")
        except Exception:
            df_click_raw = generate_synthetic_clickstream(600)
            df_click, sessions = preprocess_clickstream(df_click_raw)

    with st.spinner("⚙️  Engineering features ..."):
        df_shoppers = engineer_features(df_shoppers)
        X, y = get_feature_matrix(df_shoppers)
        X_cluster = get_cluster_features(df_shoppers)

    with st.spinner("⛏️  Mining patterns (Apriori + FP-Growth) ..."):
        mining_results = compare_algorithms(sessions, min_support, min_confidence)

    with st.spinner("🗂️  Clustering sessions ..."):
        clustering_results = run_full_clustering(X_cluster, k_range=range(2, 8))
        if n_clusters_override:
            from clustering import fit_kmeans, pca_project, cluster_profiles
            from sklearn.preprocessing import MinMaxScaler
            sc = MinMaxScaler()
            X_sc = sc.fit_transform(X_cluster)
            km, labels = fit_kmeans(X_sc, n_clusters=n_clusters_override)
            pca_coords, var_ratio = pca_project(X_sc)
            profiles = cluster_profiles(X_cluster, labels)
            clustering_results.update({
                "kmeans": km, "labels": labels, "best_k": n_clusters_override,
                "pca_coords": pca_coords, "pca_variance_ratio": var_ratio, "profiles": profiles,
            })

    with st.spinner("🤖 Training classifiers (ID3, Gini DT, Neural Network) ..."):
        clf_output = train_and_evaluate(X, y, scale=True)
        comparison_table = get_comparison_table(clf_output["results"])

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
    return results


def _generate_synthetic_shoppers(n=12330, seed=42):
    """Fallback synthetic Online Shoppers dataset."""
    rng = np.random.default_rng(seed)
    months = ["Jan","Feb","Mar","Apr","May","June","Jul","Aug","Sep","Oct","Nov","Dec"]
    visitor_types = ["Returning_Visitor","New_Visitor","Other"]
    df = pd.DataFrame({
        "Administrative": rng.integers(0, 20, n),
        "Administrative_Duration": rng.uniform(0, 3600, n),
        "Informational": rng.integers(0, 10, n),
        "Informational_Duration": rng.uniform(0, 1800, n),
        "ProductRelated": rng.integers(0, 50, n),
        "ProductRelated_Duration": rng.uniform(0, 7200, n),
        "BounceRates": rng.uniform(0, 0.2, n),
        "ExitRates": rng.uniform(0, 0.4, n),
        "PageValues": rng.exponential(10, n),
        "SpecialDay": rng.choice([0, 0.2, 0.4, 0.6, 0.8, 1.0], n),
        "Month": rng.choice(months, n),
        "OperatingSystems": rng.integers(1, 9, n),
        "Browser": rng.integers(1, 14, n),
        "Region": rng.integers(1, 10, n),
        "TrafficType": rng.integers(1, 21, n),
        "VisitorType": rng.choice(visitor_types, n),
        "Weekend": rng.choice([True, False], n),
        "Revenue": rng.choice([True, False], n, p=[0.155, 0.845]),
    })
    return df


def metric_card(value, label, delta=None, prefix="", suffix=""):
    delta_html = f'<div class="metric-delta">▲ {delta}</div>' if delta else ""
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-value">{prefix}{value}{suffix}</div>
      <div class="metric-label">{label}</div>
      {delta_html}
    </div>""", unsafe_allow_html=True)


# ─── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding: 1rem 0 0.5rem;">
          <span style="font-size:2.5rem;">🌐</span>
          <h2 style="color:#e2e8f0; margin:0; font-size:1.3rem; font-weight:800;">WebPathMine</h2>
          <p style="color:#64748b; font-size:0.75rem; margin:0;">E-Commerce Analytics Platform</p>
        </div>
        <hr style="border-color:#334155; margin:0.8rem 0;">
        """, unsafe_allow_html=True)

        page = st.radio(
            "Navigation",
            ["🏠 Overview", "🔍 Data Explorer", "⛏️ Pattern Mining",
             "🤖 Classification", "🗂️ Clustering", "📊 Evaluation"],
            label_visibility="collapsed",
        )

        st.markdown("<hr style='border-color:#334155; margin:0.8rem 0;'>", unsafe_allow_html=True)
        st.markdown("<p style='color:#64748b; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.08em;'>⛏️ Mining Parameters</p>", unsafe_allow_html=True)

        min_support = st.slider("Min Support", 0.01, 0.5, 0.05, 0.01)
        min_confidence = st.slider("Min Confidence", 0.1, 1.0, 0.3, 0.05)

        st.markdown("<p style='color:#64748b; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.08em; margin-top:0.8rem;'>🗂️ Clustering</p>", unsafe_allow_html=True)
        n_clusters = st.slider("K (clusters)", 2, 8, 3, 1)

        st.markdown("<hr style='border-color:#334155; margin:0.8rem 0;'>", unsafe_allow_html=True)

        run_btn = st.button("▶ Run Pipeline", width='stretch')

        st.markdown("""
        <div style="margin-top:1rem;">
          <p style="color:#475569; font-size:0.72rem; text-align:center; line-height:1.6;">
            Johan Gejo (320)<br>
            Karthikeyan Biju (323)<br>
            Ron John Kurien (351)<br>
            Vaishnav S Kumar (366)
          </p>
        </div>
        """, unsafe_allow_html=True)

    return page, min_support, min_confidence, n_clusters, run_btn


# ─── Pages ────────────────────────────────────────────────────────────────────

def page_overview(results):
    st.markdown("""
    <div class="hero-header">
      <h1 class="hero-title">🌐 WebPathMine</h1>
      <p class="hero-subtitle">Web Usage & Navigation Pattern Mining · User Behaviour Prediction · E-Commerce Analytics</p>
    </div>
    """, unsafe_allow_html=True)

    # Top KPIs
    if results:
        df = results["shoppers_df"]
        sessions = results["sessions"]
        mining = results["mining"]
        clf = results["classification"]

        n_rules_ap = len(mining["apriori"]["rules"])
        n_rules_fp = len(mining["fpgrowth"]["rules"])
        purchase_rate = df["Revenue"].mean() * 100

        best_model = max(
            clf["results"].items(),
            key=lambda kv: kv[1]["f1"]
        )
        best_f1 = best_model[1]["f1"] * 100

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: metric_card(f"{df.shape[0]:,}", "Total Sessions", suffix="")
        with c2: metric_card(f"{purchase_rate:.1f}", "Purchase Rate", suffix="%")
        with c3: metric_card(f"{len(sessions):,}", "Click Sessions")
        with c4: metric_card(f"{max(n_rules_ap, n_rules_fp)}", "Association Rules")
        with c5: metric_card(f"{best_f1:.1f}", "Best F1 Score", suffix="%")
    else:
        st.info("⬅️  Click **Run Pipeline** in the sidebar to start analysis.")
        # Demo metrics
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: metric_card("12,330", "Total Sessions")
        with c2: metric_card("15.5", "Purchase Rate", suffix="%")
        with c3: metric_card("—", "Click Sessions")
        with c4: metric_card("—", "Association Rules")
        with c5: metric_card("—", "Best F1 Score")

    st.markdown("---")

    # Pipeline flow
    st.markdown('<div class="section-title">Pipeline Architecture</div>', unsafe_allow_html=True)

    pipeline_steps = [
        ("1", "📥 Collect", "UCI Datasets\nClickstream + Shoppers"),
        ("2", "🔧 Preprocess", "Clean · Encode\nSessionize"),
        ("3", "⚙️ Features", "Duration · PageViews\nCart · HighIntent"),
        ("4", "⛏️ Mine", "Apriori · FP-Growth\nAssociation Rules"),
        ("5", "🗂️ Cluster", "K-Means · Elbow\nUser Segments"),
        ("6", "🤖 Predict", "ID3 · Gini DT\nNeural Network"),
        ("7", "📊 Evaluate", "F1 · AUC · Lift\nConfusion Matrix"),
    ]

    cols = st.columns(7)
    for col, (num, title, desc) in zip(cols, pipeline_steps):
        with col:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#1e293b,#0f172a);
                        border:1px solid #334155; border-radius:12px;
                        padding:0.8rem 0.5rem; text-align:center;">
              <div style="background:linear-gradient(135deg,#6366f1,#8b5cf6);
                          border-radius:50%; width:2rem; height:2rem;
                          display:flex; align-items:center; justify-content:center;
                          margin:0 auto 0.4rem; font-size:0.75rem; font-weight:700;
                          color:white;">{num}</div>
              <div style="color:#e2e8f0; font-weight:700; font-size:0.8rem;">{title}</div>
              <div style="color:#64748b; font-size:0.65rem; margin-top:0.3rem;
                          white-space:pre-line; line-height:1.4;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Dataset info
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">📊 Datasets</div>', unsafe_allow_html=True)
        ds_info = pd.DataFrame([
            {"Dataset": "Online Shoppers Intention", "Rows": "12,330", "Features": "17+5",
             "Target": "Revenue (binary)", "Source": "UCI #468"},
            {"Dataset": "Clickstream Shopping", "Rows": "~165K", "Features": "6",
             "Target": "Page sequences", "Source": "UCI #553"},
        ])
        st.dataframe(ds_info, width='stretch', hide_index=True)

    with col2:
        st.markdown('<div class="section-title">🛠️ Technology Stack</div>', unsafe_allow_html=True)
        tech = {
            "Data Processing": ["pandas", "numpy"],
            "Pattern Mining": ["mlxtend (Apriori, FP-Growth)"],
            "Machine Learning": ["scikit-learn (DT, NN, KMeans)"],
            "Visualization": ["plotly", "streamlit"],
        }
        for category, tools in tech.items():
            badges = " ".join([f'<span class="pipeline-badge">{t}</span>' for t in tools])
            st.markdown(f"""
            <div style="margin-bottom:0.5rem;">
              <span style="color:#94a3b8; font-size:0.75rem; font-weight:600;">{category}</span><br>
              {badges}
            </div>""", unsafe_allow_html=True)


def page_data_explorer(results):
    st.markdown('<div class="section-title">🔍 Data Explorer</div>', unsafe_allow_html=True)

    if not results:
        st.info("⬅️  Run the pipeline first.")
        return

    df = results["shoppers_df"]
    df_raw = results["shoppers_raw_df"]

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📋 Dataset Preview", "📈 Distributions", "🔗 Correlations", "🖱️ Session Stats"]
    )

    with tab1:
        st.markdown(f"**{df.shape[0]:,} rows × {df.shape[1]} columns** — Online Shoppers Purchasing Intention")
        col1, col2, col3 = st.columns(3)
        with col1: metric_card(f"{df['Revenue'].sum():,}", "Purchase Sessions")
        with col2: metric_card(f"{df.shape[0]-df['Revenue'].sum():,}", "Non-Purchase Sessions")
        with col3: metric_card(f"{df['Revenue'].mean():.1%}", "Conversion Rate")
        st.dataframe(df.head(50), width='stretch')

    with tab2:
        col_to_plot = st.selectbox(
            "Select feature to visualize",
            ["ProductRelated", "TotalPages", "TotalDuration", "PageValues",
             "BounceRates", "ExitRates", "Avg_Duration_Per_Page"],
        )
        fig = px.histogram(
            df, x=col_to_plot, color="Revenue",
            color_discrete_map={0: "#6366f1", 1: "#22c55e"},
            barmode="overlay", opacity=0.7,
            template=PLOT_TEMPLATE,
            labels={"Revenue": "Purchased"},
            title=f"Distribution of {col_to_plot} by Purchase Outcome",
        )
        fig.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            font_color="#e2e8f0", legend_title_text="Purchased",
        )
        st.plotly_chart(fig, width='stretch')

        # Box plot
        fig2 = px.box(
            df, x="Revenue", y=col_to_plot,
            color="Revenue",
            color_discrete_map={0: "#6366f1", 1: "#22c55e"},
            template=PLOT_TEMPLATE,
            labels={"Revenue": "Purchased (0=No, 1=Yes)"},
            title=f"{col_to_plot}: Purchase vs No-Purchase",
        )
        fig2.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#0d1117", font_color="#e2e8f0")
        st.plotly_chart(fig2, width='stretch')

    with tab3:
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        corr = df[num_cols].corr()
        fig = px.imshow(
            corr,
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            template=PLOT_TEMPLATE,
            title="Feature Correlation Heatmap",
            aspect="auto",
        )
        fig.update_layout(
            paper_bgcolor="#0d1117",
            font_color="#e2e8f0",
            height=600,
        )
        st.plotly_chart(fig, width='stretch')

    with tab4:
        sessions = results["sessions"]
        lengths = [len(s) for s in sessions]
        fig = px.histogram(
            x=lengths, nbins=30,
            template=PLOT_TEMPLATE,
            title="Session Length Distribution (Clickstream)",
            labels={"x": "Pages per Session", "y": "Count"},
            color_discrete_sequence=[ACCENT],
        )
        fig.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#0d1117", font_color="#e2e8f0")
        st.plotly_chart(fig, width='stretch')

        c1, c2, c3, c4 = st.columns(4)
        with c1: metric_card(f"{np.mean(lengths):.1f}", "Avg Session Length")
        with c2: metric_card(f"{np.median(lengths):.0f}", "Median Session Length")
        with c3: metric_card(f"{np.max(lengths)}", "Max Session Length")
        with c4: metric_card(f"{len(sessions):,}", "Total Click Sessions")

        # Top pages
        all_pages = [page for sess in sessions for page in sess]
        page_counts = pd.Series(all_pages).value_counts().head(15)
        fig2 = px.bar(
            x=page_counts.values, y=page_counts.index,
            orientation="h",
            template=PLOT_TEMPLATE,
            title="Top 15 Most Visited Page Categories",
            labels={"x": "Visit Count", "y": "Page Category"},
            color=page_counts.values,
            color_continuous_scale="Viridis",
        )
        fig2.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                            font_color="#e2e8f0", showlegend=False, yaxis={"autorange": "reversed"})
        st.plotly_chart(fig2, width='stretch')


def page_pattern_mining(results, min_support, min_confidence):
    st.markdown('<div class="section-title">⛏️ Pattern Mining — Apriori vs FP-Growth</div>', unsafe_allow_html=True)

    if not results:
        st.info("⬅️  Run the pipeline first.")
        return

    mining = results["mining"]

    # Algorithm comparison strip
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card(
            len(mining["apriori"]["frequent_itemsets"]),
            "Apriori Freq. Itemsets",
        )
    with col2:
        metric_card(
            len(mining["fpgrowth"]["frequent_itemsets"]),
            "FP-Growth Freq. Itemsets",
        )
    with col3:
        metric_card(
            f"{mining['apriori']['runtime_sec']:.3f}",
            "Apriori Runtime (s)",
        )
    with col4:
        metric_card(
            f"{mining['fpgrowth']['runtime_sec']:.3f}",
            "FP-Growth Runtime (s)",
        )

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(
        ["📋 Association Rules", "📉 Support vs Confidence", "⚡ FP-Growth vs Apriori Comparison"]
    )

    with tab1:
        algo = st.radio("Algorithm", ["Apriori", "FP-Growth"], horizontal=True)
        key = "apriori" if algo == "Apriori" else "fpgrowth"
        rules_df = mining[key]["rules"]

        if rules_df.empty:
            st.warning(f"No rules found with current thresholds. Try lowering min_support ({min_support}) or min_confidence ({min_confidence}).")
        else:
            # Format for display
            from evaluator import format_rules_table
            display_df = format_rules_table(rules_df)
            st.markdown(f"**{len(display_df)} rules found** — sorted by Lift ↓")
            import matplotlib
            st.dataframe(
                display_df.style.background_gradient(subset=["Lift"], cmap=matplotlib.colormaps["plasma"]),
                width='stretch',
            )

            # Top rules as cards
            st.markdown("#### 🏆 Top 5 Rules by Lift")
            top5 = display_df.head(5)
            for _, row in top5.iterrows():
                st.markdown(f"""
                <div class="rule-card">
                  <span style="color:#e2e8f0; font-weight:600;">
                    {row['Antecedent']}
                    <span class="rule-arrow"> ⟹ </span>
                    {row['Consequent']}
                  </span><br>
                  <span class="persona-stat">
                    Support: {row['Support']:.4f} &nbsp;|&nbsp;
                    Confidence: {row['Confidence']:.4f} &nbsp;|&nbsp;
                    Lift: <strong style="color:#22c55e">{row['Lift']:.4f}</strong>
                  </span>
                </div>
                """, unsafe_allow_html=True)

    with tab2:
        for algo_key, algo_name, color in [
            ("apriori", "Apriori", "#6366f1"),
            ("fpgrowth", "FP-Growth", "#22c55e"),
        ]:
            rules = mining[algo_key]["rules"]
            if not rules.empty:
                fig = px.scatter(
                    rules,
                    x="support", y="confidence",
                    size="lift",
                    color="lift",
                    color_continuous_scale="Plasma",
                    template=PLOT_TEMPLATE,
                    title=f"{algo_name}: Support vs Confidence (bubble size = Lift)",
                    labels={"support": "Support", "confidence": "Confidence", "lift": "Lift"},
                    hover_data=["lift"],
                )
                fig.update_layout(
                    paper_bgcolor="#0d1117", plot_bgcolor="#0d1117", font_color="#e2e8f0"
                )
                st.plotly_chart(fig, width='stretch')

    with tab3:
        # Comparison bar chart
        comparison_data = pd.DataFrame({
            "Metric": ["Frequent Itemsets", "Association Rules", "Runtime (ms)"],
            "Apriori": [
                len(mining["apriori"]["frequent_itemsets"]),
                len(mining["apriori"]["rules"]),
                mining["apriori"]["runtime_sec"] * 1000,
            ],
            "FP-Growth": [
                len(mining["fpgrowth"]["frequent_itemsets"]),
                len(mining["fpgrowth"]["rules"]),
                mining["fpgrowth"]["runtime_sec"] * 1000,
            ],
        })

        fig = px.bar(
            comparison_data.melt(id_vars="Metric", var_name="Algorithm", value_name="Value"),
            x="Metric", y="Value", color="Algorithm",
            barmode="group",
            color_discrete_map={"Apriori": "#6366f1", "FP-Growth": "#22c55e"},
            template=PLOT_TEMPLATE,
            title="Apriori vs FP-Growth — Performance Comparison",
        )
        fig.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#0d1117", font_color="#e2e8f0")
        st.plotly_chart(fig, width='stretch')

        # Insight
        ap_time = mining["apriori"]["runtime_sec"]
        fp_time = mining["fpgrowth"]["runtime_sec"]
        faster = "FP-Growth" if fp_time < ap_time else "Apriori"
        speedup = max(ap_time, fp_time) / (min(ap_time, fp_time) + 1e-9)
        st.markdown(f"""
        <div class="info-box">
          <strong>⚡ Speed Insight:</strong> {faster} was {speedup:.1f}× faster on this dataset.<br>
          FP-Growth avoids candidate generation, making it more efficient on dense datasets.
          Apriori is simpler to implement but slower on large, frequent item sets.
        </div>""", unsafe_allow_html=True)

        # Frequent itemsets by length
        for algo_key, algo_name in [("apriori", "Apriori"), ("fpgrowth", "FP-Growth")]:
            fi = mining[algo_key]["frequent_itemsets"]
            if not fi.empty and "length" in fi.columns:
                length_dist = fi["length"].value_counts().sort_index()
                fig2 = px.bar(
                    x=length_dist.index, y=length_dist.values,
                    labels={"x": "Itemset Size", "y": "Count"},
                    title=f"{algo_name}: Frequent Itemsets by Size",
                    template=PLOT_TEMPLATE,
                    color=length_dist.values,
                    color_continuous_scale="Viridis",
                )
                fig2.update_layout(
                    paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                    font_color="#e2e8f0", showlegend=False,
                )
                st.plotly_chart(fig2, width='stretch')


def page_classification(results):
    st.markdown('<div class="section-title">🤖 Classification — Purchase Intent Prediction</div>', unsafe_allow_html=True)

    if not results:
        st.info("⬅️  Run the pipeline first.")
        return

    clf = results["classification"]
    clf_results = clf["results"]
    model_names = list(clf_results.keys())

    # Summary metrics
    cols = st.columns(len(model_names))
    for col, name in zip(cols, model_names):
        r = clf_results[name]
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div style="font-size:0.8rem; color:#94a3b8; text-transform:uppercase;
                          letter-spacing:0.08em; margin-bottom:0.5rem;">{name}</div>
              <div class="metric-value">{r['f1']:.4f}</div>
              <div class="metric-label">F1 Score</div>
              <div style="font-size:0.75rem; color:#64748b; margin-top:0.5rem;">
                Acc: {r['accuracy']:.4f} | AUC: {f"{r['auc']:.4f}" if r['auc'] is not None else 'N/A'}
              </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📋 Comparison Table", "📉 Confusion Matrices", "🎯 ROC Curves", "🔑 Feature Importance"]
    )

    with tab1:
        ct = results["comparison_table"]
        st.markdown("#### Model Performance Comparison")
        st.dataframe(ct.style.highlight_max(axis=0, color="#1a3a1a"), width='stretch')

        # Radar chart
        metrics = ["accuracy", "precision", "recall", "f1"]
        fig = go.Figure()
        for name in model_names:
            r = clf_results[name]
            values = [r[m] for m in metrics]
            values += values[:1]  # close the loop
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=["Accuracy", "Precision", "Recall", "F1", "Accuracy"],
                fill="toself",
                name=name,
                opacity=0.6,
            ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], color="#e2e8f0"),
                bgcolor="#1e293b",
            ),
            paper_bgcolor="#0d1117",
            font_color="#e2e8f0",
            template=PLOT_TEMPLATE,
            title="Model Performance Radar Chart",
            legend=dict(bgcolor="#1e293b", bordercolor="#334155"),
        )
        st.plotly_chart(fig, width='stretch')

    with tab2:
        cols = st.columns(len(model_names))
        for col, name in zip(cols, model_names):
            r = clf_results[name]
            cm = r["confusion_matrix"]
            with col:
                fig = px.imshow(
                    cm,
                    text_auto=True,
                    color_continuous_scale="Blues",
                    labels={"x": "Predicted", "y": "Actual", "color": "Count"},
                    x=["No Purchase", "Purchase"],
                    y=["No Purchase", "Purchase"],
                    title=f"{name}",
                    template=PLOT_TEMPLATE,
                )
                fig.update_layout(
                    paper_bgcolor="#0d1117", font_color="#e2e8f0", height=350
                )
                st.plotly_chart(fig, width='stretch')

    with tab3:
        fig = go.Figure()
        colors = ["#6366f1", "#22c55e", "#f97316"]
        for (name, r), color in zip(clf_results.items(), colors):
            if r["fpr"] is not None:
                fig.add_trace(go.Scatter(
                    x=r["fpr"], y=r["tpr"],
                    name=f"{name} (AUC={r['auc']:.4f})",
                    mode="lines",
                    line=dict(color=color, width=2.5),
                ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            name="Random Classifier",
            mode="lines",
            line=dict(color="#475569", dash="dash"),
        ))
        fig.update_layout(
            template=PLOT_TEMPLATE,
            title="ROC Curves — All Models",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            font_color="#e2e8f0",
            legend=dict(bgcolor="#1e293b", bordercolor="#334155"),
        )
        st.plotly_chart(fig, width='stretch')

    with tab4:
        for name in model_names:
            r = clf_results[name]
            feat_imp = r["feature_importance"].head(15)
            fig = px.bar(
                x=feat_imp.values,
                y=feat_imp.index,
                orientation="h",
                title=f"{name} — Top 15 Features",
                template=PLOT_TEMPLATE,
                labels={"x": "Importance", "y": "Feature"},
                color=feat_imp.values,
                color_continuous_scale="Viridis",
            )
            fig.update_layout(
                paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                font_color="#e2e8f0", showlegend=False,
                yaxis={"autorange": "reversed"},
            )
            st.plotly_chart(fig, width='stretch')


def page_clustering(results):
    st.markdown('<div class="section-title">🗂️ User Segmentation — K-Means Clustering</div>', unsafe_allow_html=True)

    if not results:
        st.info("⬅️  Run the pipeline first.")
        return

    cl = results["clustering"]

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card(cl["best_k"], "Optimal Clusters (k)")
    with c2:
        sil_score = cl["elbow"]["silhouettes"][
            cl["elbow"]["k_values"].index(cl["best_k"])
        ]
        metric_card(f"{sil_score:.4f}", "Silhouette Score")
    with c3:
        metric_card(
            f"{cl['pca_variance_ratio'][0]*100:.1f}",
            "PCA Comp-1 Variance",
            suffix="%",
        )
    with c4:
        metric_card(
            f"{sum(cl['pca_variance_ratio'])*100:.1f}",
            "PCA Total Variance",
            suffix="%",
        )

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(
        ["📉 Elbow & Silhouette", "🔵 Cluster Scatter", "👤 Cluster Profiles"]
    )

    with tab1:
        elbow = cl["elbow"]
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Elbow Method (Inertia)", "Silhouette Scores"),
        )
        fig.add_trace(go.Scatter(
            x=elbow["k_values"], y=elbow["inertias"],
            mode="lines+markers",
            line=dict(color=ACCENT, width=2.5),
            marker=dict(size=8),
            name="Inertia",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=elbow["k_values"], y=elbow["silhouettes"],
            mode="lines+markers",
            line=dict(color=GREEN, width=2.5),
            marker=dict(size=8),
            name="Silhouette",
        ), row=1, col=2)
        k_val = cl["best_k"]
        label_text = f"Optimal k={k_val}" if k_val == elbow.get("best_k") else f"Selected k={k_val}"

        # Indicator line on Elbow plot
        fig.add_shape(
            type="line",
            x0=k_val, x1=k_val,
            y0=0, y1=1,
            xref="x", yref="y domain",
            line=dict(color=ORANGE, dash="dash", width=1.5),
        )
        # Indicator line on Silhouette plot
        fig.add_shape(
            type="line",
            x0=k_val, x1=k_val,
            y0=0, y1=1,
            xref="x2", yref="y2 domain",
            line=dict(color=ORANGE, dash="dash", width=1.5),
        )
        # Badge annotation placed inside the plot canvas, safely below the subplot title
        fig.add_annotation(
            x=k_val, y=0.92,
            xref="x2", yref="y2 domain",
            text=label_text,
            showarrow=False,
            font=dict(color=ORANGE, size=11),
            bgcolor="rgba(15, 23, 42, 0.85)",
            bordercolor=ORANGE,
            borderwidth=1,
            borderpad=4,
            yanchor="top",
        )
        fig.update_layout(
            template=PLOT_TEMPLATE,
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            font_color="#e2e8f0", height=420,
            showlegend=False,
            margin=dict(t=55, b=40, l=40, r=40),
        )
        st.plotly_chart(fig, width='stretch')

    with tab2:
        labels = cl["labels"]
        pca_coords = cl["pca_coords"]
        profiles = cl["profiles"]

        # Build persona label map
        persona_map = dict(zip(profiles["Cluster"], profiles["Persona"]))
        persona_labels = [persona_map.get(l, f"Cluster {l}") for l in labels]

        scatter_df = pd.DataFrame({
            "PCA-1": pca_coords[:, 0],
            "PCA-2": pca_coords[:, 1],
            "Cluster": [str(l) for l in labels],
            "Persona": persona_labels,
        })

        fig = px.scatter(
            scatter_df,
            x="PCA-1", y="PCA-2",
            color="Persona",
            template=PLOT_TEMPLATE,
            title=f"Session Clusters (k={cl['best_k']}) — PCA Projection",
            color_discrete_sequence=COLOR_SEQ,
            opacity=0.6,
            hover_data=["Cluster", "Persona"],
        )
        # Add centroids
        km = cl["kmeans"]  # noqa: F841  (available for future centroid overlay)
        # We can't re-project without the original data, so just label visually
        fig.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            font_color="#e2e8f0", height=550,
        )
        st.plotly_chart(fig, width='stretch')

    with tab3:
        profiles = cl["profiles"]
        st.markdown("#### Cluster Persona Profiles")

        # Cards
        for _, row in profiles.iterrows():
            cluster_id = row["Cluster"]
            persona = row["Persona"]
            count = int(row["Count"])
            pct = row["Pct"]

            feature_stats = []
            for feat in ["TotalPages", "TotalDuration", "PageValues", "BounceRates"]:
                if feat in row.index:
                    feature_stats.append(f"{feat}: {row[feat]:.3f}")

            st.markdown(f"""
            <div class="persona-card">
              <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                  <span class="persona-title">Cluster {cluster_id} · {persona}</span><br>
                  <span class="persona-stat">{', '.join(feature_stats)}</span>
                </div>
                <div style="text-align:right;">
                  <div class="metric-value" style="font-size:1.5rem;">{count:,}</div>
                  <div class="metric-label">{pct}% of sessions</div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

        # Heatmap of cluster feature profiles
        profile_num = profiles.select_dtypes(include=np.number).drop(
            columns=["Cluster", "Count", "Pct"], errors="ignore"
        )
        if not profile_num.empty:
            fig = px.imshow(
                profile_num.T,
                color_continuous_scale="Plasma",
                template=PLOT_TEMPLATE,
                title="Cluster Feature Heatmap (mean values)",
                labels={"x": "Cluster", "y": "Feature", "color": "Value"},
                aspect="auto",
            )
            fig.update_layout(
                paper_bgcolor="#0d1117", font_color="#e2e8f0",
                xaxis=dict(tickvals=list(range(len(profiles))),
                           ticktext=[str(i) for i in profiles["Cluster"]]),
            )
            st.plotly_chart(fig, width='stretch')


def page_evaluation(results):
    st.markdown('<div class="section-title">📊 Full Evaluation Report</div>', unsafe_allow_html=True)

    if not results:
        st.info("⬅️  Run the pipeline first.")
        return

    clf = results["classification"]
    mining = results["mining"]

    # Cross-validation results
    st.markdown("#### 🔄 5-Fold Cross-Validation Results")
    cv_data = []
    for name, r in clf["results"].items():
        cv_data.append({
            "Model": name,
            "CV Mean Accuracy": f"{r['cv_mean']:.4f}",
            "CV Std": f"± {r['cv_std']:.4f}",
        })
    st.dataframe(pd.DataFrame(cv_data).set_index("Model"), width='stretch')

    st.markdown("---")
    st.markdown("#### 📐 Association Rule Quality Metrics")

    for algo_key, algo_name in [("apriori", "Apriori"), ("fpgrowth", "FP-Growth")]:
        rules = mining[algo_key]["rules"]
        if not rules.empty:
            col1, col2, col3, col4 = st.columns(4)
            with col1: metric_card(f"{rules['support'].mean():.4f}", f"{algo_name} Avg Support")
            with col2: metric_card(f"{rules['confidence'].mean():.4f}", f"{algo_name} Avg Confidence")
            with col3: metric_card(f"{rules['lift'].mean():.4f}", f"{algo_name} Avg Lift")
            with col4: metric_card(len(rules), f"{algo_name} Total Rules")

    st.markdown("---")
    st.markdown("#### 📊 Metric Comparison — Bar Chart")

    metric_names = ["accuracy", "precision", "recall", "f1"]
    bar_data = []
    for name, r in clf["results"].items():
        for m in metric_names:
            bar_data.append({
                "Model": name,
                "Metric": m.capitalize(),
                "Score": r[m],
            })

    fig = px.bar(
        pd.DataFrame(bar_data),
        x="Metric", y="Score", color="Model", barmode="group",
        template=PLOT_TEMPLATE,
        title="Classification Metrics Comparison",
        color_discrete_sequence=["#6366f1", "#22c55e", "#f97316"],
        range_y=[0, 1],
    )
    fig.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117", font_color="#e2e8f0",
        legend=dict(bgcolor="#1e293b", bordercolor="#334155"),
    )
    st.plotly_chart(fig, width='stretch')

    # AUC comparison
    auc_data = [
        {"Model": name, "AUC-ROC": r["auc"]}
        for name, r in clf["results"].items()
        if r["auc"] is not None
    ]
    if auc_data:
        fig2 = px.bar(
            pd.DataFrame(auc_data),
            x="Model", y="AUC-ROC",
            template=PLOT_TEMPLATE,
            title="AUC-ROC Comparison",
            color="AUC-ROC",
            color_continuous_scale="Plasma",
            range_y=[0, 1],
        )
        fig2.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            font_color="#e2e8f0", showlegend=False,
        )
        st.plotly_chart(fig2, width='stretch')

    st.markdown("---")
    st.markdown("#### 💾 Download Results")
    ct = results["comparison_table"].reset_index()
    csv = ct.to_csv(index=False)
    st.download_button(
        "📥 Download Model Comparison CSV",
        data=csv,
        file_name="webpathmine_model_comparison.csv",
        mime="text/csv",
    )


# ─── Main App ─────────────────────────────────────────────────────────────────

def main():
    page, min_support, min_confidence, n_clusters, run_btn = render_sidebar()

    # Load or run
    if "results" not in st.session_state:
        st.session_state.results = load_cached_results()

    if run_btn:
        try:
            st.session_state.results = run_pipeline_inline(
                min_support=min_support,
                min_confidence=min_confidence,
                n_clusters_override=n_clusters,
            )
            st.success("✅ Pipeline complete! Explore results in the tabs above.")
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            import traceback
            st.code(traceback.format_exc())

    results = st.session_state.get("results")

    if page == "🏠 Overview":
        page_overview(results)
    elif page == "🔍 Data Explorer":
        page_data_explorer(results)
    elif page == "⛏️ Pattern Mining":
        page_pattern_mining(results, min_support, min_confidence)
    elif page == "🤖 Classification":
        page_classification(results)
    elif page == "🗂️ Clustering":
        page_clustering(results)
    elif page == "📊 Evaluation":
        page_evaluation(results)


if __name__ == "__main__":
    main()
