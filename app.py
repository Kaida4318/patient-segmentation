import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PatientIQ — Cluster Analysis",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Global styles ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
    background-color: #050D1A !important;
    color: #C8D8E8 !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { 
    padding: 0 2.5rem 4rem 2.5rem !important; 
    max-width: 1300px !important;
}

/* ── HERO ── */
.hero-wrap {
    position: relative;
    padding: 4rem 0 3rem 0;
    margin-bottom: 1rem;
    overflow: hidden;
}
.hero-wrap::before {
    content: '';
    position: absolute;
    top: -120px; right: -200px;
    width: 600px; height: 600px;
    background: radial-gradient(circle, rgba(0,212,255,0.07) 0%, transparent 65%);
    border-radius: 50%;
    pointer-events: none;
}
.hero-wrap::after {
    content: '';
    position: absolute;
    bottom: -80px; left: 30%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(0,229,160,0.04) 0%, transparent 65%);
    border-radius: 50%;
    pointer-events: none;
}
.hero-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    color: #00D4FF;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.8rem;
}
.hero-eyebrow::before {
    content: '';
    display: inline-block;
    width: 28px; height: 1px;
    background: #00D4FF;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 4.2rem;
    font-weight: 700;
    color: #FFFFFF;
    line-height: 1.05;
    letter-spacing: -0.02em;
    margin-bottom: 1rem;
}
.hero-title .accent { 
    color: #00D4FF;
    position: relative;
}
.hero-subtitle {
    font-size: 1.05rem;
    color: #6A8AA0;
    max-width: 540px;
    line-height: 1.7;
    font-weight: 400;
}
.hero-stat {
    position: absolute;
    right: 0; top: 50%;
    transform: translateY(-50%);
    text-align: right;
}
.hero-stat-num {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 7rem;
    font-weight: 700;
    color: rgba(0,212,255,0.08);
    line-height: 1;
    letter-spacing: -0.04em;
}

/* ── METRIC STRIP ── */
.metric-strip {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1px;
    background: #0D1E30;
    border: 1px solid #0D1E30;
    border-radius: 14px;
    overflow: hidden;
    margin-bottom: 2.5rem;
}
.metric-item {
    background: #060F1D;
    padding: 1.4rem 1.8rem;
    transition: background 0.2s;
}
.metric-item:hover { background: #091522; }
.metric-lbl {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.16em;
    color: #2A4A6A;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.metric-num {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.1rem;
    font-weight: 700;
    color: #FFFFFF;
    line-height: 1;
}
.metric-num.cyan { color: #00D4FF; }
.metric-sub { 
    font-size: 0.75rem; 
    color: #2A4A6A; 
    margin-top: 0.35rem; 
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    gap: 0 !important;
    border-bottom: 1px solid #0D1E30 !important;
    padding-bottom: 0 !important;
    margin-bottom: 2.5rem !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #3A5A7A !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    padding: 0.8rem 1.8rem !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    transition: all 0.2s !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #8AAAC0 !important;
}
.stTabs [aria-selected="true"] {
    color: #00D4FF !important;
    border-bottom: 2px solid #00D4FF !important;
    background: transparent !important;
}

/* ── CLUSTER CARDS ── */
.ccard {
    border-radius: 16px;
    padding: 2rem;
    height: 100%;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s;
}
.ccard:hover { transform: translateY(-3px); }

/* ── SECTION LABELS ── */
.sec-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #2A4A6A;
    margin-bottom: 0.4rem;
    margin-top: 2rem;
}
.sec-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.3rem;
    font-weight: 600;
    color: #FFFFFF;
    margin-bottom: 0.5rem;
}
.sec-desc {
    font-size: 0.85rem;
    color: #4A6A80;
    margin-bottom: 1.4rem;
    line-height: 1.6;
}

/* ── INSIGHT BOX ── */
.ibox {
    background: #070F1C;
    border-left: 2px solid #00D4FF;
    border-radius: 0 10px 10px 0;
    padding: 1.1rem 1.5rem;
    margin: 1.2rem 0;
    font-size: 0.87rem;
    color: #6A8AA0;
    line-height: 1.65;
}
.ibox b { color: #00D4FF; font-weight: 600; }

/* ── PREDICTOR RESULT ── */
.pred-card {
    border-radius: 16px;
    padding: 2.2rem;
    text-align: center;
    margin-bottom: 1.5rem;
}
.pred-cluster-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}
.pred-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.9rem;
    font-weight: 700;
    color: #FFFFFF;
    margin-bottom: 0.5rem;
}
.pred-rate {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3.2rem;
    font-weight: 700;
    line-height: 1;
    margin: 0.5rem 0;
}
.pred-rate-sub {
    font-size: 0.8rem;
    color: #4A6A80;
    margin-bottom: 1rem;
}
.pred-desc {
    font-size: 0.88rem;
    color: #6A8AA0;
    line-height: 1.65;
}

/* ── INPUT SUMMARY TABLE ── */
.sum-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.55rem 0;
    border-bottom: 1px solid #0A1828;
}
.sum-key {
    font-size: 0.82rem;
    color: #3A5A7A;
}
.sum-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    font-weight: 500;
    color: #C8D8E8;
}

/* ── SLIDER OVERRIDES ── */
.slider-lbl {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #2A4A6A;
    margin-top: 0.8rem;
    margin-bottom: 0.1rem;
}

/* ── DIVIDER ── */
.hdiv {
    height: 1px;
    background: linear-gradient(90deg, #00D4FF22, #0D1E30 80%);
    margin: 2rem 0;
}

/* Streamlit slider track color */
.stSlider [data-baseweb="slider"] [role="slider"] {
    background: #00D4FF !important;
}
</style>
""", unsafe_allow_html=True)


# ── Load artifacts ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open('patient_segmentation_artifacts.pkl', 'rb') as f:
        artifacts = pickle.load(f)
    km       = artifacts['kmeans']
    scaler   = artifacts['scaler']
    pca      = artifacts['pca']
    df_clean = artifacts['df_clean']
    X_scaled = artifacts['X_scaled']
    X        = df_clean.drop('Outcome', axis=1)
    y        = df_clean['Outcome']
    X_pca    = pca.transform(X_scaled)
    return df_clean, X_scaled, X, y, scaler, km, pca, X_pca

df_clean, X_scaled, X, y, scaler, km, pca, X_pca = load_artifacts()

# ── Computed values ────────────────────────────────────────────────────────────
diabetes_rate  = df_clean.groupby('Cluster')['Outcome'].mean()
cluster_counts = df_clean['Cluster'].value_counts().sort_index()
overall_rate   = df_clean['Outcome'].mean()
sil            = silhouette_score(X_scaled, km.labels_)
pca_var        = pca.explained_variance_ratio_

CLUSTER_META = {
    0: {
        "name": "Age-Driven Risk",
        "color": "#FF6B6B",
        "bg": "linear-gradient(135deg, #1A0A0A 0%, #120808 100%)",
        "border": "#FF6B6B",
        "icon": "⏳",
        "desc": "Oldest cohort (avg 46 yrs) with the highest pregnancy count. Risk is driven by age and reproductive history rather than metabolic markers alone.",
        "key_feature": "Age + Pregnancies",
    },
    1: {
        "name": "Low Risk",
        "color": "#00E5A0",
        "bg": "linear-gradient(135deg, #071A10 0%, #050F0A 100%)",
        "border": "#00E5A0",
        "icon": "✦",
        "desc": "Youngest cohort (avg 26 yrs) with the healthiest metabolic profile across all indicators. Lowest diabetes prevalence in the dataset.",
        "key_feature": "Youth + Low Glucose",
    },
    2: {
        "name": "Metabolic Risk",
        "color": "#FFB347",
        "bg": "linear-gradient(135deg, #140F02 0%, #0D0A02 100%)",
        "border": "#FFB347",
        "icon": "⚡",
        "desc": "Relatively young patients (avg 29 yrs) but with the highest glucose, insulin, BMI, and pedigree scores. Risk is entirely metabolic.",
        "key_feature": "Glucose + BMI + Insulin",
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="hero-wrap">
    <div class="hero-stat"><div class="hero-stat-num">768</div></div>
    <div class="hero-eyebrow">Unsupervised Machine Learning &nbsp;·&nbsp; Healthcare Analytics</div>
    <div class="hero-title">Patient<span class="accent">IQ</span></div>
    <div class="hero-subtitle">
        Three clinically distinct risk profiles — discovered from 8 physiological indicators
        with zero knowledge of diagnosis outcomes.
    </div>
</div>
""", unsafe_allow_html=True)

# ── Metric strip ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="metric-strip">
    <div class="metric-item">
        <div class="metric-lbl">Patients</div>
        <div class="metric-num">768</div>
        <div class="metric-sub">Pima Indians Diabetes DB</div>
    </div>
    <div class="metric-item">
        <div class="metric-lbl">Features</div>
        <div class="metric-num">8</div>
        <div class="metric-sub">Clinical indicators</div>
    </div>
    <div class="metric-item">
        <div class="metric-lbl">Clusters (K)</div>
        <div class="metric-num cyan">3</div>
        <div class="metric-sub">Elbow + Silhouette method</div>
    </div>
    <div class="metric-item">
        <div class="metric-lbl">Silhouette Score</div>
        <div class="metric-num cyan">{sil:.3f}</div>
        <div class="metric-sub">Cluster separation quality</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["  Overview  ", "  Cluster Insights  ", "  Patient Predictor  "])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
with tab1:

    st.markdown("""
    <div class="ibox">
        <b>Core finding —</b> An unsupervised algorithm, with zero access to diagnosis labels,
        separated a low-risk population (13% diabetes rate) from two high-risk groups (~52% each),
        each driven by entirely different clinical pathways. This is the power of unsupervised learning:
        it finds what you didn't tell it to look for.
    </div>
    """, unsafe_allow_html=True)

    # ── Cluster cards ─────────────────────────────────────────────────────────
    st.markdown('<div class="sec-label">Discovered Profiles</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Three Patient Risk Groups</div>', unsafe_allow_html=True)

    col0, col1, col2 = st.columns(3, gap="medium")

    for col, c in zip([col0, col1, col2], [0, 1, 2]):
        m = CLUSTER_META[c]
        rate = diabetes_rate[c] * 100
        n = cluster_counts[c]
        with col:
            st.markdown(f"""
            <div style="background:{m['bg']};border:1px solid {m['color']}22;
                        border-top:2px solid {m['color']};border-radius:16px;
                        padding:2rem 1.8rem;height:100%;position:relative;">
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;
                             letter-spacing:0.18em;text-transform:uppercase;
                             color:{m['color']}88;margin-bottom:1rem;">
                    Cluster {c} &nbsp;·&nbsp; {m['key_feature']}
                </div>
                <div style="font-size:2rem;margin-bottom:0.3rem;">{m['icon']}</div>
                <div style="font-family:'Space Grotesk',sans-serif;font-size:1.4rem;
                             font-weight:700;color:#FFFFFF;margin-bottom:1.2rem;">
                    {m['name']}
                </div>
                <div style="font-family:'Space Grotesk',sans-serif;font-size:3rem;
                             font-weight:700;color:{m['color']};line-height:1;
                             margin-bottom:0.3rem;">
                    {rate:.1f}<span style="font-size:1.2rem;color:{m['color']}88;">%</span>
                </div>
                <div style="font-size:0.75rem;color:#2A4A6A;margin-bottom:1.2rem;
                             font-family:'JetBrains Mono',monospace;">
                    diabetes prevalence
                </div>
                <div style="font-size:0.83rem;color:#4A6A80;line-height:1.65;
                             margin-bottom:1.2rem;">
                    {m['desc']}
                </div>
                <div style="font-size:0.7rem;color:#2A4A6A;
                             font-family:'JetBrains Mono',monospace;
                             border-top:1px solid {m['color']}18;padding-top:0.8rem;">
                    {n} patients &nbsp;·&nbsp; {n/768*100:.0f}% of cohort
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="hdiv"></div>', unsafe_allow_html=True)

    # ── Methodology ───────────────────────────────────────────────────────────
    st.markdown('<div class="sec-label">Methodology</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">How the Clusters Were Found</div>', unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3, gap="medium")
    with m1:
        st.markdown("""
        <div style="background:#060F1D;border:1px solid #0D1E30;border-radius:12px;padding:1.6rem;">
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;
                         letter-spacing:0.16em;text-transform:uppercase;
                         color:#00D4FF;margin-bottom:0.8rem;">Step 01</div>
            <div style="font-family:'Space Grotesk',sans-serif;font-weight:600;
                         color:#FFFFFF;margin-bottom:0.8rem;">Data Cleaning</div>
            <div style="font-size:0.83rem;color:#4A6A80;line-height:1.65;">
                Five features contained biologically impossible zeros masking missing values.
                Replaced with column medians — robust to the heavy skew in Insulin and Skin Thickness.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown("""
        <div style="background:#060F1D;border:1px solid #0D1E30;border-radius:12px;padding:1.6rem;">
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;
                         letter-spacing:0.16em;text-transform:uppercase;
                         color:#00D4FF;margin-bottom:0.8rem;">Step 02</div>
            <div style="font-family:'Space Grotesk',sans-serif;font-weight:600;
                         color:#FFFFFF;margin-bottom:0.8rem;">Feature Scaling</div>
            <div style="font-size:0.83rem;color:#4A6A80;line-height:1.65;">
                StandardScaler applied across all 8 features. K-Means uses Euclidean distance —
                unscaled features like Insulin (0–846) would dominate the calculation entirely.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown("""
        <div style="background:#060F1D;border:1px solid #0D1E30;border-radius:12px;padding:1.6rem;">
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;
                         letter-spacing:0.16em;text-transform:uppercase;
                         color:#00D4FF;margin-bottom:0.8rem;">Step 03</div>
            <div style="font-family:'Space Grotesk',sans-serif;font-weight:600;
                         color:#FFFFFF;margin-bottom:0.8rem;">Choosing K = 3</div>
            <div style="font-size:0.83rem;color:#4A6A80;line-height:1.65;">
                Elbow method: curve flattens after K=4. Silhouette peaks at K=2, holds at K=3.
                Domain reasoning settled the tie — three profiles are clinically actionable.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — CLUSTER INSIGHTS
# ─────────────────────────────────────────────────────────────────────────────
with tab2:

    left, right = st.columns([1.15, 1], gap="large")

    # ── PCA scatter ───────────────────────────────────────────────────────────
    with left:
        st.markdown('<div class="sec-label">Dimensionality Reduction</div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">Clusters in 2D Space (PCA)</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="sec-desc">8 features compressed into 2 components capturing {sum(pca_var)*100:.1f}% of total variance. Separation is stronger in the original 8D space.</div>', unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        fig.patch.set_facecolor('#060F1D')
        ax.set_facecolor('#040C18')

        colors_map = {0: '#FF6B6B', 1: '#00E5A0', 2: '#FFB347'}
        labels_map = {0: 'Age-Driven Risk', 1: 'Low Risk', 2: 'Metabolic Risk'}

        pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
        pca_df['Cluster'] = km.labels_

        for c in [0, 1, 2]:
            mask = pca_df['Cluster'] == c
            ax.scatter(pca_df.loc[mask, 'PC1'], pca_df.loc[mask, 'PC2'],
                       c=colors_map[c], label=labels_map[c],
                       alpha=0.55, edgecolors='none', s=22, zorder=3)

        ax.set_xlabel(f'PC1  —  {pca_var[0]*100:.1f}% variance', color='#1E3A54', fontsize=8.5, labelpad=10)
        ax.set_ylabel(f'PC2  —  {pca_var[1]*100:.1f}% variance', color='#1E3A54', fontsize=8.5, labelpad=10)
        ax.tick_params(colors='#1E3A54', labelsize=7.5)
        for spine in ax.spines.values():
            spine.set_edgecolor('#0A1828')

        legend = ax.legend(frameon=True, fontsize=8.5, facecolor='#060F1D',
                           edgecolor='#0D1E30', labelcolor='#8AAAC0',
                           loc='upper right', markerscale=1.5)
        ax.grid(True, alpha=0.05, color='#1E3A54', linewidth=0.5)
        fig.tight_layout(pad=1.5)
        st.pyplot(fig)
        plt.close()

    # ── Diabetes prevalence bar ───────────────────────────────────────────────
    with right:
        st.markdown('<div class="sec-label">Validation Against Labels</div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">Diabetes Prevalence by Cluster</div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-desc">Diagnosis labels were never seen during training. This is purely post-hoc validation.</div>', unsafe_allow_html=True)

        fig2, ax2 = plt.subplots(figsize=(5.5, 5.5))
        fig2.patch.set_facecolor('#060F1D')
        ax2.set_facecolor('#040C18')

        rates   = [diabetes_rate[c] * 100 for c in [0, 1, 2]]
        blabels = ['Age-Driven\nRisk', 'Low\nRisk', 'Metabolic\nRisk']
        bcolors = ['#FF6B6B', '#00E5A0', '#FFB347']

        bars = ax2.bar(blabels, rates, color=bcolors, width=0.52,
                       edgecolor='none', zorder=3)

        for bar, rate in zip(bars, rates):
            ax2.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 1.5,
                     f'{rate:.1f}%',
                     ha='center', va='bottom',
                     fontsize=12, fontweight='700', color='#FFFFFF')

        ax2.axhline(overall_rate * 100, color='#00D4FF', linewidth=1,
                    linestyle='--', alpha=0.4, zorder=2)
        ax2.text(2.42, overall_rate * 100 + 1.2,
                 f'Dataset avg  {overall_rate*100:.1f}%',
                 color='#00D4FF', fontsize=7, ha='right', alpha=0.7)

        ax2.set_ylim(0, 75)
        ax2.set_ylabel('Diabetes Rate (%)', color='#1E3A54', fontsize=8.5, labelpad=10)
        ax2.tick_params(colors='#3A5A7A', labelsize=9)
        for spine in ax2.spines.values():
            spine.set_edgecolor('#0A1828')
        ax2.grid(axis='y', alpha=0.05, color='#1E3A54', linewidth=0.5, zorder=1)
        fig2.tight_layout(pad=1.5)
        st.pyplot(fig2)
        plt.close()

    st.markdown('<div class="hdiv"></div>', unsafe_allow_html=True)

    # ── Normalised feature profile ────────────────────────────────────────────
    st.markdown('<div class="sec-label">Feature Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Clinical Profile per Cluster (Normalised)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-desc">Mean feature values normalised 0–1 across clusters. Reveals the dominant risk driver for each group.</div>', unsafe_allow_html=True)

    profiles      = df_clean.groupby('Cluster')[X.columns.tolist()].mean().round(1)
    profiles_norm = profiles.copy()
    for col in X.columns:
        cmin, cmax = profiles[col].min(), profiles[col].max()
        profiles_norm[col] = (profiles[col] - cmin) / (cmax - cmin) if cmax > cmin else 0.5

    fig3, ax3 = plt.subplots(figsize=(12, 4.2))
    fig3.patch.set_facecolor('#060F1D')
    ax3.set_facecolor('#040C18')

    features = X.columns.tolist()
    x        = np.arange(len(features))
    width    = 0.26
    bcolors3 = ['#FF6B6B', '#00E5A0', '#FFB347']
    bnames   = ['Age-Driven Risk', 'Low Risk', 'Metabolic Risk']

    for i, c in enumerate([0, 1, 2]):
        vals = [profiles_norm.loc[c, f] for f in features]
        ax3.bar(x + i * width, vals, width, label=bnames[i],
                color=bcolors3[i], alpha=0.8, edgecolor='none', zorder=3)

    ax3.set_xticks(x + width)
    ax3.set_xticklabels([f.replace('_', '\n') for f in features],
                         fontsize=8, color='#3A5A7A')
    ax3.set_ylabel('Relative Value (0–1)', color='#1E3A54', fontsize=8.5, labelpad=10)
    ax3.tick_params(colors='#1E3A54', labelsize=8)
    for spine in ax3.spines.values():
        spine.set_edgecolor('#0A1828')
    ax3.grid(axis='y', alpha=0.05, color='#1E3A54', linewidth=0.5, zorder=1)
    ax3.legend(frameon=True, facecolor='#060F1D', edgecolor='#0D1E30',
               labelcolor='#8AAAC0', fontsize=8.5)
    fig3.tight_layout(pad=1.5)
    st.pyplot(fig3)
    plt.close()

    st.markdown("""
    <div class="ibox">
        <b>Key insight —</b> Clusters 0 and 2 share nearly identical diabetes rates (~52%)
        but are clinically opposite. Cluster 0's risk is driven by age and pregnancy history;
        Cluster 2's risk is entirely metabolic — high glucose, insulin, and BMI despite being younger.
        A classification model would not surface this distinction. This is what unsupervised
        learning uniquely contributes.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — PATIENT PREDICTOR
# ─────────────────────────────────────────────────────────────────────────────
with tab3:

    st.markdown('<div class="sec-label">Live Inference</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">New Patient Risk Assessment</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-desc">Adjust clinical measurements below. The model assigns the patient to the closest cluster in real time.</div>', unsafe_allow_html=True)

    inp_col, res_col = st.columns([1, 1], gap="large")

    with inp_col:
        st.markdown('<div style="background:#060F1D;border:1px solid #0D1E30;border-radius:14px;padding:1.8rem 2rem;">', unsafe_allow_html=True)

        st.markdown('<div class="slider-lbl">Pregnancies</div>', unsafe_allow_html=True)
        pregnancies = st.slider('Pregnancies', 0, 17, 3, label_visibility='collapsed')

        st.markdown('<div class="slider-lbl">Glucose (mg/dL)</div>', unsafe_allow_html=True)
        glucose = st.slider('Glucose', 44, 199, 120, label_visibility='collapsed')

        st.markdown('<div class="slider-lbl">Blood Pressure (mmHg)</div>', unsafe_allow_html=True)
        blood_pressure = st.slider('Blood Pressure', 24, 122, 70, label_visibility='collapsed')

        st.markdown('<div class="slider-lbl">Skin Thickness (mm)</div>', unsafe_allow_html=True)
        skin_thickness = st.slider('Skin Thickness', 7, 99, 23, label_visibility='collapsed')

        st.markdown('<div class="slider-lbl">Insulin (μU/mL)</div>', unsafe_allow_html=True)
        insulin = st.slider('Insulin', 14, 846, 80, label_visibility='collapsed')

        st.markdown('<div class="slider-lbl">BMI (kg/m²)</div>', unsafe_allow_html=True)
        bmi = st.slider('BMI', 18.0, 67.1, 32.0, step=0.1, label_visibility='collapsed')

        st.markdown('<div class="slider-lbl">Diabetes Pedigree Function</div>', unsafe_allow_html=True)
        dpf = st.slider('DPF', 0.078, 2.420, 0.47, step=0.001, label_visibility='collapsed')

        st.markdown('<div class="slider-lbl">Age (years)</div>', unsafe_allow_html=True)
        age = st.slider('Age', 21, 81, 30, label_visibility='collapsed')

        st.markdown('</div>', unsafe_allow_html=True)

    with res_col:
        # predict
        input_arr    = np.array([[pregnancies, glucose, blood_pressure,
                                  skin_thickness, insulin, bmi, dpf, age]])
        input_scaled = scaler.transform(input_arr)
        pred_cluster = int(km.predict(input_scaled)[0])
        m            = CLUSTER_META[pred_cluster]
        rate         = diabetes_rate[pred_cluster] * 100

        st.markdown(f"""
        <div style="background:{m['bg']};border:1px solid {m['color']}33;
                     border-top:3px solid {m['color']};border-radius:16px;
                     padding:2.2rem;text-align:center;margin-bottom:1.2rem;">
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.62rem;
                         letter-spacing:0.2em;text-transform:uppercase;
                         color:{m['color']}88;margin-bottom:0.6rem;">
                Cluster {pred_cluster} &nbsp;·&nbsp; Assigned Profile
            </div>
            <div style="font-size:2.2rem;margin:0.5rem 0;">{m['icon']}</div>
            <div style="font-family:'Space Grotesk',sans-serif;font-size:1.8rem;
                         font-weight:700;color:#FFFFFF;margin-bottom:0.8rem;">
                {m['name']}
            </div>
            <div style="font-family:'Space Grotesk',sans-serif;font-size:3.8rem;
                         font-weight:700;color:{m['color']};line-height:1;">
                {rate:.1f}<span style="font-size:1.4rem;color:{m['color']}77;">%</span>
            </div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                         color:#2A4A6A;margin:0.4rem 0 1.2rem;">
                diabetes prevalence in this group
            </div>
            <div style="font-size:0.85rem;color:#4A6A80;line-height:1.65;
                         max-width:360px;margin:0 auto;">
                {m['desc']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Input summary
        st.markdown("""
        <div style="background:#060F1D;border:1px solid #0D1E30;
                     border-radius:12px;padding:1.4rem 1.6rem;">
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;
                         letter-spacing:0.16em;text-transform:uppercase;
                         color:#2A4A6A;margin-bottom:1rem;">Input Summary</div>
        """, unsafe_allow_html=True)

        rows = [
            ("Pregnancies",     pregnancies),
            ("Glucose",         f"{glucose} mg/dL"),
            ("Blood Pressure",  f"{blood_pressure} mmHg"),
            ("Skin Thickness",  f"{skin_thickness} mm"),
            ("Insulin",         f"{insulin} μU/mL"),
            ("BMI",             f"{bmi:.1f} kg/m²"),
            ("Pedigree (DPF)",  f"{dpf:.3f}"),
            ("Age",             f"{age} yrs"),
        ]
        rows_html = ""
        for k, v in rows:
            rows_html += f"""
            <div class="sum-row">
                <span class="sum-key">{k}</span>
                <span class="sum-val">{v}</span>
            </div>"""

        st.markdown(rows_html + "</div>", unsafe_allow_html=True)

