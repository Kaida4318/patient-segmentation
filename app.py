import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pickle
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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #080C14;
    color: #E8EDF5;
}

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem 3rem; max-width: 1200px; }

/* ── Hero header ── */
.hero {
    background: linear-gradient(135deg, #0D1B2A 0%, #0F2A4A 50%, #091E35 100%);
    border: 1px solid #1E3A5F;
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(0,180,216,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.18em;
    color: #00B4D8;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 700;
    color: #FFFFFF;
    line-height: 1.15;
    margin-bottom: 0.5rem;
}
.hero-title span { color: #00B4D8; }
.hero-sub {
    font-size: 1rem;
    color: #8899AA;
    font-weight: 400;
    max-width: 560px;
    line-height: 1.6;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0D1520;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #1E3A5F;
    margin-bottom: 2rem;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #7A8FA6;
    font-weight: 500;
    font-size: 0.9rem;
    padding: 0.55rem 1.4rem;
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    background: #00B4D8 !important;
    color: #FFFFFF !important;
    font-weight: 600;
}

/* ── Metric cards ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin: 1.5rem 0;
}
.metric-card {
    background: #0D1520;
    border: 1px solid #1E3A5F;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #00B4D8; }
.metric-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    color: #5A7A9A;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #FFFFFF;
    line-height: 1;
}
.metric-sub {
    font-size: 0.78rem;
    color: #5A7A9A;
    margin-top: 0.3rem;
}

/* ── Cluster cards ── */
.cluster-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1.2rem;
    margin: 1.5rem 0;
}
.cluster-card {
    border-radius: 14px;
    padding: 1.6rem;
    border: 1px solid transparent;
    position: relative;
    overflow: hidden;
}
.cluster-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 3px;
}
.cluster-0 { background: #1A1018; border-color: #E74C3C; }
.cluster-0::after { background: #E74C3C; }
.cluster-1 { background: #0E1A12; border-color: #2ECC71; }
.cluster-1::after { background: #2ECC71; }
.cluster-2 { background: #0E1525; border-color: #3498DB; }
.cluster-2::after { background: #3498DB; }
.cluster-badge {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    padding: 0.25rem 0.6rem;
    border-radius: 4px;
    display: inline-block;
    margin-bottom: 0.8rem;
}
.badge-0 { background: rgba(231,76,60,0.15); color: #E74C3C; }
.badge-1 { background: rgba(46,204,113,0.15); color: #2ECC71; }
.badge-2 { background: rgba(52,152,219,0.15); color: #3498DB; }
.cluster-name {
    font-size: 1.15rem;
    font-weight: 700;
    color: #FFFFFF;
    margin-bottom: 0.4rem;
}
.cluster-rate {
    font-size: 2.2rem;
    font-weight: 700;
    margin: 0.5rem 0 0.3rem;
}
.rate-0 { color: #E74C3C; }
.rate-1 { color: #2ECC71; }
.rate-2 { color: #3498DB; }
.cluster-desc {
    font-size: 0.82rem;
    color: #7A8FA6;
    line-height: 1.55;
    margin-top: 0.6rem;
}
.cluster-stat {
    font-size: 0.78rem;
    color: #5A7A9A;
    margin-top: 0.8rem;
    font-family: 'DM Mono', monospace;
}

/* ── Section headers ── */
.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #FFFFFF;
    margin: 2rem 0 0.4rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-sub {
    font-size: 0.85rem;
    color: #5A7A9A;
    margin-bottom: 1.2rem;
    line-height: 1.5;
}

/* ── Predictor result ── */
.result-card {
    border-radius: 14px;
    padding: 2rem 2.4rem;
    border: 1px solid;
    margin-top: 1.5rem;
    text-align: center;
}
.result-cluster { font-size: 0.75rem; letter-spacing: 0.15em; text-transform: uppercase; font-family: 'DM Mono', monospace; margin-bottom: 0.5rem; }
.result-title { font-size: 1.8rem; font-weight: 700; color: #FFFFFF; margin-bottom: 0.5rem; }
.result-rate { font-size: 1rem; margin-bottom: 1rem; }
.result-desc { font-size: 0.9rem; color: #8899AA; line-height: 1.65; max-width: 480px; margin: 0 auto; }

/* ── Insight box ── */
.insight-box {
    background: #0A1628;
    border-left: 3px solid #00B4D8;
    border-radius: 0 10px 10px 0;
    padding: 1.1rem 1.4rem;
    margin: 1.2rem 0;
    font-size: 0.88rem;
    color: #8899AA;
    line-height: 1.6;
}
.insight-box strong { color: #00B4D8; }

/* ── Divider ── */
.divider {
    height: 1px;
    background: linear-gradient(90deg, #1E3A5F 0%, transparent 100%);
    margin: 2rem 0;
}

/* ── Slider labels ── */
.slider-label {
    font-size: 0.78rem;
    color: #5A7A9A;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.08em;
    margin-bottom: 0.2rem;
}

/* ── Feature table ── */
.feature-row {
    display: flex;
    justify-content: space-between;
    padding: 0.6rem 0;
    border-bottom: 1px solid #1A2A3A;
    font-size: 0.85rem;
}
.feature-name { color: #8899AA; }
.feature-val { color: #FFFFFF; font-family: 'DM Mono', monospace; font-weight: 500; }
</style>
""", unsafe_allow_html=True)



# ── Cluster metadata ───────────────────────────────────────────────────────────

# ── Load saved model artifacts ─────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open("patient_segmentation_artifacts.pkl", "rb") as f:
        artifacts = pickle.load(f)
    km       = artifacts["kmeans"]
    scaler   = artifacts["scaler"]
    pca      = artifacts["pca"]
    df_clean = artifacts["df_clean"]
    X_scaled = artifacts["X_scaled"]
    X        = df_clean.drop("Outcome", axis=1)
    y        = df_clean["Outcome"]
    X_pca    = pca.transform(X_scaled)
    return df_clean, X_scaled, X, y, scaler, km, pca, X_pca

df_clean, X_scaled, X, y, scaler, km, pca, X_pca = load_artifacts()

CLUSTER_META = {
    0: {
        "name": "Age-Driven Risk",
        "color": "#E74C3C",
        "bg": "#1A1018",
        "badge": "badge-0",
        "card": "cluster-0",
        "rate_class": "rate-0",
        "desc": "Older patients (avg 46 yrs) with high pregnancy history. Risk driven by age and reproductive factors rather than metabolic markers.",
        "icon": "⏳"
    },
    1: {
        "name": "Low Risk",
        "color": "#2ECC71",
        "bg": "#0E1A12",
        "badge": "badge-1",
        "card": "cluster-1",
        "rate_class": "rate-1",
        "desc": "Youngest cohort (avg 26 yrs) with the healthiest metabolic profile across all indicators. Lowest diabetes prevalence by far.",
        "icon": "✅"
    },
    2: {
        "name": "Metabolic Risk",
        "color": "#3498DB",
        "bg": "#0E1525",
        "badge": "badge-2",
        "card": "cluster-2",
        "rate_class": "rate-2",
        "desc": "Relatively young patients (avg 29 yrs) with elevated glucose, insulin, and BMI. Risk is driven entirely by metabolic and obesity markers.",
        "icon": "⚠️"
    }
}

# ── Diabetes rate per cluster ──────────────────────────────────────────────────
diabetes_rate = df_clean.groupby('Cluster')['Outcome'].mean()
cluster_counts = df_clean['Cluster'].value_counts().sort_index()

# ══════════════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <div class="hero-label">🧬 Unsupervised Machine Learning · Healthcare Analytics</div>
    <div class="hero-title">Patient<span>IQ</span></div>
    <div class="hero-sub">
        K-Means clustering applied to 768 patients across 8 clinical indicators —
        surfacing distinct risk profiles without ever seeing a diagnosis label.
    </div>
</div>
""", unsafe_allow_html=True)

# ── Top metrics ────────────────────────────────────────────────────────────────
overall_rate = df_clean['Outcome'].mean()
sil = silhouette_score(X_scaled, km.labels_)

st.markdown(f"""
<div class="metric-grid">
    <div class="metric-card">
        <div class="metric-label">Patients Analysed</div>
        <div class="metric-value">768</div>
        <div class="metric-sub">Pima Indians Diabetes Dataset</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Clusters Discovered</div>
        <div class="metric-value">3</div>
        <div class="metric-sub">Optimal K via Elbow + Silhouette</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Silhouette Score</div>
        <div class="metric-value">{sil:.3f}</div>
        <div class="metric-sub">Cluster separation quality</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["📋  Overview", "📊  Cluster Insights", "🔬  Patient Predictor"])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
with tab1:

    st.markdown('<div class="section-header">🎯 What This Project Does</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">An unsupervised machine learning pipeline that discovers natural patient groupings from clinical data alone — no diagnosis labels used during training.</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
        <strong>Key finding:</strong> Three clinically distinct patient profiles emerged spontaneously from the data.
        The algorithm, with zero knowledge of actual diagnoses, separated a low-risk population (13% diabetes rate)
        from two high-risk groups (~52% each) — each driven by entirely different clinical pathways.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">🗂️ Discovered Risk Profiles</div>', unsafe_allow_html=True)

    # Cluster cards
    cards_html = '<div class="cluster-grid">'
    for c in [0, 1, 2]:
        m = CLUSTER_META[c]
        rate = diabetes_rate[c] * 100
        n = cluster_counts[c]
        cards_html += f"""
        <div class="cluster-card {m['card']}">
            <span class="cluster-badge {m['badge']}">Cluster {c}</span>
            <div class="cluster-name">{m['icon']} {m['name']}</div>
            <div class="cluster-rate {m['rate_class']}">{rate:.1f}% <span style="font-size:0.9rem;color:#5A7A9A;">diabetes rate</span></div>
            <div class="cluster-desc">{m['desc']}</div>
            <div class="cluster-stat">{n} patients · {n/768*100:.0f}% of cohort</div>
        </div>
        """
    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">🔬 Methodology</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="insight-box">
            <strong>Data Preprocessing</strong><br><br>
            Five features contained biologically impossible zero values used as missing data placeholders.
            These were replaced with column medians — robust to skew — preserving all 768 records.
            Features were then standardised with StandardScaler so no single variable dominates
            the distance calculation.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="insight-box">
            <strong>Choosing K = 3</strong><br><br>
            The Elbow Method showed diminishing inertia returns beyond K=4.
            Silhouette scores peaked at K=2 but remained strong at K=3.
            Domain reasoning settled the tie: three clusters map cleanly to
            low / age-driven / metabolic risk profiles — more actionable
            than two groups, more interpretable than four.
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — CLUSTER INSIGHTS
# ─────────────────────────────────────────────────────────────────────────────
with tab2:

    col_left, col_right = st.columns([1.1, 1], gap="large")

    # ── PCA scatter ──────────────────────────────────────────────────────────
    with col_left:
        st.markdown('<div class="section-header">🗺️ Patient Clusters in 2D Space</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">PCA compresses 8 clinical features into 2 dimensions for visualisation. Clusters are more separated in the original 8D space than shown here.</div>', unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor('#0D1520')
        ax.set_facecolor('#080C14')

        colors_map = {0: '#E74C3C', 1: '#2ECC71', 2: '#3498DB'}
        labels_map = {0: 'Age-Driven Risk', 1: 'Low Risk', 2: 'Metabolic Risk'}

        pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
        pca_df['Cluster'] = km.labels_

        for c in [0, 1, 2]:
            mask = pca_df['Cluster'] == c
            ax.scatter(pca_df.loc[mask, 'PC1'], pca_df.loc[mask, 'PC2'],
                       c=colors_map[c], label=labels_map[c],
                       alpha=0.65, edgecolors='none', s=28)

        pca_obj = PCA(n_components=2)
        pca_obj.fit(X_scaled)
        var = pca_obj.explained_variance_ratio_

        ax.set_xlabel(f'PC1  ({var[0]*100:.1f}% variance)', color='#5A7A9A', fontsize=9)
        ax.set_ylabel(f'PC2  ({var[1]*100:.1f}% variance)', color='#5A7A9A', fontsize=9)
        ax.tick_params(colors='#3A5A7A', labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#1E3A5F')

        legend = ax.legend(frameon=True, fontsize=8.5, facecolor='#0D1520',
                           edgecolor='#1E3A5F', labelcolor='#C0D0E0')
        ax.grid(True, alpha=0.08, color='#3A5A7A')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        total_var = sum(pca_obj.explained_variance_ratio_) * 100
        st.markdown(f'<div class="section-sub" style="margin-top:-0.5rem;">Total variance captured by 2 components: <strong style="color:#00B4D8">{total_var:.1f}%</strong></div>', unsafe_allow_html=True)

    # ── Bar chart ─────────────────────────────────────────────────────────────
    with col_right:
        st.markdown('<div class="section-header">📊 Diabetes Prevalence by Cluster</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Validation against actual diagnosis labels — never used during training.</div>', unsafe_allow_html=True)

        fig2, ax2 = plt.subplots(figsize=(5.5, 5))
        fig2.patch.set_facecolor('#0D1520')
        ax2.set_facecolor('#080C14')

        cluster_labels = ['Age-Driven\nRisk', 'Low\nRisk', 'Metabolic\nRisk']
        rates = [diabetes_rate[c] * 100 for c in [0, 1, 2]]
        bar_colors = ['#E74C3C', '#2ECC71', '#3498DB']

        bars = ax2.bar(cluster_labels, rates, color=bar_colors,
                       width=0.5, edgecolor='none', zorder=3)

        for bar, rate in zip(bars, rates):
            ax2.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 1.2,
                     f'{rate:.1f}%', ha='center',
                     fontsize=11, fontweight='700', color='#FFFFFF')

        # overall baseline
        ax2.axhline(overall_rate * 100, color='#00B4D8', linewidth=1,
                    linestyle='--', alpha=0.6, zorder=2)
        ax2.text(2.42, overall_rate * 100 + 1.5,
                 f'Dataset avg\n{overall_rate*100:.1f}%',
                 color='#00B4D8', fontsize=7.5, ha='right')

        ax2.set_ylim(0, 72)
        ax2.set_ylabel('Diabetes Rate (%)', color='#5A7A9A', fontsize=9)
        ax2.tick_params(colors='#5A7A9A', labelsize=9)
        for spine in ax2.spines.values():
            spine.set_edgecolor('#1E3A5F')
        ax2.grid(axis='y', alpha=0.08, color='#3A5A7A', zorder=1)

        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Feature profile table ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">📋 Clinical Feature Averages by Cluster</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Mean values per cluster reveal what defines each patient group.</div>', unsafe_allow_html=True)

    profiles = df_clean.groupby('Cluster')[X.columns.tolist()].mean().round(1)

    fig3, ax3 = plt.subplots(figsize=(11, 4.5))
    fig3.patch.set_facecolor('#0D1520')
    ax3.set_facecolor('#0D1520')

    features = X.columns.tolist()
    x = np.arange(len(features))
    width = 0.25
    bar_colors = ['#E74C3C', '#2ECC71', '#3498DB']
    cluster_names_list = ['Age-Driven Risk', 'Low Risk', 'Metabolic Risk']

    # Normalize each feature to 0-1 for visual comparison
    profiles_norm = profiles.copy()
    for col in features:
        col_min, col_max = profiles[col].min(), profiles[col].max()
        if col_max > col_min:
            profiles_norm[col] = (profiles[col] - col_min) / (col_max - col_min)
        else:
            profiles_norm[col] = 0.5

    for i, c in enumerate([0, 1, 2]):
        vals = [profiles_norm.loc[c, f] for f in features]
        bars3 = ax3.bar(x + i * width, vals, width,
                        label=cluster_names_list[i],
                        color=bar_colors[i], alpha=0.85, edgecolor='none')

    ax3.set_xticks(x + width)
    ax3.set_xticklabels([f.replace('_', '\n') for f in features],
                         fontsize=8.5, color='#8899AA')
    ax3.set_ylabel('Relative Value (normalised)', color='#5A7A9A', fontsize=9)
    ax3.tick_params(colors='#5A7A9A')
    for spine in ax3.spines.values():
        spine.set_edgecolor('#1E3A5F')
    ax3.grid(axis='y', alpha=0.08, color='#3A5A7A')
    ax3.legend(frameon=True, facecolor='#0D1520', edgecolor='#1E3A5F',
               labelcolor='#C0D0E0', fontsize=9)
    ax3.set_title('Feature Profiles by Cluster (Normalised)', color='#8899AA',
                   fontsize=10, pad=12)

    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

    st.markdown("""
    <div class="insight-box">
        <strong>Notable finding:</strong> Clusters 0 and 2 share almost identical diabetes rates (~52%)
        but are clinically opposite. Cluster 0 patients are older with high pregnancy counts;
        Cluster 2 patients are younger but metabolically compromised — high glucose, insulin, and BMI.
        A classification model would not surface this distinction.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — PATIENT PREDICTOR
# ─────────────────────────────────────────────────────────────────────────────
with tab3:

    st.markdown('<div class="section-header">🔬 New Patient Risk Assessment</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Enter a patient\'s clinical measurements. The model will assign them to the most similar cluster and explain what that means.</div>', unsafe_allow_html=True)

    col_inputs, col_result = st.columns([1, 1], gap="large")

    with col_inputs:
        st.markdown('<div class="slider-label">PREGNANCIES</div>', unsafe_allow_html=True)
        pregnancies = st.slider('Pregnancies', 0, 17, 3, label_visibility='collapsed')

        st.markdown('<div class="slider-label">GLUCOSE (mg/dL)</div>', unsafe_allow_html=True)
        glucose = st.slider('Glucose', 44, 199, 120, label_visibility='collapsed')

        st.markdown('<div class="slider-label">BLOOD PRESSURE (mmHg)</div>', unsafe_allow_html=True)
        blood_pressure = st.slider('Blood Pressure', 24, 122, 70, label_visibility='collapsed')

        st.markdown('<div class="slider-label">SKIN THICKNESS (mm)</div>', unsafe_allow_html=True)
        skin_thickness = st.slider('Skin Thickness', 7, 99, 23, label_visibility='collapsed')

        st.markdown('<div class="slider-label">INSULIN (μU/mL)</div>', unsafe_allow_html=True)
        insulin = st.slider('Insulin', 14, 846, 80, label_visibility='collapsed')

        st.markdown('<div class="slider-label">BMI (kg/m²)</div>', unsafe_allow_html=True)
        bmi = st.slider('BMI', 18.0, 67.1, 32.0, step=0.1, label_visibility='collapsed')

        st.markdown('<div class="slider-label">DIABETES PEDIGREE FUNCTION</div>', unsafe_allow_html=True)
        dpf = st.slider('DPF', 0.078, 2.420, 0.47, step=0.001, label_visibility='collapsed')

        st.markdown('<div class="slider-label">AGE (years)</div>', unsafe_allow_html=True)
        age = st.slider('Age', 21, 81, 30, label_visibility='collapsed')

    with col_result:
        # ── Predict ──────────────────────────────────────────────────────────
        input_data = np.array([[pregnancies, glucose, blood_pressure,
                                skin_thickness, insulin, bmi, dpf, age]])
        input_scaled = scaler.transform(input_data)
        predicted_cluster = int(km.predict(input_scaled)[0])
        m = CLUSTER_META[predicted_cluster]
        rate = diabetes_rate[predicted_cluster] * 100

        st.markdown(f"""
        <div class="result-card" style="background:{m['bg']}; border-color:{m['color']};">
            <div class="result-cluster" style="color:{m['color']};">Assigned to Cluster {predicted_cluster}</div>
            <div class="result-title">{m['icon']} {m['name']}</div>
            <div class="result-rate" style="color:{m['color']};">{rate:.1f}% diabetes prevalence in this group</div>
            <div class="result-desc">{m['desc']}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div style="margin-top:1.5rem;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header" style="font-size:0.9rem;">Patient Input Summary</div>', unsafe_allow_html=True)

        feature_names = ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness',
                         'Insulin', 'BMI', 'Diabetes Pedigree', 'Age']
        feature_vals = [pregnancies, glucose, blood_pressure, skin_thickness,
                        insulin, f'{bmi:.1f}', f'{dpf:.3f}', age]

        rows_html = ''
        for name, val in zip(feature_names, feature_vals):
            rows_html += f'<div class="feature-row"><span class="feature-name">{name}</span><span class="feature-val">{val}</span></div>'

        st.markdown(f'<div style="background:#0D1520;border:1px solid #1E3A5F;border-radius:10px;padding:1rem 1.4rem;">{rows_html}</div>', unsafe_allow_html=True)
