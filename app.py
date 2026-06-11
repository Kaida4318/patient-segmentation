import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="PatientIQ — Cluster Analysis",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
    background-color: #F0F4F8 !important;
    color: #1A2B3C !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2.5rem 4rem 2.5rem !important; max-width: 1280px !important; }

/* ── HERO ── */
.hero {
    background: linear-gradient(135deg, #0A2540 0%, #1B4F72 60%, #0E6655 100%);
    border-radius: 24px;
    padding: 4rem 4rem 3.5rem 4rem;
    margin: 2rem 0 2rem 0;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -100px; right: -100px;
    width: 500px; height: 500px;
    background: radial-gradient(circle, rgba(255,255,255,0.04) 0%, transparent 70%);
    border-radius: 50%;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -60px; left: 40%;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(30,215,96,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.45);
    margin-bottom: 1.2rem;
}
.hero-h1 {
    font-family: 'Sora', sans-serif;
    font-size: 3.8rem;
    font-weight: 800;
    color: #FFFFFF;
    line-height: 1.08;
    letter-spacing: -0.03em;
    margin-bottom: 1rem;
}
.hero-h1 em { color: #1ED760; font-style: normal; }
.hero-body {
    font-size: 1.05rem;
    color: rgba(255,255,255,0.6);
    max-width: 520px;
    line-height: 1.75;
}
.hero-ghost {
    position: absolute;
    right: 3rem; top: 50%;
    transform: translateY(-50%);
    font-family: 'Sora', sans-serif;
    font-size: 10rem;
    font-weight: 800;
    color: rgba(255,255,255,0.04);
    letter-spacing: -0.05em;
    line-height: 1;
    pointer-events: none;
    user-select: none;
}

/* ── METRIC STRIP ── */
.mstrip {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 2rem;
}
.mcard {
    background: #FFFFFF;
    border-radius: 16px;
    padding: 1.5rem 1.8rem;
    border: 1px solid #E2EAF0;
    box-shadow: 0 2px 12px rgba(10,37,64,0.06);
    transition: box-shadow 0.2s;
}
.mcard:hover { box-shadow: 0 6px 24px rgba(10,37,64,0.1); }
.mlbl {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #94A8B8;
    margin-bottom: 0.5rem;
}
.mnum {
    font-family: 'Sora', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: #0A2540;
    line-height: 1;
}
.mnum.green { color: #0E9E5A; }
.msub { font-size: 0.75rem; color: #94A8B8; margin-top: 0.3rem; }

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    gap: 0 !important;
    border-bottom: 2px solid #E2EAF0 !important;
    margin-bottom: 2rem !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #94A8B8 !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    padding: 0.8rem 1.8rem !important;
    border-bottom: 3px solid transparent !important;
    border-radius: 0 !important;
    margin-bottom: -2px !important;
}
.stTabs [aria-selected="true"] {
    color: #0A2540 !important;
    border-bottom: 3px solid #0A2540 !important;
    background: transparent !important;
    font-weight: 600 !important;
}

/* ── SECTION ── */
.slbl {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #94A8B8;
    margin-bottom: 0.3rem;
    margin-top: 1.8rem;
}
.stitle {
    font-family: 'Sora', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #0A2540;
    margin-bottom: 0.4rem;
}
.sdesc {
    font-size: 0.85rem;
    color: #6A8497;
    margin-bottom: 1.4rem;
    line-height: 1.65;
    max-width: 680px;
}

/* ── INSIGHT BOX ── */
.ibox {
    background: #EBF8F2;
    border-left: 3px solid #0E9E5A;
    border-radius: 0 12px 12px 0;
    padding: 1.1rem 1.5rem;
    margin: 1.2rem 0;
    font-size: 0.87rem;
    color: #1A5C3A;
    line-height: 1.65;
}
.ibox b { color: #0A7A44; }

/* ── METHOD CARDS ── */
.meth {
    background: #FFFFFF;
    border: 1px solid #E2EAF0;
    border-radius: 14px;
    padding: 1.6rem;
    height: 100%;
    box-shadow: 0 2px 8px rgba(10,37,64,0.04);
}
.meth-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.16em;
    color: #0E9E5A;
    margin-bottom: 0.6rem;
}
.meth-title {
    font-family: 'Sora', sans-serif;
    font-weight: 600;
    color: #0A2540;
    margin-bottom: 0.7rem;
    font-size: 1rem;
}
.meth-body { font-size: 0.83rem; color: #6A8497; line-height: 1.65; }

/* ── DIVIDER ── */
.hdiv { height: 1px; background: #E2EAF0; margin: 2rem 0; }

/* ── SLIDER LABEL ── */
.sldr {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.63rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #94A8B8;
    margin-top: 0.6rem;
    margin-bottom: 0.1rem;
}

/* ── SUM ROW ── */
.srow {
    display: flex;
    justify-content: space-between;
    padding: 0.5rem 0;
    border-bottom: 1px solid #F0F4F8;
}
.skey { font-size: 0.82rem; color: #94A8B8; }
.sval {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    font-weight: 500;
    color: #0A2540;
}
</style>
""", unsafe_allow_html=True)

# ── Load artifacts ──────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open('patient_segmentation_artifacts.pkl', 'rb') as f:
        arts = pickle.load(f)
    km       = arts['kmeans']
    scaler   = arts['scaler']
    pca      = arts['pca']
    df_clean = arts['df_clean']
    X_scaled = arts['X_scaled']
    X        = df_clean.drop('Outcome', axis=1)
    y        = df_clean['Outcome']
    X_pca    = pca.transform(X_scaled)
    return df_clean, X_scaled, X, y, scaler, km, pca, X_pca

df_clean, X_scaled, X, y, scaler, km, pca, X_pca = load_artifacts()

diabetes_rate  = df_clean.groupby('Cluster')['Outcome'].mean()
cluster_counts = df_clean['Cluster'].value_counts().sort_index()
overall_rate   = df_clean['Outcome'].mean()
sil            = silhouette_score(X_scaled, km.labels_)
pca_var        = pca.explained_variance_ratio_

CLUSTERS = {
    0: {
        "name":    "Age-Driven Risk",
        "icon":    "⏳",
        "color":   "#E8410A",
        "light":   "#FEF0EB",
        "border":  "#FBCDB8",
        "text":    "#7A2000",
        "key":     "Age + Pregnancies",
        "desc":    "Oldest cohort (avg 46 yrs) with the highest number of pregnancies. Risk is driven by age and reproductive history rather than metabolic markers.",
    },
    1: {
        "name":    "Low Risk",
        "icon":    "✦",
        "color":   "#0E9E5A",
        "light":   "#EBF8F2",
        "border":  "#A8DFC4",
        "text":    "#0A5C34",
        "key":     "Youth + Low Glucose",
        "desc":    "Youngest cohort (avg 26 yrs) with the healthiest metabolic profile across all indicators. Only 13% diabetes prevalence.",
    },
    2: {
        "name":    "Metabolic Risk",
        "icon":    "⚡",
        "color":   "#D4780A",
        "light":   "#FEF7EC",
        "border":  "#F9D898",
        "text":    "#7A4200",
        "key":     "Glucose + BMI + Insulin",
        "desc":    "Relatively young patients (avg 29 yrs) but with the highest glucose, insulin, and BMI. Risk is entirely metabolic and obesity-driven.",
    },
}

# ── HERO ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-ghost">IQ</div>
    <div class="hero-tag">🧬 &nbsp; Unsupervised Machine Learning &nbsp;·&nbsp; Healthcare Analytics</div>
    <div class="hero-h1">Patient<em>IQ</em></div>
    <div class="hero-body">
        Three clinically distinct risk profiles discovered from 768 patients
        across 8 physiological indicators — with zero knowledge of diagnosis outcomes.
    </div>
</div>
""", unsafe_allow_html=True)

# ── METRICS ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="mstrip">
    <div class="mcard">
        <div class="mlbl">Patients Analysed</div>
        <div class="mnum">768</div>
        <div class="msub">Pima Indians Diabetes DB</div>
    </div>
    <div class="mcard">
        <div class="mlbl">Clinical Features</div>
        <div class="mnum">8</div>
        <div class="msub">Used for clustering</div>
    </div>
    <div class="mcard">
        <div class="mlbl">Clusters (K)</div>
        <div class="mnum green">3</div>
        <div class="msub">Elbow + Silhouette method</div>
    </div>
    <div class="mcard">
        <div class="mlbl">Silhouette Score</div>
        <div class="mnum green">{sil:.3f}</div>
        <div class="msub">Cluster separation quality</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── TABS ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["  📋  Overview  ", "  📊  Cluster Insights  ", "  🔬  Patient Predictor  "])

# ────────────────────────────────────────────────────────────────────────────
# TAB 1 — OVERVIEW
# ────────────────────────────────────────────────────────────────────────────
with tab1:

    st.markdown("""
    <div class="ibox">
        <b>Core finding —</b> An unsupervised algorithm, with zero access to diagnosis labels,
        separated a low-risk population (13% diabetes rate) from two high-risk groups (~52% each),
        each driven by entirely different clinical pathways. This is what unsupervised learning
        uniquely reveals — structure you didn't tell it to look for.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="slbl">Discovered Profiles</div>', unsafe_allow_html=True)
    st.markdown('<div class="stitle">Three Patient Risk Groups</div>', unsafe_allow_html=True)

    c0, c1, c2 = st.columns(3, gap="medium")
    for col, cid in zip([c0, c1, c2], [0, 1, 2]):
        m    = CLUSTERS[cid]
        rate = diabetes_rate[cid] * 100
        n    = cluster_counts[cid]
        with col:
            st.markdown(f"""
            <div style="background:#FFFFFF;border:1px solid {m['border']};
                         border-top:4px solid {m['color']};border-radius:16px;
                         padding:2rem 1.8rem;box-shadow:0 4px 20px rgba(10,37,64,0.07);
                         height:100%;">
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;
                             letter-spacing:0.15em;text-transform:uppercase;
                             color:{m['color']};margin-bottom:1rem;opacity:0.8;">
                    Cluster {cid} &nbsp;·&nbsp; {m['key']}
                </div>
                <div style="font-size:2rem;margin-bottom:0.4rem;">{m['icon']}</div>
                <div style="font-family:'Sora',sans-serif;font-size:1.3rem;font-weight:700;
                             color:#0A2540;margin-bottom:1.2rem;">{m['name']}</div>
                <div style="background:{m['light']};border-radius:10px;padding:1rem 1.2rem;
                             margin-bottom:1.2rem;">
                    <div style="font-family:'Sora',sans-serif;font-size:2.8rem;font-weight:800;
                                 color:{m['color']};line-height:1;">
                        {rate:.1f}<span style="font-size:1rem;font-weight:500;color:{m['color']};opacity:0.6;">%</span>
                    </div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;
                                 color:{m['text']};margin-top:0.2rem;letter-spacing:0.08em;">
                        diabetes prevalence
                    </div>
                </div>
                <div style="font-size:0.83rem;color:#6A8497;line-height:1.65;margin-bottom:1.2rem;">
                    {m['desc']}
                </div>
                <div style="font-size:0.72rem;color:#94A8B8;font-family:'JetBrains Mono',monospace;
                             border-top:1px solid #F0F4F8;padding-top:0.8rem;">
                    {n} patients &nbsp;·&nbsp; {n/768*100:.0f}% of cohort
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="hdiv"></div>', unsafe_allow_html=True)
    st.markdown('<div class="slbl">Methodology</div>', unsafe_allow_html=True)
    st.markdown('<div class="stitle">How the Clusters Were Found</div>', unsafe_allow_html=True)

    s1, s2, s3 = st.columns(3, gap="medium")
    steps = [
        ("01", "Data Cleaning",    "Five features contained biologically impossible zeros masking missing values. Replaced with column medians — robust to the heavy skew seen in Insulin and Skin Thickness distributions."),
        ("02", "Feature Scaling",  "StandardScaler applied across all 8 features. K-Means uses Euclidean distance — unscaled features like Insulin (0–846) would otherwise dominate the distance calculation entirely."),
        ("03", "Choosing K = 3",   "Elbow method: curve flattens after K=4. Silhouette peaks at K=2, holds strong at K=3. Domain reasoning settled the tie — three profiles are clinically actionable and clearly nameable."),
    ]
    for col, (num, title, body) in zip([s1, s2, s3], steps):
        with col:
            st.markdown(f"""
            <div class="meth">
                <div class="meth-num">Step {num}</div>
                <div class="meth-title">{title}</div>
                <div class="meth-body">{body}</div>
            </div>
            """, unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────
# TAB 2 — CLUSTER INSIGHTS
# ────────────────────────────────────────────────────────────────────────────
with tab2:

    left, right = st.columns([1.2, 1], gap="large")

    with left:
        st.markdown('<div class="slbl">Dimensionality Reduction</div>', unsafe_allow_html=True)
        st.markdown('<div class="stitle">Clusters in 2D Space (PCA)</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="sdesc">8 features compressed into 2 PCA components capturing {sum(pca_var)*100:.1f}% of total variance. Clusters are more separated in the original 8D space.</div>', unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(7.5, 5.2))
        fig.patch.set_facecolor('#FFFFFF')
        ax.set_facecolor('#F8FAFC')

        colors_map = {0: '#E8410A', 1: '#0E9E5A', 2: '#D4780A'}
        labels_map = {0: 'Age-Driven Risk', 1: 'Low Risk', 2: 'Metabolic Risk'}

        pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
        pca_df['Cluster'] = km.labels_

        for c in [0, 1, 2]:
            mask = pca_df['Cluster'] == c
            ax.scatter(pca_df.loc[mask, 'PC1'], pca_df.loc[mask, 'PC2'],
                       c=colors_map[c], label=labels_map[c],
                       alpha=0.55, edgecolors='white', linewidth=0.3,
                       s=28, zorder=3)

        ax.set_xlabel(f'PC1  —  {pca_var[0]*100:.1f}% variance explained',
                      color='#94A8B8', fontsize=8.5, labelpad=10)
        ax.set_ylabel(f'PC2  —  {pca_var[1]*100:.1f}% variance explained',
                      color='#94A8B8', fontsize=8.5, labelpad=10)
        ax.tick_params(colors='#C4D4E0', labelsize=7.5)
        for spine in ax.spines.values():
            spine.set_edgecolor('#E2EAF0')
        legend = ax.legend(frameon=True, fontsize=8.5, facecolor='#FFFFFF',
                           edgecolor='#E2EAF0', labelcolor='#0A2540',
                           loc='upper right', markerscale=1.6)
        ax.grid(True, alpha=0.5, color='#E2EAF0', linewidth=0.6)
        fig.tight_layout(pad=1.5)
        st.pyplot(fig)
        plt.close()

    with right:
        st.markdown('<div class="slbl">Validation Against Labels</div>', unsafe_allow_html=True)
        st.markdown('<div class="stitle">Diabetes Prevalence by Cluster</div>', unsafe_allow_html=True)
        st.markdown('<div class="sdesc">Diagnosis labels were withheld during training. This chart validates what the algorithm discovered blindly.</div>', unsafe_allow_html=True)

        fig2, ax2 = plt.subplots(figsize=(5.5, 5.2))
        fig2.patch.set_facecolor('#FFFFFF')
        ax2.set_facecolor('#F8FAFC')

        rates   = [diabetes_rate[c] * 100 for c in [0, 1, 2]]
        blabels = ['Age-Driven\nRisk', 'Low\nRisk', 'Metabolic\nRisk']
        bcolors = ['#E8410A', '#0E9E5A', '#D4780A']

        bars = ax2.bar(blabels, rates, color=bcolors, width=0.5,
                       edgecolor='none', zorder=3)

        for bar, rate in zip(bars, rates):
            ax2.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 1.2,
                     f'{rate:.1f}%',
                     ha='center', va='bottom',
                     fontsize=12, fontweight='700', color='#0A2540')

        ax2.axhline(overall_rate * 100, color='#0A2540', linewidth=1,
                    linestyle='--', alpha=0.25, zorder=2)
        ax2.text(2.42, overall_rate * 100 + 1,
                 f'Dataset avg\n{overall_rate*100:.1f}%',
                 color='#94A8B8', fontsize=7, ha='right')

        ax2.set_ylim(0, 75)
        ax2.set_ylabel('Diabetes Rate (%)', color='#94A8B8', fontsize=8.5, labelpad=10)
        ax2.tick_params(colors='#94A8B8', labelsize=9)
        for spine in ax2.spines.values():
            spine.set_edgecolor('#E2EAF0')
        ax2.grid(axis='y', alpha=0.5, color='#E2EAF0', linewidth=0.6, zorder=1)
        fig2.tight_layout(pad=1.5)
        st.pyplot(fig2)
        plt.close()

    st.markdown('<div class="hdiv"></div>', unsafe_allow_html=True)
    st.markdown('<div class="slbl">Feature Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="stitle">Clinical Profile per Cluster (Normalised)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sdesc">Mean feature values normalised 0–1 across clusters. Shows which clinical markers define each group.</div>', unsafe_allow_html=True)

    profiles      = df_clean.groupby('Cluster')[X.columns.tolist()].mean().round(1)
    profiles_norm = profiles.copy()
    for col in X.columns:
        cmin, cmax = profiles[col].min(), profiles[col].max()
        profiles_norm[col] = (profiles[col] - cmin) / (cmax - cmin) if cmax > cmin else 0.5

    fig3, ax3 = plt.subplots(figsize=(12, 4))
    fig3.patch.set_facecolor('#FFFFFF')
    ax3.set_facecolor('#F8FAFC')

    features = X.columns.tolist()
    x        = np.arange(len(features))
    width    = 0.26
    bcolors3 = ['#E8410A', '#0E9E5A', '#D4780A']
    bnames3  = ['Age-Driven Risk', 'Low Risk', 'Metabolic Risk']

    for i, c in enumerate([0, 1, 2]):
        vals = [profiles_norm.loc[c, f] for f in features]
        ax3.bar(x + i * width, vals, width, label=bnames3[i],
                color=bcolors3[i], alpha=0.82, edgecolor='none', zorder=3)

    ax3.set_xticks(x + width)
    ax3.set_xticklabels([f.replace('_', '\n') for f in features],
                         fontsize=8, color='#6A8497')
    ax3.set_ylabel('Relative Value (0–1)', color='#94A8B8', fontsize=8.5, labelpad=10)
    ax3.tick_params(colors='#C4D4E0', labelsize=8)
    for spine in ax3.spines.values():
        spine.set_edgecolor('#E2EAF0')
    ax3.grid(axis='y', alpha=0.5, color='#E2EAF0', linewidth=0.6, zorder=1)
    ax3.legend(frameon=True, facecolor='#FFFFFF', edgecolor='#E2EAF0',
               labelcolor='#0A2540', fontsize=8.5)
    fig3.tight_layout(pad=1.5)
    st.pyplot(fig3)
    plt.close()

    st.markdown("""
    <div class="ibox">
        <b>Notable finding —</b> Clusters 0 and 2 share nearly identical diabetes rates (~52%)
        but are clinically opposite in their drivers. Cluster 0 is shaped by age and reproductive
        history; Cluster 2 is shaped by metabolic dysfunction — high glucose, insulin, and BMI
        despite being younger. A supervised classification model would not surface this distinction.
        This is unsupervised learning's unique clinical contribution.
    </div>
    """, unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────
# TAB 3 — PATIENT PREDICTOR
# ────────────────────────────────────────────────────────────────────────────
with tab3:

    st.markdown('<div class="slbl">Live Inference</div>', unsafe_allow_html=True)
    st.markdown('<div class="stitle">New Patient Risk Assessment</div>', unsafe_allow_html=True)
    st.markdown('<div class="sdesc">Adjust the clinical measurements. The model assigns the patient to the closest discovered cluster in real time using your trained K-Means model.</div>', unsafe_allow_html=True)

    inp, res = st.columns([1, 1], gap="large")

    with inp:
        st.markdown('<div style="background:#FFFFFF;border:1px solid #E2EAF0;border-radius:16px;padding:1.8rem 2rem;box-shadow:0 2px 12px rgba(10,37,64,0.06);">', unsafe_allow_html=True)

        st.markdown('<div class="sldr">Pregnancies</div>', unsafe_allow_html=True)
        pregnancies = st.slider('Pregnancies', 0, 17, 3, label_visibility='collapsed')

        st.markdown('<div class="sldr">Glucose (mg/dL)</div>', unsafe_allow_html=True)
        glucose = st.slider('Glucose', 44, 199, 120, label_visibility='collapsed')

        st.markdown('<div class="sldr">Blood Pressure (mmHg)</div>', unsafe_allow_html=True)
        blood_pressure = st.slider('Blood Pressure', 24, 122, 70, label_visibility='collapsed')

        st.markdown('<div class="sldr">Skin Thickness (mm)</div>', unsafe_allow_html=True)
        skin_thickness = st.slider('Skin Thickness', 7, 99, 23, label_visibility='collapsed')

        st.markdown('<div class="sldr">Insulin (μU/mL)</div>', unsafe_allow_html=True)
        insulin = st.slider('Insulin', 14, 846, 80, label_visibility='collapsed')

        st.markdown('<div class="sldr">BMI (kg/m²)</div>', unsafe_allow_html=True)
        bmi = st.slider('BMI', 18.0, 67.1, 32.0, step=0.1, label_visibility='collapsed')

        st.markdown('<div class="sldr">Diabetes Pedigree Function</div>', unsafe_allow_html=True)
        dpf = st.slider('DPF', 0.078, 2.420, 0.47, step=0.001, label_visibility='collapsed')

        st.markdown('<div class="sldr">Age (years)</div>', unsafe_allow_html=True)
        age = st.slider('Age', 21, 81, 30, label_visibility='collapsed')

        st.markdown('</div>', unsafe_allow_html=True)

    with res:
        input_arr    = np.array([[pregnancies, glucose, blood_pressure,
                                  skin_thickness, insulin, bmi, dpf, age]])
        input_scaled = scaler.transform(input_arr)
        pred_cluster = int(km.predict(input_scaled)[0])
        m            = CLUSTERS[pred_cluster]
        rate         = diabetes_rate[pred_cluster] * 100

        # Result card
        st.markdown(f"""
        <div style="background:#FFFFFF;border:1px solid {m['border']};
                     border-top:5px solid {m['color']};border-radius:16px;
                     padding:2.2rem;text-align:center;margin-bottom:1.2rem;
                     box-shadow:0 4px 24px rgba(10,37,64,0.08);">
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.62rem;
                         letter-spacing:0.18em;text-transform:uppercase;
                         color:{m['color']};margin-bottom:0.6rem;opacity:0.8;">
                Cluster {pred_cluster} &nbsp;·&nbsp; Assigned Profile
            </div>
            <div style="font-size:2.2rem;margin:0.5rem 0;">{m['icon']}</div>
            <div style="font-family:'Sora',sans-serif;font-size:1.8rem;font-weight:800;
                         color:#0A2540;margin-bottom:1rem;">{m['name']}</div>
            <div style="background:{m['light']};border-radius:12px;
                         padding:1.2rem 1.5rem;margin-bottom:1.2rem;">
                <div style="font-family:'Sora',sans-serif;font-size:4rem;font-weight:800;
                             color:{m['color']};line-height:1;">
                    {rate:.1f}<span style="font-size:1.4rem;color:{m['color']};opacity:0.5;">%</span>
                </div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;
                             color:{m['text']};letter-spacing:0.08em;">
                    diabetes prevalence in this group
                </div>
            </div>
            <div style="font-size:0.85rem;color:#6A8497;line-height:1.65;
                         max-width:340px;margin:0 auto;">{m['desc']}</div>
        </div>
        """, unsafe_allow_html=True)

        # Summary table
        st.markdown("""
        <div style="background:#FFFFFF;border:1px solid #E2EAF0;border-radius:12px;
                     padding:1.4rem 1.6rem;box-shadow:0 2px 8px rgba(10,37,64,0.04);">
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;
                         letter-spacing:0.16em;text-transform:uppercase;
                         color:#94A8B8;margin-bottom:1rem;">Input Summary</div>
        """, unsafe_allow_html=True)

        rows = [
            ("Pregnancies",    pregnancies),
            ("Glucose",        f"{glucose} mg/dL"),
            ("Blood Pressure", f"{blood_pressure} mmHg"),
            ("Skin Thickness", f"{skin_thickness} mm"),
            ("Insulin",        f"{insulin} μU/mL"),
            ("BMI",            f"{bmi:.1f} kg/m²"),
            ("Pedigree (DPF)", f"{dpf:.3f}"),
            ("Age",            f"{age} yrs"),
        ]
        rows_html = "".join(f"""
        <div class="srow">
            <span class="skey">{k}</span>
            <span class="sval">{v}</span>
        </div>""" for k, v in rows)

        st.markdown(rows_html + "</div>", unsafe_allow_html=True)