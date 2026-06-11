# 🧬 PatientIQ — Patient Segmentation Using K-Means Clustering

> Unsupervised machine learning applied to clinical data — discovering distinct patient risk profiles without ever seeing a diagnosis label.

---

## 📌 Project Overview

This project applies **K-Means clustering** to the Pima Indians Diabetes Dataset to segment 768 patients into clinically meaningful risk groups based on 8 physiological indicators.

The algorithm was given **no diagnosis information** during training. The resulting clusters were validated against actual outcomes only at the end — and the separation was striking.

**Live App →** [Deploy link here]

---

## 🔍 Key Finding

Three distinct patient profiles emerged from the data:

| Cluster | Profile | Diabetes Rate | Defining Features |
|---|---|---|---|
| 0 | Age-Driven Risk | 51.6% | Oldest cohort (avg 46), highest pregnancies |
| 1 | Low Risk | 13.1% | Youngest cohort (avg 26), healthiest metabolic markers |
| 2 | Metabolic Risk | 51.9% | Elevated glucose, insulin, BMI despite younger age |

> Clusters 0 and 2 share nearly identical diabetes rates but represent completely different clinical risk pathways — a distinction that supervised classification alone would not surface.

---

## 🗂️ Dataset

**Pima Indians Diabetes Database** — UCI Machine Learning Repository  
768 patients · 8 clinical features · Binary diabetes outcome

| Feature | Description |
|---|---|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration (mg/dL) |
| Blood_Pressure | Diastolic blood pressure (mmHg) |
| Skin_Thickness | Triceps skin fold thickness (mm) |
| Insulin | 2-hour serum insulin (μU/mL) |
| BMI | Body mass index (kg/m²) |
| Diabetes_Pedigree | Diabetes pedigree function score |
| Age | Age in years |

---

## ⚙️ Methodology

### 1. Data Preprocessing
- Five columns contained zeros used as missing value placeholders (biologically impossible values)
- Replaced with column **medians** — robust to the heavy skew present in Insulin and Skin Thickness
- Features standardised with `StandardScaler` — essential for distance-based algorithms like K-Means

### 2. Optimal K Selection
- **Elbow Method** — inertia curve shows diminishing returns beyond K=4
- **Silhouette Score** — peaks at K=2, remains strong at K=3, declines beyond
- **Final choice: K=3** — balances statistical evidence with clinical interpretability

### 3. Dimensionality Reduction
- PCA applied post-clustering for 2D visualisation only
- 47.2% of total variance captured (PC1: 28.5%, PC2: 18.7%)
- Clusters are more separated in original 8D space than the projection suggests

---

## 🚀 Running the App Locally

```bash
git clone https://github.com/yourusername/patient-segmentation
cd patient-segmentation
pip install -r requirements.txt
streamlit run app.py
```

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **scikit-learn** — KMeans, StandardScaler, PCA, silhouette_score
- **pandas / numpy** — data manipulation
- **matplotlib / seaborn** — visualisation
- **Streamlit** — interactive web application

---

## 📁 Repository Structure

```
patient-segmentation/
│
├── app.py               # Streamlit application
├── diabetes.csv         # Dataset
├── requirements.txt     # Dependencies
└── README.md            # This file
```

---

## 👤 Author

**Muhammad Zubairu**  
Data Science Student - Machine Learning Engineer 
[GitHub](https://github.com/Kaida4318) · [LinkedIn](https://linkedin.com/in/muhammad-zubairu-rabiu)
