import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── 1) Load patient-level data (cached to disk) ───────────────────────────────
@st.cache_data(persist="disk")
def load_data():
    url = (
      "https://www.dropbox.com/scl/fi/8o976w9g9k3heeclbe8mb/"
      "my_table.csv?rlkey=0114ysndn9aa27d15j2nd36je&dl=1"
    )
    df = pd.read_csv(url)
    df["mechanically_ventilated"] = df["mechanically_ventilated"].map({
        "Yes": 1, "No": 0,
        "InvasiveVent": 1, "SupplementalOxygen": 0,
        np.nan: np.nan
    }).astype(float)
    df["sepsis3"] = df["sepsis3"].fillna(0).astype(int)
    df["outcome"] = np.where(
        df["days_until_death"].notna() & (df["days_until_death"] < 10),
        "early_death",
        np.where(df["hospital_los_days"] >= 10, "long_los", "short_los")
    )
    return df

# ── 2) Load pre-trained & pre-calibrated models ───────────────────────────────
@st.cache_resource
def load_models():
    # Use mmap_mode to reduce RAM footprint
    sep_m = joblib.load("model_sepsis_calibrated.pkl", mmap_mode="r")
    non_m = joblib.load("model_nonsepsis_calibrated.pkl", mmap_mode="r")
    # Load saved classification reports
    sep_r = joblib.load("model_sepsis_report.pkl")
    non_r = joblib.load("model_nonsepsis_report.pkl")
    return sep_r, sep_m, non_r, non_m

# ── 3) App UI ────────────────────────────────────────────────────────────────
st.title("Multiclass LOS Classifier (Early Death, Short, Long)")

st.markdown("""
**Class definitions**:
- **early_death**: died in hospital < 10 days  
- **short_los**: stay < 10 days  
- **long_los**: stay ≥ 10 days  
""")

# Load once
df = load_data()
report_sep, model_sep, report_non, model_non = load_models()

# Display precomputed metrics
st.subheader("Performance: Sepsis-3 Positive (Internal CV)")
st.table(report_sep.style.format({
    "precision": "{:.2f}", "recall": "{:.2f}",
    "f1-score": "{:.2f}", "support": "{:.0f}"
}))
st.subheader("Performance: Sepsis-3 Negative (Internal CV)")
st.table(report_non.style.format({
    "precision": "{:.2f}", "recall": "{:.2f}",
    "f1-score": "{:.2f}", "support": "{:.0f}"
}))

# ── 4) Individual patient prediction ─────────────────────────────────────────
FEATURES = [
    "age", "charlson", "sapsii", "ventilation",
    "bun_mg_dl", "creatinine_mg_dl", "mechanically_ventilated",
    "sofa_score", "respiration", "coagulation",
    "liver", "cardiovascular", "cns", "renal"
]
NUMERIC = [f for f in FEATURES if f not in ["ventilation", "mechanically_ventilated"]]
BINARY = ["mechanically_ventilated"]

st.subheader("Predict an Individual Patient Outcome")
with st.form("patient_form"):
    inputs = {}
    for feat in FEATURES:
        label = feat.replace('_', ' ').title()
        if feat in NUMERIC:
            inputs[feat] = st.number_input(label, value=float('nan'))
        elif feat in BINARY:
            inputs[feat] = st.selectbox(label, ['Unknown','Yes','No'])
        else:
            choices = ['Unknown'] + list(df[feat].dropna().unique())
            inputs[feat] = st.selectbox(label, choices)
    seps = st.selectbox("Meets Sepsis-3 criteria?", ['Yes','No'])
    submitted = st.form_submit_button("Compute")

if submitted:
    pat = {}
    for feat, val in inputs.items():
        if feat in NUMERIC:
            pat[feat] = val
        elif feat in BINARY:
            pat[feat] = 1 if val=='Yes' else 0 if val=='No' else np.nan
        else:
            pat[feat] = val if val!='Unknown' else np.nan
    pat_df = pd.DataFrame([pat])
    model = model_sep if seps=='Yes' else model_non
    pred = model.predict(pat_df)[0]
    probs = model.predict_proba(pat_df)[0]
    st.write(f"**Predicted class:** {pred}")
    st.write("**Class probabilities:**")
    proba_df = pd.DataFrame([probs], columns=model.classes_)
    st.table(proba_df.style.format("{:.2%}"))