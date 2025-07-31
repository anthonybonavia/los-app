import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from io import BytesIO
import sys, os

import subprocess, pkgutil
st.write("Python:", sys.version.replace("\n"," "))
st.write("Installed numpy:", __import__("numpy").__version__)
st.write("Installed scikit-learn:", __import__("sklearn").__version__)
st.write("Installed joblib:", __import__("joblib").__version__)

subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=False)


# ── 1) Load patient reference data for form dropdowns ─────────────────────────
@st.cache_data
def load_data():
    # Replace with your actual Dropbox direct-download link for my_table.csv
    url = (
        "https://www.dropbox.com/scl/fi/8o976w9g9k3heeclbe8mb/my_table.csv?rlkey=0114ysndn9aa27d15j2nd36je&dl=1"
    )
    df = pd.read_csv(url, low_memory=False)
    # map mechanically_ventilated strings → 0/1
    df["mechanically_ventilated"] = df["mechanically_ventilated"].map({
        "Yes": 1,
        "No": 0,
        "InvasiveVent": 1,
        "SupplementalOxygen": 0,
        np.nan: np.nan
    }).astype(float)
    # ensure sepsis3 flag
    df["sepsis3"] = df["sepsis3"].fillna(0).astype(int)
    # return only the columns needed for form choices
    return df

# ── 2) Load pre‐built, calibrated models from Dropbox ─────────────────────────
@st.cache_resource
def load_models():
    sep_url = "https://www.dropbox.com/scl/fi/puhntj2y9c4mv9k7fhmoo/model_sepsis_calibrated.pkl?rlkey=ty804av5nlg1ab8892u22w2xi&dl=1"
    non_url = "https://www.dropbox.com/scl/fi/c2oetaenrktyxe9kbj8nh/model_nonsepsis_calibrated.pkl?rlkey=repy9bvq99hl90bc4jwnhk3a3&dl=1"

    # fetch bytes and load pickles
    r1 = requests.get(sep_url)
    r1.raise_for_status()
    model_sep = joblib.load(BytesIO(r1.content))

    r2 = requests.get(non_url)
    r2.raise_for_status()
    model_non = joblib.load(BytesIO(r2.content))

    return model_sep, model_non

# ── 3) Feature definitions ──────────────────────────────────────────────────
FEATURES    = [
    "age", "charlson", "sapsii", "ventilation",
    "bun_mg_dl", "creatinine_mg_dl", "mechanically_ventilated",
    "sofa_score", "respiration", "coagulation",
    "liver", "cardiovascular", "cns", "renal"
]
NUMERIC     = [f for f in FEATURES if f not in ("ventilation", "mechanically_ventilated")]
BINARY      = ["mechanically_ventilated"]
CATEGORICAL = ["ventilation"]

# ── 4) Load data & models ───────────────────────────────────────────────────
df = load_data()
model_sep, model_non = load_models()

# ── 5) Build the Streamlit UI ────────────────────────────────────────────────
st.title("Multiclass LOS Classifier (Early Death, Short Stay, Long Stay)")

st.markdown("""
**Class definitions**  
- **early_death**: died in hospital < 10 days  
- **short_los**: stay < 10 days  
- **long_los**: stay ≥ 10 days  
""")

st.subheader("Predict an Individual Patient Outcome")
with st.form("patient_form"):
    inputs = {}
    for feat in FEATURES:
        label = feat.replace("_", " ").title()
        if feat in NUMERIC:
            inputs[feat] = st.number_input(label, value=float("nan"))
        elif feat in BINARY:
            inputs[feat] = st.selectbox(label, ["Unknown", "Yes", "No"])
        else:
            options = ["Unknown"] + sorted(df[feat].dropna().unique().tolist())
            inputs[feat] = st.selectbox(label, options)
    sepsis_flag = st.selectbox("Meets Sepsis-3 Criteria?", ["Yes", "No"])
    submitted = st.form_submit_button("Compute")

if submitted:
    # Assemble patient DataFrame
    pat = {}
    for feat, val in inputs.items():
        if feat in NUMERIC:
            pat[feat] = val
        elif feat in BINARY:
            if val == "Yes":
                pat[feat] = 1
            elif val == "No":
                pat[feat] = 0
            else:
                pat[feat] = np.nan
        else:
            pat[feat] = val if val != "Unknown" else np.nan

    pat_df = pd.DataFrame([pat])

    # Select the appropriate pre‐trained model
    model = model_sep if sepsis_flag == "Yes" else model_non

    # Predict and display
    pred = model.predict(pat_df)[0]
    probs = model.predict_proba(pat_df)[0]

    st.write(f"**Predicted class:** {pred}")
    st.write("**Class probabilities:**")
    proba_df = pd.DataFrame([probs], columns=model.classes_)
    st.table(proba_df.style.format("{:.2%}"))
