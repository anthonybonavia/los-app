import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import os
from pathlib import Path

# ── 1) Load patient data (cached to disk) ─────────────────────────────────
@st.cache_data(persist="disk")
def load_data():
    local_path = Path(__file__).parent / "my_table.csv"
    if local_path.exists():
        df = pd.read_csv(local_path, low_memory=False)
    else:
        url = (
            "https://www.dropbox.com/scl/fi/8o976w9g9k3heeclbe8mb/"
            "my_table.csv?rlkey=0114ysndn9aa27d15j2nd36je&dl=1"
        )
        df = pd.read_csv(url, low_memory=False)

    # map mechanically_ventilated and cast
    df["mechanically_ventilated"] = df["mechanically_ventilated"].map({
        "Yes": 1,
        "No": 0,
        "InvasiveVent": 1,
        "SupplementalOxygen": 0,
        np.nan: np.nan
    }).astype(float)
    df["sepsis3"] = df["sepsis3"].fillna(0).astype(int)
    df["outcome"] = np.where(
        df["days_until_death"].notna() & (df["days_until_death"] < 10),
        "early_death",
        np.where(df["hospital_los_days"] >= 10, "long_los", "short_los")
    )
    return df

# ── 2) Load pre-trained models & reports from Dropbox ───────────────────────
@st.cache_resource
def load_models():
    # these all must be the /s/XXXXXXXX/ URLs, with www→dl.dropboxusercontent and dl=1
    urls = {
        "model_nonsepsis":    "https://dl.dropboxusercontent.com/s/c2oetaenrktyxe9kbj8nh/model_nonsepsis_calibrated.pkl?rlkey=repy9bvq99hl90bc4jwnhk3a3&dl=1",
        "report_nonsepsis":   "https://dl.dropboxusercontent.com/s/bmavnuly2ec9vj6cgf1ht/model_nonsepsis_report.pkl?rlkey=2pcotjymy9ktr41gcewb6ic7s&dl=1",
        "model_sepsis":       "https://dl.dropboxusercontent.com/s/puhntj2y9c4mv9k7fhmoo/model_sepsis_calibrated.pkl?rlkey=ty804av5nlg1ab8892u22w2xi&dl=1",
        "report_sepsis":      "https://dl.dropboxusercontent.com/s/kirx37p85nbbuwa3oliob/model_sepsis_report.pkl?rlkey=q1wfbh75aqsvnsf9sxchaee7m&dl=1",
    }

    local = {}
    for key, url in urls.items():
        fname = url.split("/")[-1].split("?")[0]
        if not Path(fname).exists() or os.path.getsize(fname) < 100:
            r = requests.get(url, stream=True)
            r.raise_for_status()
            with open(fname, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
            # sanity check: ensure it’s not an HTML error page
            hdr = open(fname, "rb").read(10)
            if hdr.startswith(b"<"):
                raise RuntimeError(
                    f"{fname} looks like HTML (got {hdr!r}). "
                    "Make sure you’re using a dl.dropboxusercontent.com URL with dl=1!"
                )
        st.write(f"🔄 Cached `{fname}` ({os.path.getsize(fname)/1e6:.1f} MB)")
        local[key] = fname

    non_model   = joblib.load(local["model_nonsepsis"], mmap_mode="r")
    sepsis_model= joblib.load(local["model_sepsis"],    mmap_mode="r")
    non_report  = joblib.load(local["report_nonsepsis"])
    sepsis_report = joblib.load(local["report_sepsis"])

    # return in the same order your UI expects:
    #   sepsis_report, sepsis_model, nonsepsis_report, nonsepsis_model
    return sepsis_report, sepsis_model, non_report, non_model
# ── 3) App UI ───────────────────────────────────────────────────────────────
st.title("Multiclass LOS Classifier (Early Death, Short, Long)")

st.markdown("""
**Class definitions**:
- **early_death**: died in hospital < 10 days  
- **short_los**: stay < 10 days  
- **long_los**: stay ≥ 10 days  
""")

# Load data & models
df = load_data()
sepsis_report, sepsis_model, nonsepsis_report, nonsepsis_model = load_models()

# Display metrics
st.subheader("Performance: Sepsis-3 Positive (Internal CV)")
st.table(sepsis_report.style.format({
    'precision': '{:.2f}', 'recall': '{:.2f}',
    'f1-score': '{:.2f}', 'support': '{:.0f}'
}))

st.subheader("Performance: Sepsis-3 Negative (Internal CV)")
st.table(nonsepsis_report.style.format({
    'precision': '{:.2f}', 'recall': '{:.2f}',
    'f1-score': '{:.2f}', 'support': '{:.0f}'
}))

# Individual patient prediction
FEATURES = [
    'age','charlson','sapsii','ventilation',
    'bun_mg_dl','creatinine_mg_dl','mechanically_ventilated',
    'sofa_score','respiration','coagulation',
    'liver','cardiovascular','cns','renal'
]
NUMERIC = [f for f in FEATURES if f not in ('ventilation','mechanically_ventilated')]
BINARY  = ['mechanically_ventilated']

st.subheader("Predict an Individual Patient Outcome")
with st.form('patient_form'):
    inputs = {}
    for feat in FEATURES:
        label = feat.replace('_',' ').title()
        if feat in NUMERIC:
            inputs[feat] = st.number_input(label, value=float('nan'))
        elif feat in BINARY:
            inputs[feat] = st.selectbox(label, ['Unknown','Yes','No'])
        else:
            choices = ['Unknown'] + list(df[feat].dropna().unique())
            inputs[feat] = st.selectbox(label, choices)
    seps = st.selectbox('Meets Sepsis-3 criteria?', ['Yes','No'])
    submitted = st.form_submit_button('Compute')

# … earlier code …

if submitted:
    # build patient row
    pat = {}
    for feat, val in inputs.items():
        if feat in NUMERIC:
            pat[feat] = val
        elif feat in BINARY:
            pat[feat] = 1 if val=='Yes' else 0 if val=='No' else np.nan
        else:
            pat[feat] = val if val!='Unknown' else np.nan

    pat_df = pd.DataFrame([pat])

    # **FILL MISSING BEFORE PREDICTING**
    pat_df[NUMERIC] = pat_df[NUMERIC].fillna(0)
    pat_df[BINARY]  = pat_df[BINARY].fillna(0)
    pat_df['ventilation'] = pat_df['ventilation'].fillna('Unknown')

    # pick model & predict
    model = sepsis_model if seps=='Yes' else nonsepsis_model
    pred  = model.predict(pat_df)[0]
    probs = model.predict_proba(pat_df)[0]

    st.write(f"**Predicted class:** {pred}")
    st.write("**Class probabilities:**")
    proba_df = pd.DataFrame([probs], columns=model.classes_)
    st.table(proba_df.style.format("{:.2%}"))
