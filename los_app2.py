import streamlit as st
import traceback
import sys
import time
from io import BytesIO

# --- startup imports with fallback diagnostics ---
try:
    import joblib
    import pandas as pd
    import numpy as np
    import requests
except Exception as e:
    st.error("Startup import failure – cannot proceed. Details below:")
    st.text("".join(traceback.format_exception(e.__class__, e, e.__traceback__)))
    sys.exit(1)

# --- environment diagnostics ---
st.write("### Environment diagnostics")
st.write("Python:", sys.version.replace("\n", " "))
st.write("Installed numpy:", getattr(np, "__version__", "missing"))
try:
    import sklearn

    st.write("Installed scikit-learn:", sklearn.__version__)
except ImportError:
    st.write("scikit-learn: **not installed**")
st.write("Installed joblib:", getattr(joblib, "__version__", "missing"))

# --- 1) Load patient reference data ---
@st.cache_data
def load_data():
    url = "https://www.dropbox.com/scl/fi/8o976w9g9k3heeclbe8mb/my_table.csv?rlkey=0114ysndn9aa27d15j2nd36je&dl=1"
    df = pd.read_csv(url, low_memory=False)

    # map mechanically_ventilated strings → 0/1
    df["mechanically_ventilated"] = df["mechanically_ventilated"].map(
        {
            "Yes": 1,
            "No": 0,
            "InvasiveVent": 1,
            "SupplementalOxygen": 0,
            np.nan: np.nan,
        }
    ).astype(float)

    df["sepsis3"] = df.get("sepsis3", 0).fillna(0).astype(int)
    return df


# --- 2) Load prebuilt models from Dropbox with retries and sanity check ---
@st.cache_resource
def load_models():
    sep_url = "https://www.dropbox.com/scl/fi/lenmtptch7wkxvycm9l2j/model_sepsis_calibrated_compressed.pkl?rlkey=pzlownxh2qnin26vjaz4rlh8o&dl=1"
    non_url = "https://www.dropbox.com/scl/fi/pmv1s1p4r00v7uogf8zgf/model_nonsepsis_calibrated_compressed.pkl?rlkey=06746p5qux626hgzj1705ubw0&dl=1"

    def fetch_model(name, url, max_attempts=3):
        last_exc = None
        for attempt in range(1, max_attempts + 1):
            st.write(f"Attempting to download {name} model, try {attempt}")
            try:
                r = requests.get(url, timeout=30)
                st.write(f"HTTP status for {name} model: {r.status_code}")
                r.raise_for_status()
                size = len(r.content)
                st.write(f"{name} model download size: {size:,} bytes")
                model = joblib.load(BytesIO(r.content))
                classes = getattr(model, "classes_", None)
                st.success(f"Successfully loaded {name} model; classes: {classes}")
                return model
            except Exception as e:
                st.warning(f"Attempt {attempt} for {name} failed: {e}")
                last_exc = e
                time.sleep(1 * attempt)
        st.error(f"All attempts to fetch {name} model failed.")
        raise last_exc

    cal_sep = fetch_model("sepsis", sep_url)
    cal_non = fetch_model("non-sepsis", non_url)

    # quick sanity check with a contrived row (should not crash)
    dummy = pd.DataFrame(
        [
            {
                "age": 60,
                "charlson": 2,
                "sapsii": 20,
                "ventilation": "Unknown",
                "bun_mg_dl": 15,
                "creatinine_mg_dl": 1.0,
                "mechanically_ventilated": np.nan,
                "sofa_score": 5,
                "respiration": 18,
                "coagulation": 1,
                "liver": 1,
                "cardiovascular": 1,
                "cns": 1,
                "renal": 1,
            }
        ]
    )
    try:
        _ = cal_sep.predict(dummy)
        _ = cal_non.predict(dummy)
        st.success("Sanity-check prediction on both models succeeded.")
    except Exception:
        st.error("Sanity-check prediction failed. Details:")
        st.text(traceback.format_exc())

    return cal_sep, cal_non


# Load data & models with error capture
df = load_data()

try:
    model_sep, model_non = load_models()
    st.write("✅ Loaded both models.")
except Exception as e:
    st.error("Failed to load models; aborting."); st.text("".join(traceback.format_exception(e.__class__, e, e.__traceback__)))
    st.stop()

# --- feature definitions ---
FEATURES = [
    "age",
    "charlson",
    "sapsii",
    "ventilation",
    "bun_mg_dl",
    "creatinine_mg_dl",
    "mechanically_ventilated",
    "sofa_score",
    "respiration",
    "coagulation",
    "liver",
    "cardiovascular",
    "cns",
    "renal",
]
NUMERIC = [f for f in FEATURES if f not in ("ventilation", "mechanically_ventilated")]
BINARY = ["mechanically_ventilated"]
CATEGORICAL = ["ventilation"]

# --- UI ---
st.title("Multiclass LOS Classifier (Early Death, Short Stay, Long Stay)")
st.markdown(
    """
**Class definitions**  
- **early_death**: died in hospital < 10 days  
- **short_los**: stay < 10 days  
- **long_los**: stay ≥ 10 days  
"""
)

with st.form("patient_form"):
    inputs = {}
    for feat in FEATURES:
        label = feat.replace("_", " ").title()
        if feat in NUMERIC:
            inputs[feat] = st.number_input(label, value=float("nan"))
        elif feat in BINARY:
            inputs[feat] = st.selectbox(label, ["Unknown", "Yes", "No"])
        else:  # categorical
            options = ["Unknown"]
            if feat in df.columns:
                options += sorted(df[feat].dropna().unique().tolist())
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
                pat[feat] = 1.0
            elif val == "No":
                pat[feat] = 0.0
            else:
                pat[feat] = np.nan
        else:  # categorical
            pat[feat] = val if val != "Unknown" else np.nan

    pat_df = pd.DataFrame([pat])

    # Validate presence of expected features
    missing = [c for c in FEATURES if c not in pat_df.columns]
    if missing:
        st.error(f"Input is missing expected features: {missing}")
        st.stop()

    # Choose the appropriate model
    model = model_sep if sepsis_flag == "Yes" else model_non

    # Do prediction with error handling
    try:
        pred = model.predict(pat_df)[0]
        probs = model.predict_proba(pat_df)[0]
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.subheader("Input row that caused failure")
        st.write(pat_df)
        st.text("".join(traceback.format_exception(e.__class__, e, e.__traceback__)))
        st.stop()

    st.write(f"**Predicted class:** {pred}")
    st.write("**Class probabilities:**")
    proba_df = pd.DataFrame([probs], columns=model.classes_)
    st.table(proba_df.style.format("{:.2%}"))
