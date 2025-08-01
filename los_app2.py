import streamlit as st
import traceback, sys

try:
    # existing imports and code follow here...
    import joblib
    import pandas as pd
    import numpy as np
    import pandas as pd
    import requests
    from io import BytesIO
except Exception as e:
    st.error("Startup failure—see details below.")
    st.text("".join(traceback.format_exception(e.__class__, e, e.__traceback__)))
    sys.exit(1)
    
#comment
# --- diagnostics (optional) ---
st.write("Python:", sys.version.replace("\n", " "))
try:
    import numpy as np_mod

    st.write("Installed numpy:", np_mod.__version__)
except ImportError:
    st.write("numpy not installed")
try:
    import sklearn

    st.write("Installed scikit-learn:", sklearn.__version__)
except ImportError:
    st.write("scikit-learn not installed")
try:
    import joblib as _joblib

    st.write("Installed joblib:", _joblib.__version__)
except ImportError:
    st.write("joblib not installed")

# --- 1) Load patient reference data for dropdowns ---
@st.cache_data
def load_data():
    url = (
        "https://www.dropbox.com/scl/fi/8o976w9g9k3heeclbe8mb/my_table.csv?rlkey=0114ysndn9aa27d15j2nd36je&dl=1"
    )
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


# --- 2) Load prebuilt models from Dropbox ---
@st.cache_resource
def load_models():
    sep_url = "https://www.dropbox.com/scl/fi/puhntj2y9c4mv9k7fhmoo/model_sepsis_calibrated.pkl?rlkey=ty804av5nlg1ab8892u22w2xi&dl=1"
    non_url = "https://www.dropbox.com/scl/fi/c2oetaenrktyxe9kbj8nh/model_nonsepsis_calibrated.pkl?rlkey=repy9bvq99hl90bc4jwnhk3a3&dl=1"

    def fetch_and_load(path, name):
        st.write(f"Attempting to download model ({name}) from {path}")
        st.write(f"{name} download size: {len(r.content)} bytes")
        if b"<!DOCTYPE html" in r.content[:200].lower():
            st.error(f"{name} appears to be HTML rather than a pickle; content preview: {r.content[:500]!r}")

        try:
            r = requests.get(path, timeout=15)
            st.write(f"HTTP status for {name}: {r.status_code}")
            r.raise_for_status()
        except Exception as e:
            st.error(f"Failed to download {name}: {e}")
            raise
        try:
            model = joblib.load(BytesIO(r.content))
            st.write(f"Successfully loaded model {name}; classes: {getattr(model, 'classes_', 'UNKNOWN')}")
            return model
        except Exception as e:
            st.error(f"Failed to deserialize {name}: {e}")
            import traceback
            st.text(traceback.format_exc())
            raise

    cal_sep = fetch_and_load(sep_url, "sepsis")
    cal_non = fetch_and_load(non_url, "nonsepsis")
    return cal_sep, cal_non



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


# --- load once ---
df = load_data()
try:
    model_sep, model_non = load_models()
except Exception:
    st.stop()

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

st.subheader("Predict an Individual Patient Outcome")
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

    # Check for missing required columns
    missing = [c for c in FEATURES if c not in pat_df.columns]
    if missing:
        st.error(f"Input is missing expected features: {missing}")
        st.stop()

    # Choose model
    model = model_sep if sepsis_flag == "Yes" else model_non

    # Prediction with error handling
    try:
        pred = model.predict(pat_df)[0]
        probs = model.predict_proba(pat_df)[0]
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.write("Input row:")
        st.write(pat_df)
        st.stop()

    st.write(f"**Predicted class:** {pred}")
    st.write("**Class probabilities:**")
    proba_df = pd.DataFrame([probs], columns=model.classes_)
    st.table(proba_df.style.format("{:.2%}"))
