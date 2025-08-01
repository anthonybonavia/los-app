import streamlit as st
import traceback, sys
import time

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
def fetch_with_logging(path, name, max_retries=2):
    for attempt in range(1, max_retries + 1):
        try:
            st.write(f"Attempting to download {name}, try {attempt}")
            r = requests.get(path, timeout=15)
            st.write(f"HTTP status for {name}: {r.status_code}")
            r.raise_for_status()
        except Exception as e:
            st.warning(f"Download attempt {attempt} for {name} failed: {e}")
            if attempt == max_retries:
                st.error(f"Giving up downloading {name} after {attempt} attempts.")
                raise
            time.sleep(1)
            continue
        try:
            model = joblib.load(BytesIO(r.content))
            st.success(f"Successfully loaded model from {name}; classes: {getattr(model, 'classes_', None)}")
            return model
        except Exception as e:
            st.error(f"Deserialization of {name} failed: {e}")
            raise

@st.cache_resource
def load_models():
    sep_url = "https://www.dropbox.com/scl/fi/puhntj2y9c4mv9k7fhmoo/model_sepsis_calibrated.pkl?rlkey=ty804av5nlg1ab8892u22w2xi&dl=1"
    non_url = "https://www.dropbox.com/scl/fi/c2oetaenrktyxe9kbj8nh/model_nonsepsis_calibrated.pkl?rlkey=repy9bvq99hl90bc4jwnhk3a3&dl=1"

    def fetch_and_load(name, url):
        for attempt in range(1, 4):
            st.write(f"Attempting to download {name} model, try {attempt}")
            try:
                r = requests.get(url, timeout=30)
                st.write(f"HTTP status for {name} model: {r.status_code}")
                r.raise_for_status()
                st.write(f"{name} model download size: {len(r.content):,} bytes")
                model = joblib.load(BytesIO(r.content))
                st.write(f"Successfully loaded model from {name} model; classes: {getattr(model, 'classes_', 'N/A')}")
                return model
            except Exception as e:
                st.warning(f"Failed to fetch/load {name} model on attempt {attempt}: {e}")
                if attempt < 3:
                    time.sleep(2 ** attempt)
                else:
                    raise
    cal_sep = fetch_and_load("sepsis", sep_url)
    cal_non = fetch_and_load("non-sepsis", non_url)
    
    st.write("Sepsis model classes:", cal_sep.classes_)
    st.write("Non-sepsis model classes:", cal_non.classes_)
    
    dummy = pd.DataFrame([{
    "age": 60, "charlson": 2, "sapsii": 20,
    "ventilation": "Unknown", "bun_mg_dl": 15, "creatinine_mg_dl": 1.0,
    "mechanically_ventilated": np.nan, "sofa_score": 5,
    "respiration": 18, "coagulation": 1, "liver": 1,
    "cardiovascular": 1, "cns": 1, "renal": 1
    }])
    try:
        _ = cal_sep.predict(dummy)
        _ = cal_non.predict(dummy)
        st.success("Sanity-check prediction on both models succeeded.")
    except Exception:
        st.error("Sanity-check prediction failed:")
        st.text(traceback.format_exc())

    
    return cal_sep, cal_non

try:
    model_sep, model_non = load_models()
    st.write("Successfully loaded both models.")
except Exception:
    st.error("load_models() raised an exception:")
    st.text(traceback.format_exc())
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

try:
    pred = model.predict(pat_df)[0]
    probs = model.predict_proba(pat_df)[0]
    st.write(f"**Predicted class:** {pred}")
    st.write("**Class probabilities:**")
    proba_df = pd.DataFrame([probs], columns=model.classes_)
    st.table(proba_df.style.format("{:.2%}"))
except Exception:
    st.error("Prediction failed with exception:")
    st.text(traceback.format_exc())
    st.write("Input row:", pat_df.to_dict(orient="records")[0])
    
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
