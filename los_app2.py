import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from io import BytesIO
import sys, os, traceback

# ---------- Environment diagnostics ----------
st.write(f"Python: {sys.version.replace(chr(10),' ')}")
try:
    import numpy as _np
    st.write("Installed numpy:", _np.__version__)
except Exception:
    st.write("numpy import failed")
try:
    import sklearn as _sk
    st.write("Installed scikit-learn:", _sk.__version__)
except Exception:
    st.write("scikit-learn import failed")
try:
    import joblib as _jl
    st.write("Installed joblib:", _jl.__version__)
except Exception:
    st.write("joblib import failed")

st.write("")  # spacer

# ---------- Model loading with diagnostics ----------
MODEL_URLS = {
    "sepsis": "https://www.dropbox.com/scl/fi/puhntj2y9c4mv9k7fhmoo/model_sepsis_calibrated.pkl?rlkey=ty804av5nlg1ab8892u22w2xi&dl=1",
    "nonsepsis": "https://www.dropbox.com/scl/fi/c2oetaenrktyxe9kbj8nh/model_nonsepsis_calibrated.pkl?rlkey=repy9bvq99hl90bc4jwnhk3a3&dl=1",
}

@st.cache_resource
def load_models(max_retries=3):
    loaded = {}
    for key, url in MODEL_URLS.items():
        st.write(f"Attempting to download {key} model, up to {max_retries} tries")
        last_exc = None
        for attempt in range(1, max_retries + 1):
            try:
                r = requests.get(url, timeout=30)
                st.write(f"HTTP status for {key} model (try {attempt}): {r.status_code}")
                r.raise_for_status()
                content = r.content
                st.write(f"{key} model download size: {len(content):,} bytes")
                # Quick sanity check: if it looks like HTML, fail early
                prefix = content[:100].lower()
                if prefix.startswith(b"<") or b"html" in prefix:
                    snippet = content[:500].decode(errors="ignore")
                    raise RuntimeError(f"Downloaded content for {key} model appears to be HTML or invalid. Prefix/snippet:\n{snippet!r}")
                # Try to load
                model = joblib.load(BytesIO(content))
                st.write(f"Successfully loaded model from {key} model; classes: {getattr(model, 'classes_', None)}")
                loaded[key] = model
                break
            except Exception as e:
                st.warning(f"Failed to fetch/load {key} model on attempt {attempt}: {e}")
                last_exc = e
        else:
            raise RuntimeError(f"Could not load {key} model after {max_retries} attempts. Last error: {last_exc}")
    return loaded["sepsis"], loaded["nonsepsis"]


# Wrap load with visible errors
try:
    model_sep, model_non = load_models()
except Exception as e:
    st.error("load_models() raised an exception:") 
    st.text("".join(traceback.format_exception(e.__class__, e, e.__traceback__)))
    st.stop()

# ---------- Feature definitions ----------
FEATURES = [
    "age", "charlson", "sapsii", "ventilation",
    "bun_mg_dl", "creatinine_mg_dl", "mechanically_ventilated",
    "sofa_score", "respiration", "coagulation",
    "liver", "cardiovascular", "cns", "renal"
]
NUMERIC = [f for f in FEATURES if f not in ("ventilation", "mechanically_ventilated")]
BINARY = ["mechanically_ventilated"]
CATEGORICAL = ["ventilation"]

# ---------- UI ----------
st.title("Multiclass LOS Classifier (Early Death, Short Stay, Long Stay)")

st.markdown("""
**Class definitions**  
- **early_death**: died in hospital < 10 days  
- **short_los**: stay < 10 days  
- **long_los**: stay ≥ 10 days  
""")

st.subheader("Predict an Individual Patient Outcome")

# Dummy dataframe for dropdowns: minimal fallback if no reference CSV
@st.cache_data
def load_reference_df():
    # This was previously loading a CSV; if you have one, replace the URL accordingly.
    # For now create minimal stub so dropdowns have something.
    return pd.DataFrame({
        "ventilation": ["Unknown", "InvasiveVent", "SupplementalOxygen"]
    })

df = load_reference_df()

with st.form("patient_form"):
    inputs = {}
    for feat in FEATURES:
        label = feat.replace("_", " ").title()
        if feat in NUMERIC:
            inputs[feat] = st.number_input(label, value=float("nan"))
        elif feat in BINARY:
            inputs[feat] = st.selectbox(label, ["Unknown", "Yes", "No"])
        else:  # categorical
            options = ["Unknown"] + sorted(df.get(feat, pd.Series([])).dropna().unique().tolist())
            inputs[feat] = st.selectbox(label, options)
    sepsis_flag = st.selectbox("Meets Sepsis-3 Criteria?", ["Yes", "No"])
    submitted = st.form_submit_button("Compute")

if submitted:
    # Assemble patient record
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
        else:
            pat[feat] = val if val != "Unknown" else np.nan

    pat_df = pd.DataFrame([pat])

    st.write("Input patient dataframe (raw):")
    st.dataframe(pat_df)

    # Check for required columns and NaNs
    missing = [c for c in FEATURES if c not in pat_df.columns]
    if missing:
        st.error(f"Missing expected features: {missing}")
        st.stop()

    # Show summary of missingness
    st.write("Input missingness per feature:")
    st.table(pat_df.isna().T.astype(int).rename(columns={0: "is_missing"}))

    # Select model
    model = model_sep if sepsis_flag == "Yes" else model_non
    st.write(f"Using model for {'sepsis' if sepsis_flag == 'Yes' else 'non-sepsis'}; classes: {model.classes_}")

    # Prediction with guard
    try:
        pred = model.predict(pat_df)[0]
        probs = model.predict_proba(pat_df)[0]
    except Exception as e:
        st.error("Prediction failed:")
        st.text("".join(traceback.format_exception(e.__class__, e, e.__traceback__)))
        st.stop()

    st.success(f"**Predicted class:** {pred}")
    st.write("**Class probabilities:**")
    proba_df = pd.DataFrame([probs], columns=model.classes_)
    st.table(proba_df.style.format("{:.2%}"))
