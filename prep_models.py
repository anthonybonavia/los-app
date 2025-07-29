#!/usr/bin/env python3
# prep_models.py

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.calibration import CalibratedClassifierCV

# ── 1) Load data (suppress mixed‑type warnings) ───────────────────────────────
DATA_PATH = Path(__file__).parent / "my_table.csv"
if DATA_PATH.exists():
    df = pd.read_csv(DATA_PATH, low_memory=False)
else:
    df = pd.read_csv(
        "https://www.dropbox.com/scl/fi/8o976w9g9k3heeclbe8mb/my_table.csv?dl=1",
        low_memory=False
    )

# ── 2) Guarantee all FEATURES exist ───────────────────────────────────────────
FEATURES = [
    "age", "charlson", "sapsii", "ventilation",
    "bun_mg_dl", "creatinine_mg_dl", "mechanically_ventilated",
    "sofa_score", "respiration", "coagulation",
    "liver", "cardiovascular", "cns", "renal"
]
for col in FEATURES:
    if col not in df.columns:
        df[col] = np.nan

# ── 3) Map & cast key columns ────────────────────────────────────────────────
df["mechanically_ventilated"] = df["mechanically_ventilated"].map({
    "Yes": 1,
    "No": 0,
    "InvasiveVent": 1,
    "SupplementalOxygen": 0,
    np.nan: np.nan
}).astype(float)

df["sepsis3"] = df.get("sepsis3", 0).fillna(0).astype(int)

df["outcome"] = np.where(
    df.get("days_until_death", np.nan).notna() & (df.get("days_until_death", np.nan) < 10),
    "early_death",
    np.where(df.get("hospital_los_days", np.nan) >= 10, "long_los", "short_los")
)

# ── 4) Define feature groups ─────────────────────────────────────────────────
NUM = [f for f in FEATURES if f not in ("ventilation", "mechanically_ventilated")]
BIN = ["mechanically_ventilated"]
CAT = ["ventilation"]

# ── 5) Build preprocessing + classifier pipeline ──────────────────────────────
preprocessor = ColumnTransformer(
    transformers=[
        ("num", IterativeImputer(random_state=0), NUM),
        ("bin", SimpleImputer(strategy="most_frequent", fill_value=0), BIN),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("oh", OneHotEncoder(handle_unknown="ignore"))
        ]), CAT),
    ],
    remainder="drop"
)

pipeline = Pipeline([
    ("preproc", preprocessor),
    ("clf", RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=0
    ))
])

# ── 6) Train + calibrate for Sepsis+ and Sepsis− ──────────────────────────────
for label, fname in [(1, "sepsis"), (0, "nonsepsis")]:
    sub = df[df["sepsis3"] == label]
    X = sub[FEATURES]
    y = sub["outcome"]

    Xtr, Xte, ytr, yte = train_test_split(
        X, y,
        test_size=0.25,
        stratify=y,
        random_state=42
    )

    # Fit the base model
    model = pipeline.fit(Xtr, ytr)

    # Calibrate on hold‑out
    calibrator = CalibratedClassifierCV(
        estimator=model,
        cv="prefit",
        method="isotonic"
    )
    calibrator.fit(Xte, yte)

    # Create a classification report DataFrame
    report_df = pd.DataFrame(
        classification_report(yte, calibrator.predict(Xte), output_dict=True)
    ).T

    # Dump models **uncompressed** so mmap_mode works
    joblib.dump(calibrator, f"model_{fname}_calibrated.pkl", compress=0)
    joblib.dump(report_df, f"model_{fname}_report.pkl")

print("Models and reports saved.")
