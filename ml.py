"""
Machine-learning pipeline (post-encoding-overhaul).

This version loads ONE artifact: a complete sklearn Pipeline created by
retrain_model.py. The pipeline contains:

    ColumnTransformer ─ OrdinalEncoder (with learned categories)
            ↓
    StandardScaler
            ↓
    Classifier (winner model, e.g. SVM)

Why this is a real improvement
------------------------------
1. Encoding matches the physical reality of each feature.
2. Case mismatches (e.g. "moderate" vs "Moderate") are normalized.
3. No hidden mode-fill behaviour — only Boruta-selected features are used.
4. One artifact instead of three — encoder, scaler, classifier bundled together.
"""

import os
import joblib
import pandas as pd

from config import Config

# ---------------------------------------------------------------------------
# Load artifacts at import time.
# ---------------------------------------------------------------------------
def _safe_load(path: str, name: str):
    """Best-effort loader; returns None on failure rather than crashing import."""
    try:
        return joblib.load(path)
    except Exception as exc:  # noqa: BLE001
        print(f"[ml] WARNING: could not load {name} from {path}: {exc}")
        return None

PIPELINE_PATH = os.path.join(Config.ARTIFACTS_DIR, "pipeline.pkl")
METADATA_PATH = os.path.join(Config.ARTIFACTS_DIR, "encoding_metadata.pkl")
FEATURES_PATH = os.path.join(Config.ARTIFACTS_DIR, "selected_features.pkl")

pipeline           = _safe_load(PIPELINE_PATH,           "pipeline")
selected_features  = _safe_load(FEATURES_PATH,           "selected_features")
encoding_metadata  = _safe_load(METADATA_PATH,           "encoding_metadata")

# Dataset is still useful for the Feature Guide page.
try:
    dataset = pd.read_csv(Config.DATASET_PATH)
    if "Elevation" in dataset.columns:
        dataset["Elevation"] = dataset["Elevation"].str.strip().replace({"moderate": "Moderate"})
except Exception as exc:  # noqa: BLE001
    print(f"[ml] WARNING: could not load dataset: {exc}")
    dataset = pd.DataFrame()

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------
def is_ready() -> bool:
    """All artifacts loaded?"""
    return (
        pipeline is not None
        and selected_features is not None
        and encoding_metadata is not None
    )

def predictor_options() -> dict:
    """
    Return {feature: [options]} for each Boruta-selected feature.
    Ordered features → physical order.
    Unordered features → alphabetical order.
    """
    out = {}
    if not is_ready():
        return out

    ordered_meta   = encoding_metadata.get("ordered",   {})
    unordered_meta = encoding_metadata.get("unordered", {})

    for feature in selected_features:
        if feature in ordered_meta:
            out[feature] = list(ordered_meta[feature])
        elif feature in unordered_meta:
            out[feature] = list(unordered_meta[feature])
        else:
            out[feature] = (
                sorted(dataset[feature].dropna().unique().tolist())
                if feature in dataset.columns else []
            )
    return out

def feature_guide() -> list:
    """
    For the Feature Guide page. Returns:
        [
          (feature_name, kind, [values]),
          ...
        ]
    """
    out = []
    if dataset.empty:
        return out

    ordered_meta   = encoding_metadata.get("ordered",   {}) if encoding_metadata else {}
    unordered_meta = encoding_metadata.get("unordered", {}) if encoding_metadata else {}

    for col in dataset.columns:
        if col == "Decision":
            continue
        if col in ordered_meta:
            out.append((col, "ordered", list(ordered_meta[col])))
        elif col in unordered_meta:
            out.append((col, "unordered", list(unordered_meta[col])))
        else:
            try:
                vals = sorted(dataset[col].dropna().unique().tolist())
            except Exception:  # noqa: BLE001
                vals = []
            out.append((col, "unordered", vals))
    return out

def predict(user_inputs: dict, geo_features: dict = None) -> dict:
    """
    Run the prediction pipeline.
    Args:
        user_inputs: {feature_name: chosen_value} for each selected feature.
        geo_features: optional shapefile attributes (future integration).
    Returns:
        {
            "prediction":         0 or 1,
            "label":              str,
            "high_potential_pct": float,
            "low_potential_pct":  float,
            "model_used":         str,
        }
    """
    if not is_ready():
        raise RuntimeError(
            "Model artifacts are not all loaded. Run `python retrain_model.py` "
            "to (re-)create them."
        )

    missing = [f for f in selected_features if f not in user_inputs]
    if missing:
        raise ValueError(f"Missing required feature(s): {missing}")

    row = {f: user_inputs[f] for f in selected_features}
    df  = pd.DataFrame([row], columns=selected_features)

    pred  = int(pipeline.predict(df)[0])
    probs = pipeline.predict_proba(df)[0]   # [low_prob, high_prob]

    return {
        "prediction":         pred,
        "label":              "High Potential Area" if pred == 1 else "Low Potential Area",
        "high_potential_pct": float(probs[1] * 100.0),
        "low_potential_pct":  float(probs[0] * 100.0),
        "model_used":         encoding_metadata.get("winner", "model"),
    }
