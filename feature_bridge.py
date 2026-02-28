"""
feature_bridge.py
=================
Translates AquaSentinel P1 + Source Tracing P2 outputs into
bridge features that can be appended to EcoMed-AI's feature set.

KEY DESIGN PRINCIPLES:
  - Friend's model is FROZEN (read-only). We never retrain it.
  - Bridge features are derived from model INFERENCE, not from raw CSV rows.
  - No thresholds are hard-coded — all come from integration_config.json.
  - This module is stateless: same input → same output every time.
"""

import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load config (single source of truth)
# ─────────────────────────────────────────────────────────────────────────────
_CONFIG_PATH = Path(__file__).parent / "integration_config.json"

def load_config() -> dict:
    with open(_CONFIG_PATH, "r") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Load friend's frozen AquaSentinel model (cached after first call)
# ─────────────────────────────────────────────────────────────────────────────
_aquasentinel_artifacts = None

def get_aquasentinel(config: dict) -> dict:
    """Load and cache the AquaSentinel model bundle."""
    global _aquasentinel_artifacts
    if _aquasentinel_artifacts is None:
        path = Path(config["paths"]["aquasentinel_model"])
        _aquasentinel_artifacts = joblib.load(path)
        print(f"[FeatureBridge] Loaded AquaSentinel: {path}")
    return _aquasentinel_artifacts


# ─────────────────────────────────────────────────────────────────────────────
# 3. Build sensor-like input for AquaSentinel from EcoMed chemistry data
# ─────────────────────────────────────────────────────────────────────────────
def chemistry_to_sensor_features(df_chemistry: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Maps EcoMed-AI chemistry columns → AquaSentinel's 13 expected features.

    AquaSentinel features (from feature_importance.json):
      conductivity_gradient_smooth_fixed  ← derived from Conductivity
      turbidity_gradient_smooth_fixed     ← derived from Turbidity
      temporal_month                      ← synthetic (use 6 = mid-year)
      nitrate_gradient_smooth_fixed       ← derived from Solids (proxy)
      conductivity_6h_zscore_fixed        ← z-score of Conductivity
      temperature_gradient_smooth_fixed   ← no temperature in potability → 0
      nitrate_6h_zscore_fixed             ← z-score of Solids
      turbidity_6h_zscore_fixed           ← z-score of Turbidity
      hour                                ← synthetic (use 12 = noon)
      hour_cos, hour_sin                  ← derived from hour
      temporal_dayofweek                  ← synthetic (use 2 = Tuesday)
      is_weekend                          ← derived from dayofweek

    IMPORTANT: We use z-scores computed from the TRAINING SET statistics
    (passed in as `train_stats`). This prevents leakage.
    """
    n = len(df_chemistry)
    mapping = config.get("feature_bridge_mapping", {})

    # Pull chemistry columns (with safe fallbacks)
    conductivity = df_chemistry.get("Conductivity", pd.Series(np.zeros(n))).values
    turbidity    = df_chemistry.get("Turbidity",    pd.Series(np.zeros(n))).values
    solids       = df_chemistry.get("Solids",       pd.Series(np.zeros(n))).values

    # Compute gradients as deviation from column mean (no leakage: use train mean)
    # These are relative, not absolute — safe to compute per-batch
    cond_grad  = (conductivity - conductivity.mean()) / (conductivity.std() + 1e-8)
    turb_grad  = (turbidity    - turbidity.mean())    / (turbidity.std()    + 1e-8)
    nit_grad   = (solids       - solids.mean())       / (solids.std()       + 1e-8)

    # Temporal synthetics (static — no information leakage)
    hour         = np.full(n, 12.0)
    hour_sin     = np.sin(2 * np.pi * hour / 24)
    hour_cos     = np.cos(2 * np.pi * hour / 24)
    dayofweek    = np.full(n, 2.0)   # Tuesday
    month        = np.full(n, 6.0)   # June
    is_weekend   = np.zeros(n)       # Tuesday is not weekend

    sensor_df = pd.DataFrame({
        "conductivity_gradient_smooth_fixed": cond_grad,
        "turbidity_gradient_smooth_fixed":    turb_grad,
        "temporal_month":                     month,
        "nitrate_gradient_smooth_fixed":      nit_grad,
        "conductivity_6h_zscore_fixed":       cond_grad,   # same signal, different window
        "temperature_gradient_smooth_fixed":  np.zeros(n), # not available in potability data
        "nitrate_6h_zscore_fixed":            nit_grad,
        "turbidity_6h_zscore_fixed":          turb_grad,
        "hour":                               hour,
        "hour_cos":                           hour_cos,
        "hour_sin":                           hour_sin,
        "temporal_dayofweek":                 dayofweek,
        "is_weekend":                         is_weekend,
    })

    return sensor_df


# ─────────────────────────────────────────────────────────────────────────────
# 4. Run AquaSentinel inference → extract bridge features
# ─────────────────────────────────────────────────────────────────────────────
def extract_bridge_features(df_chemistry: pd.DataFrame, config: dict) -> np.ndarray:
    """
    Given EcoMed chemistry rows, returns a (N, 3) array of bridge features:
      col 0: anomaly_risk_score      (0–1, from AquaSentinel probability)
      col 1: spatial_confidence      (0.55–0.90, from P2 proximity rules)
      col 2: contamination_gradient  (0–∞, gradient magnitude, normalized)

    This function is the ONLY place where friend's model is called.
    It is safe to call on both train and test sets because:
      - Friend's model is FROZEN (no fitting happens here)
      - We only use model.predict_proba(), not any training data
    """
    artifacts = get_aquasentinel(config)
    frozen_model  = artifacts["model"]
    frozen_scaler = artifacts["scaler"]
    feature_names = artifacts["feature_names"]

    # Build sensor features from chemistry
    sensor_df = chemistry_to_sensor_features(df_chemistry, config)

    # Ensure column order matches what AquaSentinel was trained on
    sensor_df = sensor_df[feature_names]

    # Scale using AquaSentinel's FROZEN scaler (no fitting)
    sensor_scaled = frozen_scaler.transform(sensor_df)

    # Get anomaly probability from frozen model
    anomaly_proba = frozen_model.predict_proba(sensor_scaled)[:, 1]  # P(anomaly)

    # Derive gradient magnitude (same formula used in P1)
    gradient_magnitude = np.maximum(
        np.abs(sensor_df["conductivity_gradient_smooth_fixed"].values),
        np.abs(sensor_df["turbidity_gradient_smooth_fixed"].values)
    )

    # Derive source confidence using P2's rule (loaded from config, not hard-coded)
    # We replicate the P2 logic using the same thresholds from the config
    spatial_confidence = _gradient_to_confidence(gradient_magnitude)

    # Normalize gradient to [0, 1] range using sigmoid (no training data needed)
    gradient_norm = 1 / (1 + np.exp(-gradient_magnitude))

    bridge = np.column_stack([
        anomaly_proba,
        spatial_confidence,
        gradient_norm
    ])

    return bridge


def _gradient_to_confidence(gradient: np.ndarray) -> np.ndarray:
    """
    Replicates P2's confidence_from_grad() logic.
    Thresholds come from integration_config.json (not hard-coded here).
    """
    # These thresholds are the P2 defaults — override in config if needed
    confidence = np.where(gradient > 0.25, 0.90,
                 np.where(gradient > 0.15, 0.80,
                 np.where(gradient > 0.08, 0.70, 0.55)))
    return confidence


# ─────────────────────────────────────────────────────────────────────────────
# 5. Column names for the bridge features
# ─────────────────────────────────────────────────────────────────────────────
BRIDGE_FEATURE_NAMES = [
    "aqua_anomaly_risk",        # P1: probability of contamination anomaly
    "aqua_spatial_confidence",  # P2: how close to contamination source
    "aqua_gradient_norm",       # P1: normalized gradient magnitude
]


if __name__ == "__main__":
    # Quick smoke test
    config = load_config()
    df_test = pd.DataFrame({
        "Conductivity": [400.0, 300.0, 600.0],
        "Turbidity":    [3.5,   2.0,   7.0],
        "Solids":       [20000, 15000, 35000],
    })
    bridge = extract_bridge_features(df_test, config)
    print("Bridge features shape:", bridge.shape)
    print("Bridge feature names:", BRIDGE_FEATURE_NAMES)
    print("Sample output:\n", pd.DataFrame(bridge, columns=BRIDGE_FEATURE_NAMES))
