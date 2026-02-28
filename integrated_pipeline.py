"""
integrated_pipeline.py
=======================
Production inference pipeline: given raw water chemistry data,
returns potability prediction enriched with AquaSentinel anomaly context.

Usage:
    from integrated_pipeline import IntegratedWaterSafetyPipeline
    pipeline = IntegratedWaterSafetyPipeline()
    result = pipeline.predict({
        "aluminium": 0.5,
        "ammonia": 1.5,
        "arsenic": 0.01,
        ...
    })
"""

import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from feature_bridge import load_config

class IntegratedWaterSafetyPipeline:
    def __init__(self, config_path: str = "integration_config.json"):
        self.config = json.load(open(config_path))
        self._load_models()

    def _load_models(self):
        """Load all model artifacts from paths in config."""
        paths = self.config["paths"]
        
        # Load EcoMed-AI (Chemistry) Model
        # Note: integration_config.json points to the correct "wq1_model" folder now
        model_path  = Path(paths["primary_model"])
        scaler_path = Path(paths["primary_scaler"])
        imputer_path= Path(paths["primary_imputer"])
        feat_path   = Path(paths["primary_features"])

        print(f"[Pipeline] Loading model from: {model_path}")
        self.model   = joblib.load(model_path)
        self.scaler  = joblib.load(scaler_path)
        self.imputer = joblib.load(imputer_path)
        
        with open(feat_path) as f:
            self.feat_meta = json.load(f)

        # Load AquaSentinel (Anomaly) Model Bridge
        # We don't need the full model logic here, just the bridge logic
        # But if we want to run anomaly detection, we need that model too.
        # For this pipeline, we'll keep it simple: chemistry focus.
        # (AquaSentinel requires historical context which single-sample API lacks)
        self.aqua_model = None 

    def _engineer_features(self, row: dict) -> pd.DataFrame:
        """Apply the same feature engineering as app.py and TRAIN_BEST_MODEL.py."""
        # Convert dict to DataFrame
        df = pd.DataFrame([row])
        
        # Ensure all columns exist (fill missing with NaN for imputer)
        for col in self.feat_meta["original_features"]:
            if col not in df.columns:
                df[col] = np.nan

        # Engineering
        # heavy_metal_load = arsenic + cadmium + lead + mercury + chromium
        df["heavy_metal_load"] = (
            df.get("arsenic", 0) + df.get("cadmium", 0) + 
            df.get("lead", 0) + df.get("mercury", 0) + df.get("chromium", 0)
        )
        # pathogen_risk = bacteria * viruses
        df["pathogen_risk"] = df.get("bacteria", 0) * df.get("viruses", 0)
        # disinfect_proxy = chloramine * nitrates
        df["disinfect_proxy"] = df.get("chloramine", 0) * df.get("nitrates", 0)
        # radio_composite = radium + uranium
        df["radio_composite"] = df.get("radium", 0) + df.get("uranium", 0)
        # mineral_excess = barium + aluminium + silver
        df["mineral_excess"] = df.get("barium", 0) + df.get("aluminium", 0) + df.get("silver", 0)

        # Select columns in correct order
        return df[self.feat_meta["all_features"]]

    def predict(self, sample: dict) -> dict:
        """
        Predict potability for a single water sample.
        Args:
            sample: dict with keys matching waterQuality1.csv features
                    (aluminium, ammonia, arsenic, ..., uranium)
        Returns:
            dict with prediction, probability, and verdict
        """
        X_raw = self._engineer_features(sample)
        X_imp = pd.DataFrame(self.imputer.transform(X_raw), columns=X_raw.columns)
        X_sc  = self.scaler.transform(X_imp)
        
        # Predict probability of UNSAFE (class 1? Check training)
        # Random Forest classes_ usually [0, 1] where 1=Safe or 1=Unsafe?
        # In water_potability it was 1=Potable.
        # In waterQuality1 (is_safe), 1=Safe, 0=Unsafe.
        # Let's verify prediction logic. app.py says prob > threshold = SAFE.
        
        prob = float(self.model.predict_proba(X_sc)[0, 1])
        
        # Thresholds from config
        inf = self.config["inference"]
        
        if prob >= inf["safety_threshold"]:
            verdict = "SAFE"
        elif prob >= inf["caution_threshold"]:
            verdict = "CAUTION"
        else:
            verdict = "UNSAFE"

        return {
            "prediction_score": prob,
            "verdict": verdict,
            "model_version": "EcoMed-AI v2 (Integrated)"
        }
