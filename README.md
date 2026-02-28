# 💧 EcoMed-AI — Integrated Water Safety System
### Hackathon Submission Guide

---

## 📁 Final Project Structure

```
EcoMed-AI/
│
├── 📄 README.md                    ← This file — start here
├── ⚙️  integration_config.json     ← All paths & thresholds (never hard-code)
│
├── 🐍 app.py                       ← ✅ DEMO — run this for the hackathon
├── 🐍 integrated_pipeline.py       ← Python API for your model
├── 🐍 feature_bridge.py            ← Connects EcoMed-AI ↔ AquaSentinel
│
├── 📂 data/
│   ├── raw/
│   │   ├── waterQuality1.csv       ← Primary dataset (7,996 samples, 20 features)
│   │   └── water_potability.csv    ← Legacy dataset (kept for reference)
│   └── processed/
│       ├── wq1_model/              ← ✅ PRIMARY MODEL (use this)
│       │   ├── model.pkl
│       │   ├── scaler.pkl
│       │   ├── imputer.pkl
│       │   └── feature_names.json
│       └── regularized_model/      ← Legacy model (kept for reference)
│
├── 📂 aquasentinel_complete_export/ ← Friend's P1: AquaSentinel anomaly detector
├── 📂 water_contaminant_ P1/        ← Friend's P1 (original export)
├── 📂 water_contaminant_source_P2/  ← Friend's P2: Source tracing
│
├── 📂 visualizations/              ← Charts & outputs
└── 📂 _archive_final/              ← All old files (ignore)
```

---

## 🚀 How to Run

### Step 1 — Activate the environment
```powershell
# From the EcoMed-AI folder:
.venv\Scripts\activate
```

### Step 2 — Launch the demo app
```powershell
streamlit run app.py
```
Opens at **http://localhost:8501** — this is your hackathon demo.

### Step 3 — Use the Python API directly
```python
from integrated_pipeline import IntegratedWaterSafetyPipeline

pipeline = IntegratedWaterSafetyPipeline()

result = pipeline.predict({
    "ph": 7.2, "Hardness": 150, "Solids": 18000,
    "Chloramines": 5, "Sulfate": 250, "Conductivity": 400,
    "Organic_carbon": 10, "Trihalomethanes": 60, "Turbidity": 3.0
})
print(result)
# → {"safety_label": "✅ SAFE", "potability_probability": 0.72,
#    "aqua_anomaly_risk": 0.12, "aqua_spatial_confidence": 0.55, ...}
```

### Step 4 — Test the feature bridge (AquaSentinel connection)
```powershell
python feature_bridge.py
```

---

## 🔗 How the Three Models Connect

This is the core of your hackathon story — **three systems, one decision**:

```
┌─────────────────────────────────────────────────────────────────┐
│                    WATER SAFETY DECISION                        │
└────────────────────────┬────────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
  │  EcoMed-AI  │ │AquaSentinel │ │Source Tracing│
  │  (YOUR MODEL)│ │ (Friend P1) │ │ (Friend P2)  │
  │             │ │             │ │              │
  │ Chemistry   │ │ Temporal    │ │ Spatial      │
  │ analysis    │ │ anomaly     │ │ proximity    │
  │             │ │ detection   │ │ to source    │
  │ 20 features │ │ Time-series │ │ GIS + rules  │
  │ RF model    │ │ VotingClf   │ │ Haversine    │
  └──────┬──────┘ └──────┬──────┘ └──────┬───────┘
         │               │               │
         ▼               ▼               ▼
   Safety score    Anomaly risk    Source proximity
   (0–1 prob)      (0–1 prob)      (CLOSE/FAR/etc)
         │               │               │
         └───────────────┴───────────────┘
                         │
                  feature_bridge.py
                  (translates between
                   the three systems)
                         │
                         ▼
              Combined interpretation
              shown in app.py dashboard
```

### The Integration Flow (step by step)

**1. User enters water chemistry readings** in the sidebar sliders.

**2. EcoMed-AI predicts** using `data/processed/wq1_model/model.pkl`:
   - Applies feature engineering (5 composite features)
   - Imputes missing values (using train-fitted imputer)
   - Scales features (using train-fitted scaler)
   - Returns `P(safe)` probability

**3. `feature_bridge.py` calls AquaSentinel (friend's P1)**:
   - Translates chemistry columns → sensor gradient features
   - Calls `frozen_model.predict_proba()` — never retrains it
   - Returns `anomaly_risk` score (0–1)

**4. Source Tracing (friend's P2)** is applied as a rule:
   - Heavy metal load (arsenic + cadmium + lead + mercury + chromium)
   - Maps to proximity: VERY CLOSE / CLOSE / MODERATE / DISTANT

**5. All three signals combine** in `app.py` for the final verdict.

---

## 🤝 How to Hand Off to Your Friend

Your friend needs to give you **one thing** to plug in their model:

### What your friend provides:
```
aquasentinel_complete_export/
└── aquasentinel_model/
    ├── anomaly_detector.pkl    ← the trained model bundle
    └── feature_importance.json ← feature names list
```

The `.pkl` file must contain a dict with these keys:
```python
{
    "model":         <VotingClassifier>,   # the trained model
    "scaler":        <StandardScaler>,     # fitted scaler
    "feature_names": [list of 13 strings], # exact feature order
    "metrics":       {...}                 # optional
}
```

### What you call on their model:
```python
# In feature_bridge.py — this is the ONLY place friend's model is called
artifacts = joblib.load("aquasentinel_complete_export/aquasentinel_model/anomaly_detector.pkl")
frozen_model  = artifacts["model"]
frozen_scaler = artifacts["scaler"]
feature_names = artifacts["feature_names"]

# Build the 13 sensor features from your chemistry data
sensor_df = chemistry_to_sensor_features(df_chemistry, config)
sensor_df = sensor_df[feature_names]          # ensure correct column order
scaled    = frozen_scaler.transform(sensor_df) # use THEIR scaler, not yours
anomaly_prob = frozen_model.predict_proba(scaled)[:, 1]  # P(anomaly)
```

### What you give your friend:
```python
# Your model as a simple function they can call:
from integrated_pipeline import IntegratedWaterSafetyPipeline
pipeline = IntegratedWaterSafetyPipeline()

# They pass chemistry readings, you return a safety score
result = pipeline.predict(their_sample_dict)
# result["potability_probability"]  → float 0–1
# result["safety_label"]            → "✅ SAFE" or "⚠️ UNSAFE"
# result["aqua_anomaly_risk"]       → float 0–1 (from their model)
```

---

## 📊 Model Performance (Honest)

| Metric | Value | Context |
|--------|-------|---------|
| Test Accuracy | **94.81%** | On `waterQuality1.csv` (synthetic dataset) |
| ROC-AUC | **0.9808** | Excellent discrimination |
| Overfitting Gap | **1.1%** | Well-generalised |
| Unsafe Recall | **95.8%** | Catches 95.8% of unsafe water |
| F1 Score (safe class) | **79.2%** | Good despite 11% class imbalance |

> **Hackathon framing:** Lead with the **architecture** (3-system integration),
> not the accuracy number. The integration is the innovation.
> If judges ask about accuracy: *"94.8% on our benchmark — but the real value
> is the multi-signal approach: chemistry alone misses temporal spikes and
> spatial proximity that AquaSentinel and Source Tracing catch."*

---

## ⚙️ Configuration

Everything is in `integration_config.json` — change paths or thresholds here:

```json
{
  "paths": {
    "primary_model":   "data/processed/wq1_model/model.pkl",
    "aquasentinel_model": "aquasentinel_complete_export/..."
  },
  "inference": {
    "safety_threshold":  0.50,   ← raise to be more conservative
    "caution_threshold": 0.35,   ← below this = UNSAFE verdict
    "anomaly_risk_high": 0.70    ← AquaSentinel alarm level
  }
}
```

---

## 🛠️ Install Dependencies

```powershell
.venv\Scripts\pip install streamlit scikit-learn pandas numpy joblib
```
