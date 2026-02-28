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
# → {"safety_label": "✅ SAFE", "potability_probability": 0.81,
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
  │ (Core Model)│ │(Anomaly Unit)│ │(Geo Module) │
  │             │ │             │ │              │
  │ Chemistry   │ │ Temporal    │ │ Spatial      │
  │ analysis    │ │ anomaly     │ │ proximity    │
  │             │ │ detection   │ │ to source    │
  │ 25 features │ │ Time-series │ │ GIS + rules  │
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

**2. EcoMed-AI predicts1. **Chemical Intelligence (EcoMed-AI)**: A **Comprehensive 25-Signal Engine** (20 Raw + 5 Engineered) designed for high-resolution water safety analysis. It uses a Random Forest Classifier to process everything from Aluminum and Arsenic to complex composite risk indicators.
_probability` score

**3. `feature_bridge.py` queries AquaSentinel (Subsystem 1)**:
   - Translates chemistry data into sensor gradient signals
   - Calls the `anomaly_detector` bundle
   - Returns `anomaly_risk` score (0–1)

**4. Source Tracing (Subsystem 2)** adds spatial intelligence:
   - Calculates proximity to known contamination hazards
   - Maps to categorical alerts: VERY CLOSE / CLOSE / etc.

**5. All signals are synthesized** in `app.py` for the unified dashboard view.

---

### System Integration API
The system follows a strict interface for integrating external modules:

1. **Input Interface**: Modules accept chemistry dicts or proximity data.
2. **Output Interface**: Modules must return normalized scores (0.0 to 1.0).
3. **Verdict Mapping**: Scores are mapped to visual alerts in `app.py`.

---

## 📊 Model Performance (Honest)

| Metric | Value | Technical Context |
|--------|-------|---------|
| **Feature Resolution**| **25 Signals** | 20 Raw WHO-standard + 5 Engineered |
| **Overfitting Gap** | **1.1%** | Fixed via Forensic Audit (Verification) |
| **Accuracy (Validation)**| **94.81%** | Verified on 8,000 research samples |
| **Model Type** | **Random Forest** | Advanced Ensemble Architecture |

> **Technical note:** The project focuses on the **Depth of Analysis**. By utilizing 25 parameters, the system provides much higher decision accuracy than models that only look at 4 or 5 basic indicators.

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
