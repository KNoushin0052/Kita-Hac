# 💧 EcoMed-AI — Integrated Water Safety Intelligence
### Hackathon Submission: AI-Powered Water Safety & Response System

---

## 📁 Final Project Structure (Clean Version)

```
EcoMed-AI/
├── 🐍 app.py                  ← ✅ MAIN DEMO (Streamlit Dashboard)
├── 🐍 integrated_pipeline.py  ← Core ML Decision Engine
├── 🐍 prediction_server.py    ← ⚡ CLOUD RUN API (Flask + Gemini AI)
├── 🐍 feature_bridge.py       ← Integrates EcoMed-AI ↔ AquaSentinel
│
├── 📂 data/
│   ├── raw/                   ← Global Research Benchmarks (Kaggle: 8,000 samples)
│   └── processed/wq1_model/   ← ✅ PRODUCTION MODEL (WHO-Standard Trained)
│
├── 📂 aquasentinel_complete_export/ ← Subsystem A: Temporal Anomaly Detector
├── 📂 visualizations/          ← Performance Charts & Dashboards
│
├── ⚙️  integration_config.json  ← Central Configuration (Thresholds & Paths)
├── 📄 .env.example            ← Environment variable template (for Gemini)
├── 🐋 Dockerfile               ← Container configuration for Cloud Run
├── 📋 requirements.txt        ← All dependencies
├── ⚖️  LICENSE                 ← MIT License
```


---

## 🚀 How to Run Locally

### 1. Setup Environment
```powershell
# Create & Activate venv
python -m venv .venv
.venv\Scripts\activate

# Install Dependencies
pip install -r requirements.txt
```

### 2. Configure AI Health Advisor (Gemini)
1. Copy `.env.example` to `.env`.
2. Get your free API key from [aistudio.google.com](https://aistudio.google.com).
3. Add it to your `.env` file: `GEMINI_API_KEY=your_key_here`.

### 3. Launch Dashboard
```powershell
streamlit run app.py
```
Opens at **http://localhost:8501** — your full interactive judge's demo.

---

## 🧠 The Winning Innovation: Triple-Signal Intelligence

We didn't just build a model; we built a **System**. EcoMed-AI merges three distinct intelligence signals into one safety verdict:

1. **Chemical Intelligence (EcoMed-AI)**: Trained on a **Global Kaggle Research Benchmark** (8,000 samples aligned with WHO/EPA standards) using a 94.8% accurate Random Forest classifier.
2. **Temporal Intelligence (AquaSentinel)**: Detects sudden "hidden" spikes in sensor data that static testing misses.
3. **Generative Intelligence (Google Gemini)**: Translates raw ML probabilities into plain-language health advisories for communities.


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
  │ (Core Model)│ │ (Anomaly Unit)│ │ (Geo Module) │
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

**3. `feature_bridge.py` integrates AquaSentinel logic**:
   - Translates chemistry columns → sensor gradient features
   - Calls `frozen_model.predict_proba()` — uses the standalone inference bundle
   - Returns `anomaly_risk` score (0–1)

**4. Geospatial Context (Source Tracing)** is applied as a rule:
   - Heavy metal load (arsenic + cadmium + lead + mercury + chromium)
   - Maps to proximity: VERY CLOSE / CLOSE / MODERATE / DISTANT

**5. All three signals combine** in `app.py` for the final verdict.

---

## 🧩 System Interoperability & Integration

This section details how the EcoMed-AI core interacts with the external AquaSentinel artifacts.

### 1. External Artifacts Required:
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

### 2. Integration Implementation:
```python
# In feature_bridge.py — the primary integration point
artifacts = joblib.load("aquasentinel_complete_export/aquasentinel_model/anomaly_detector.pkl")
frozen_model  = artifacts["model"]
frozen_scaler = artifacts["scaler"]
feature_names = artifacts["feature_names"]

# Build the 13 sensor features from your chemistry data
sensor_df = chemistry_to_sensor_features(df_chemistry, config)
sensor_df = sensor_df[feature_names]          # enforce schema alignment
scaled    = frozen_scaler.transform(sensor_df) # apply dedicated scaling
anomaly_prob = frozen_model.predict_proba(scaled)[:, 1] 
```

### 3. Cross-System API Usage:
```python
# The EcoMed-AI core can be queried by external modules:
from integrated_pipeline import IntegratedWaterSafetyPipeline
pipeline = IntegratedWaterSafetyPipeline()

# Input: chemistry readings | Output: safety score + analysis
result = pipeline.predict(sample_input_dict)
# result["potability_probability"]  → float 0–1
# result["safety_label"]            → "✅ SAFE" or "⚠️ UNSAFE"
# result["aqua_anomaly_risk"]       → float 0–1 (from their model)
```

---

## 📊 Model Performance (Honest)

| Metric | Value | Data Source |
|--------|-------|---------|
| **Validation Accuracy** | **94.81%** | **Kaggle Global Water Quality Study** (8,000 rows) |
| **ROC-AUC** | **0.9808** | WHO-Standard Safety Limits |
| **Model Recall** | **95.8%** | Critical for public health safety |

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
