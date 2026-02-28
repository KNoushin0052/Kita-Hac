# EcoMed-AI: Integrated Water Safety Intelligence
## Technical Specification & Architectural Overview

---

## 1. Project Overview
EcoMed-AI is a **multi-regional water safety intelligence system** that integrates three distinct layers of analysis to detect contamination:
1.  **Chemical Safety (EcoMed-AI):** A rigorous, audited **Random Forest Classifier** achieving **81.42% accuracy** on the standard Water Potability Study. We intentionally regularized this model to eliminate technical "leakage" found in higher-scoring synthetic benchmarks.
2.  **Anomaly Detection (AquaSentinel):** A temporal model detecting sudden spikes in sensor readings (conductivity/turbidity gradients).
3.  **Source Tracing Layer:** A geospatial module that maps contamination to specific pipe infrastructure (in Dubai) or groundwater districts (in Bangladesh).

---

## 2. System Architecture

The system is built on a **modular architecture** where independent models communicate via a central bridge:

```
[ User Input / Sensors ]
       │
       ▼
[ feature_bridge.py ] ───────▶ [ AquaSentinel Model ]
       │                                  │
       │ (Feature Engineering)            ▼ (Anomaly Score)
       │
       ▼
[ EcoMed-AI Model ] ◀────────── [ Geospatial Context Module ]
       │
       ▼
[ integrated_pipeline.py ] ───▶ [ Final Safety Verdict ]
       │
       ▼
[ app.py (Streamlit Dashboard) ] ───▶ [ Live Map & Alerts ]
```

---

## 3. Global Adaptability (The "Generalisation" Feature)
A key innovation is the system's ability to adapt its visualization based on region, while keeping the core chemistry engine universal.

| Region | Data Source | Visualization Layer |
| :--- | :--- | :--- |
| **Global (Core)** | `waterQuality1.csv` | **Chemical Safety Score** (Universal) |
| **Dubai, UAE** | `infrastructure_sensors.csv` | **Pipe Network & Leak Detection** (Urban Context) |
| **Bangladesh** | `bangladesh_water.csv` | **Groundwater Arsenic & Bacteria Risk** (Regional Context) |

---

## 4. Key Files & Scripts

### 🟢 Core Application
- **`app.py`**: The main entry point. Runs the Streamlit dashboard with the 3-tab interface (Safety, Map, WHO Limits).
- **`integrated_pipeline.py`**: The Python API that orchestrates the models. Can be used headless (without UI).
- **`integration_config.json`**: Central configuration file for all paths, thresholds, and hyperparameters.

### 🟡 Integration Logic
- **`feature_bridge.py`**: The "Glue Code". Translates basic chemistry inputs into the complex time-series features expected by the external AquaSentinel model.
- **`generate_bangladesh_data.py`**: a script that generates realistic groundwater data based on BGS/WHO statistics for the Bangladesh demo.

### 🔴 Training & Modeling
- **`train_regularized_model.py`**: The script used to train the primary "Honest Logic" model.
    - **Algorithm:** Random Forest Classifier (Pruned)
    - **Accuracy:** 81.42% (Validated, Leakage-Free)
    - **Audit:** We rejected a 94.8% synthetic model for failing our Forensic Audit (data leakage). This 81% model is the one we stand behind for real-world safety.

---

## 5. Dataset Specifications

### Primary Training Data: `data/raw/water_potability.csv`
- **Rows:** 3,276 samples (Kaggle Research Standard)
- **Features:** pH, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity.
- **Why this dataset?** It provided the most realistic challenge for regularization and feature engineering, which we used to prove our system's reliability.

### Regional Data:
- **`bangladesh_water.csv`**: 2,000 samples simulating Bangladesh groundwater, with high arsenic in Comilla/Chandpur and high salinity in coastal areas.
- **`aquasentinel/`**: Friend's Dubai pipe sensor dataset.

---

## 6. How to Run

1.  **Install Requirements:**
    ```bash
    pip install streamlit folium streamlit-folium pandas numpy scikit-learn joblib
    ```

2.  **Run the App:**
    ```bash
    streamlit run app.py
    ```

3.  **Retrain the Model (Audit Mode):**
    ```bash
    python train_regularized_model.py
    ```

---

## 7. Future Roadmap
- **IoT Integration:** Connect `feature_bridge.py` directly to MQTT streams from real water sensors.
- **Expanded GIS:** Add datasets for India (Ganga basin) and USA (Flint, MI) to the geospatial layer.
- **Mobile Alert:** Push notification system when `Safety Score < 0.35` (Critical).

---
*Production-Grade Technical Documentation*
