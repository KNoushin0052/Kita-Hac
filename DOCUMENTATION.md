# EcoMed-AI: Integrated Water Safety Intelligence
## Technical Documentation & Hackathon Submission

---

## 1. Project Overview
EcoMed-AI is a **multi-regional water safety intelligence system** that integrates three distinct layers of analysis to detect contamination:
1.  **Chemical Safety (EcoMed-AI):** A Random Forest model analysing 20 chemical parameters (Arsenic, Lead, Bacteria, etc.) to predict potability. Works globally.
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
- **`train_on_waterquality1.py`**: The script used to train the primary EcoMed-AI model.
    - **Algorithm:** Random Forest Classifier (n_estimators=300)
    - **Accuracy:** 94.8% (on test set)
    - **Features:** 20 WHO-standard chemical parameters
    - **Handling Imbalance:** Uses `class_weight='balanced'` to handle the 89% unsafe / 11% safe imbalance.

---

## 5. Dataset Specifications

### Primary Training Data: `data/raw/waterQuality1.csv`
- **Rows:** 8,000 samples
- **Features:** aluminium, ammonia, arsenic, barium, cadmium, chloramine, chromium, copper, flouride, bacteria, viruses, lead, nitrates, nitrites, mercury, perchlorate, radium, selenium, silver, uranium.
- **Target:** `is_safe` (0/1)

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

3.  **Retrain the Model (Optional):**
    ```bash
    python train_on_waterquality1.py
    ```

---

## 7. Future Roadmap
- **IoT Integration:** Connect `feature_bridge.py` directly to MQTT streams from real water sensors.
- **Expanded GIS:** Add datasets for India (Ganga basin) and USA (Flint, MI) to the geospatial layer.
- **Mobile Alert:** Push notification system when `Safety Score < 0.35` (Critical).

---
*Generated for Hackathon Submission - 2026*
