# EcoMed-AI — Dataset Specification
## What Any Input Dataset Must Contain

---

## For INFERENCE (running predictions on new water samples)

You only need the 20 chemistry columns. No labels, no GPS required.

| Column       | Unit    | WHO/EPA Limit | Notes                                      |
|--------------|---------|---------------|--------------------------------------------|
| aluminium    | mg/L    | 0.2 (WHO)     | Elevated in industrial areas               |
| ammonia      | mg/L    | 1.5 (WHO)     | Sewage indicator                           |
| arsenic      | mg/L    | 0.01 (WHO)    | ⚠️ Critical for Bangladesh groundwater     |
| barium       | mg/L    | 0.7 (WHO)     | Industrial/natural geology                 |
| cadmium      | mg/L    | 0.003 (WHO)   | Industrial pollution                       |
| chloramine   | mg/L    | 5.0 (WHO)     | Disinfection byproduct                     |
| chromium     | mg/L    | 0.05 (WHO)    | Industrial waste                           |
| copper       | mg/L    | 2.0 (WHO)     | Pipe corrosion                             |
| flouride     | mg/L    | 1.5 (WHO)     | Natural geology / dental programs          |
| bacteria     | CFU/mL  | 0 (WHO)       | Must be zero for safe drinking water       |
| viruses      | PFU/mL  | 0 (WHO)       | Must be zero for safe drinking water       |
| lead         | mg/L    | 0.01 (WHO)    | Old pipe infrastructure                    |
| nitrates     | mg/L    | 50 (WHO)      | Agricultural runoff                        |
| nitrites     | mg/L    | 3.0 (WHO)     | Sewage / agricultural                      |
| mercury      | mg/L    | 0.006 (WHO)   | Industrial / mining                        |
| perchlorate  | mg/L    | 15 (EPA)      | Rocket fuel / fertiliser contamination     |
| radium       | pCi/L   | 5 (EPA)       | Natural radioactivity                      |
| selenium     | mg/L    | 0.04 (WHO)    | Natural geology / mining                   |
| silver       | mg/L    | 0.1 (EPA)     | Colloidal silver / industrial              |
| uranium      | mg/L    | 0.015 (WHO)   | Natural geology / mining                   |

### Minimum CSV format for batch inference:
```csv
aluminium,ammonia,arsenic,barium,cadmium,chloramine,chromium,copper,flouride,bacteria,viruses,lead,nitrates,nitrites,mercury,perchlorate,radium,selenium,silver,uranium
0.5,1.5,0.01,0.7,0.003,2.0,0.05,2.0,1.5,0.0,0.0,0.01,5.0,0.1,0.001,4.0,0.1,0.01,0.1,0.015
```

---

## For TRAINING (retraining the model on new country data)

Same 20 columns PLUS:

| Column   | Type    | Values | Notes                          |
|----------|---------|--------|--------------------------------|
| is_safe  | integer | 0 or 1 | 0 = Unsafe, 1 = Safe           |

### Optional but useful for richer datasets:
| Column      | Notes                                              |
|-------------|----------------------------------------------------|
| latitude    | For spatial analysis / friend's P2 integration     |
| longitude   | For spatial analysis / friend's P2 integration     |
| timestamp   | For temporal anomaly detection / friend's P1       |
| source_type | e.g. "groundwater", "tap", "river", "well"         |
| country     | For multi-country models                           |

---

## Geographic Constraints — What Works Where

### ✅ Your EcoMed-AI model works GLOBALLY
The model only sees chemical concentrations — no GPS, no country code.
Any water sample from any country with these 20 measurements can be predicted.

### ⚠️ Friend's AquaSentinel (P1) — Partially global
The anomaly detection uses temporal gradients (conductivity, turbidity, nitrate changes over time).
These are physical properties that exist everywhere. However, the model was trained on Dubai's
sensor network, so anomaly thresholds may be calibrated to that network's baseline.

### ❌ Friend's Source Tracing (P2) — Dubai only
The pipe infrastructure (zones, blocks, pipes) is physically located in Dubai.
The Haversine distance calculations reference Dubai sensor coordinates.
To use P2 in Bangladesh, your friend would need a Bangladesh water network dataset
with GPS-tagged pipe sensors.

---

## Real-World Datasets You Could Use for Bangladesh

| Dataset | Source | URL |
|---------|--------|-----|
| Bangladesh DPHE Water Quality | Dept of Public Health Engineering | https://dphe.gov.bd |
| WHO/UNICEF JMP Bangladesh | Joint Monitoring Programme | https://washdata.org |
| USGS Global Groundwater | US Geological Survey | https://waterdata.usgs.gov |
| World Bank WASH Data | World Bank Open Data | https://data.worldbank.org |
| Bangladesh Arsenic Crisis Data | British Geological Survey | https://www.bgs.ac.uk/research/groundwater/health/arsenic/Bangladesh |

### Why Bangladesh specifically matters:
Bangladesh has one of the world's worst arsenic contamination crises in groundwater.
~20 million people drink water with arsenic > WHO limit (0.01 mg/L).
Your model's arsenic feature would be the single most important predictor for Bangladesh data.

---

## How to Add a New Country Dataset

1. Collect water quality measurements in the 20-column format above
2. Add `is_safe` labels (0/1) based on WHO limits or local standards
3. Place CSV in `data/raw/your_country_water.csv`
4. Run `train_on_waterquality1.py` (modify the data path at the top)
5. New model saved to `data/processed/wq1_model/` — ready for inference

The integration pipeline (`integrated_pipeline.py`, `feature_bridge.py`) requires no changes.
