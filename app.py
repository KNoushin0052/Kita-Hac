"""
app.py  —  EcoMed-AI Hackathon Demo
=====================================
Streamlit dashboard combining:
  • EcoMed-AI chemistry classifier  (your model)
  • AquaSentinel anomaly risk       (friend's P1)
  • Source Tracing map              (friend's P2 — live Folium map)

Run:  streamlit run app.py
"""

import json
import joblib
import numpy as np
import pandas as pd
import folium
import streamlit as st
from pathlib import Path
from streamlit_folium import st_folium

# ── Gemini AI (direct — used as fallback) ────────────────────────────────────
try:
    from google import genai as _genai
    GEMINI_API_KEY = "AIzaSyBd8vveqDJoBjCwovJ6q8tUIOxIV8yKZ7g"
    _gemini_client = _genai.Client(api_key=GEMINI_API_KEY)
    GEMINI_ENABLED = True
except Exception:
    _gemini_client = None
    GEMINI_ENABLED = False

# ── Live Cloud Run APIs (Microservices Architecture) ──────────────────────────
import urllib.request as _urllib_req

# YOUR API (EcoMed-AI Intelligence Node)
ECOMED_API_URL = "https://ecomed-api-jvabpvtrfq-uc.a.run.app/predict"

# FRIEND'S API (AquaSentinel Source Tracing Node)
FRIEND_API_URL = "https://ecomed-backend-474707939537.us-central1.run.app/trace-pollution"

@st.cache_data(show_spinner=False, ttl=60)
def _ecomed_api_predict(features_json: str) -> dict | None:
    """
    Call YOUR live Cloud Run API (ML + Gemini + Safety).
    Returns result dict with 'verdict', 'prediction_score', 'ai_explanation'.
    """
    import json as _json
    # Cloud Run expects just the features dict inside 'instances'
    payload = _json.dumps({
        "instances": [_json.loads(features_json)],
        "parameters": {"explain": True}
    }).encode()
    
    req = _urllib_req.Request(
        ECOMED_API_URL, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        with _urllib_req.urlopen(req, timeout=15) as resp:
            data = _json.loads(resp.read())
            return data.get("predictions", [None])[0]
    except Exception:
        return None

@st.cache_data(show_spinner=False, ttl=60)
def _friend_api_trace(lat: float, lon: float, contaminant: str, level: float) -> dict | None:
    """
    Call FRIEND'S live Cloud Run API for source tracing logic.
    """
    import json as _json
    import datetime
    
    # Construct payload matching her expected format
    payload = _json.dumps({
        "sensor_readings": [{
            "location": {"lat": lat, "lng": lon},
            "contaminant_level": level,
            "timestamp": datetime.datetime.now().isoformat() + "Z"
        }],
        "contaminant_type": contaminant
    }).encode()

    req = _urllib_req.Request(
        FRIEND_API_URL, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        with _urllib_req.urlopen(req, timeout=15) as resp:
            return _json.loads(resp.read())
    except Exception:
        return None

# ── NOTE: Vertex AI endpoint was shut down (2026-02-21). App uses Cloud Run. ──

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EcoMed-AI | Water Safety Intelligence",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Styling
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.verdict-safe {
    background: linear-gradient(135deg, #0f5132, #198754);
    color: white; padding: 28px; border-radius: 16px;
    text-align: center; font-size: 1.4rem; font-weight: 700;
    box-shadow: 0 4px 20px rgba(25,135,84,0.4);
}
.verdict-caution {
    background: linear-gradient(135deg, #664d03, #ffc107);
    color: white; padding: 28px; border-radius: 16px;
    text-align: center; font-size: 1.4rem; font-weight: 700;
    box-shadow: 0 4px 20px rgba(255,193,7,0.4);
}
.verdict-unsafe {
    background: linear-gradient(135deg, #58151c, #dc3545);
    color: white; padding: 28px; border-radius: 16px;
    text-align: center; font-size: 1.4rem; font-weight: 700;
    box-shadow: 0 4px 20px rgba(220,53,69,0.4);
}
.signal-card {
    background: #1e1e2e; color: #cdd6f4;
    border-radius: 12px; padding: 18px;
    border: 1px solid #313244; margin-bottom: 10px;
}
.signal-label { font-size: 0.75rem; color: #a6adc8; text-transform: uppercase; letter-spacing: 1px; }
.signal-value { font-size: 1.6rem; font-weight: 700; color: #cba6f7; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
P2_BASE = Path("water_contaminant_source_P2/water_contaminant_source_P2/water_sourcing - Copy/aquasentinel")
SOURCE_RESULTS_CSV = P2_BASE / "source_tracing_results.csv"
GIS_CSV            = P2_BASE / "location_aware_gis_leakage_dataset.csv"

# ─────────────────────────────────────────────────────────────────────────────
# Load config + models (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    with open("integration_config.json") as f:
        cfg = json.load(f)
    p = cfg["paths"]
    model   = joblib.load(p["primary_model"])
    scaler  = joblib.load(p["primary_scaler"])
    imputer = joblib.load(p["primary_imputer"])
    with open(p["primary_features"]) as f:
        feat_meta = json.load(f)
    aqua = None
    try:
        aqua = joblib.load(p["aquasentinel_model"])
    except Exception:
        pass
    return cfg, model, scaler, imputer, feat_meta, aqua

BANGLADESH_CSV   = "data/raw/bangladesh_water.csv"

@st.cache_data
def load_map_data():
    results, gis, bd_data = None, None, None
    try:
        results = pd.read_csv(SOURCE_RESULTS_CSV)
    except Exception:
        pass
    try:
        gis = pd.read_csv(GIS_CSV)
    except Exception:
        pass
    try:
        bd_data = pd.read_csv(BANGLADESH_CSV)
    except Exception:
        pass
    return results, gis, bd_data

try:
    cfg, model, scaler, imputer, feat_meta, aqua = load_assets()
except Exception as e:
    st.error(f"❌ Could not load model: {e}")
    st.stop()

inf = cfg["inference"]
results_df, gis_df, bd_df = load_map_data()

# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────────────────────────────────────
def engineer_features(row: dict) -> pd.DataFrame:
    df = pd.DataFrame([row])
    df["heavy_metal_load"] = df["arsenic"] + df["cadmium"] + df["lead"] + df["mercury"] + df["chromium"]
    df["pathogen_risk"]    = df["bacteria"] * df["viruses"]
    df["disinfect_proxy"]  = df["chloramine"] * df["nitrates"]
    df["radio_composite"]  = df["radium"] + df["uranium"]
    df["mineral_excess"]   = df["barium"] + df["aluminium"] + df["silver"]
    return df[feat_meta["all_features"]]

# ─────────────────────────────────────────────────────────────────────────────
# AquaSentinel bridge
# ─────────────────────────────────────────────────────────────────────────────
def get_anomaly_risk(chloramine_val, bacteria_val, nitrates_val):
    if aqua is None:
        return None
    try:
        frozen_model  = aqua["model"]
        frozen_scaler = aqua["scaler"]
        feature_names = aqua["feature_names"]
        n = 1
        cond_grad = np.array([0.0])
        turb_grad = np.array([0.0])
        nit_grad  = np.array([0.0])
        hour      = np.array([12.0])
        sensor_df = pd.DataFrame({
            "conductivity_gradient_smooth_fixed": cond_grad,
            "turbidity_gradient_smooth_fixed":    turb_grad,
            "temporal_month":                     np.array([6.0]),
            "nitrate_gradient_smooth_fixed":      nit_grad,
            "conductivity_6h_zscore_fixed":       cond_grad,
            "temperature_gradient_smooth_fixed":  np.zeros(n),
            "nitrate_6h_zscore_fixed":            nit_grad,
            "turbidity_6h_zscore_fixed":          turb_grad,
            "hour":                               hour,
            "hour_cos":                           np.cos(2 * np.pi * hour / 24),
            "hour_sin":                           np.sin(2 * np.pi * hour / 24),
            "temporal_dayofweek":                 np.array([2.0]),
            "is_weekend":                         np.zeros(n),
        })
        sensor_df = sensor_df[feature_names]
        scaled    = frozen_scaler.transform(sensor_df)
        return float(frozen_model.predict_proba(scaled)[0, 1])
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────────────────────
# Build the Folium map (friend's P2 output + GIS layer + live prediction pin)
# ─────────────────────────────────────────────────────────────────────────────
PROXIMITY_COLORS = {
    "VERY_CLOSE (0-100m)":  "red",
    "CLOSE (100-300m)":     "orange",
    "MODERATE (300-500m)":  "beige",
    "FAR (500m+)":          "green",
}

def build_map(live_lat=None, live_lon=None, live_label="", live_prob=None, live_anomaly=None, location_name="Dubai"):
    # Centre on the user's chosen location (works globally)
    center = [live_lat if live_lat is not None else 25.19,
              live_lon if live_lon is not None else 55.23]
    m = folium.Map(location=center, zoom_start=11, tiles=None)

    # Multiple colourful tile options
    folium.TileLayer("OpenStreetMap",          name="🗺️ Street Map").add_to(m)
    folium.TileLayer("CartoDB Positron",        name="🌫️ Light").add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="🛰️ Satellite"
    ).add_to(m)
    
    # ── REGION SPECIFIC LAYERS ────────────────────────────────────────────────
    
    # CASE 1: BANGLADESH (Show Groundwater Risks)
    if "Bangladesh" in location_name and bd_df is not None:
        # Arsenic Layer
        arsenic_layer = folium.FeatureGroup(name="☠️ Arsenic Risk (Groundwater)", show=True)
        # Bacteria Layer
        bacteria_layer = folium.FeatureGroup(name="🦠 Bacteria Risk (Ponds/Wells)", show=False)
        
        # Sample 500 points to keep map fast
        sample_bd = bd_df.sample(min(500, len(bd_df)), random_state=42)
        
        for _, row in sample_bd.iterrows():
            # ARSENIC MARKERS
            if row['arsenic'] > 0.01:
                color = "#dc2626" if row['arsenic'] > 0.05 else "#f97316"
                radius = 8 if row['arsenic'] > 0.05 else 5
                
                folium.CircleMarker(
                    location=[row["latitude"], row["longitude"]],
                    radius=radius, color=color, fill=True, fill_opacity=0.7, weight=1,
                    popup=f"<b>Arsenic:</b> {row['arsenic']:.3f} mg/L<br>(Limit: 0.01)",
                    tooltip=f"Arsenic: {row['arsenic']:.3f} mg/L"
                ).add_to(arsenic_layer)
            else:
                folium.CircleMarker(
                    location=[row["latitude"], row["longitude"]],
                    radius=2, color="#22c55e", fill=True, fill_opacity=0.5, weight=0
                ).add_to(arsenic_layer)

            # BACTERIA MARKERS
            if row['bacteria'] > 0:
                folium.CircleMarker(
                    location=[row["latitude"], row["longitude"]],
                    radius=6, color="#7e22ce", fill=True, fill_opacity=0.6, weight=1,
                    tooltip="Bacteria Detected"
                ).add_to(bacteria_layer)

        arsenic_layer.add_to(m)
        bacteria_layer.add_to(m)

    # CASE 2: DUBAI (Show Friend's Infrastructure)
    elif "Dubai" in location_name and gis_df is not None:
        leaks    = gis_df[gis_df["Leakage_Flag"] == 1]
        no_leaks = gis_df[gis_df["Leakage_Flag"] == 0]

        gis_layer = folium.FeatureGroup(name="🔧 Infrastructure Sensors", show=True)
        for _, row in no_leaks.sample(min(300, len(no_leaks)), random_state=42).iterrows():
            folium.CircleMarker(
                location=[row["Latitude"], row["Longitude"]],
                radius=7, color="#16a34a", fill=True,
                fill_color="#22c55e", fill_opacity=0.75, weight=2,
                tooltip=f"✅ {row['Location_Code']} | Normal"
            ).add_to(gis_layer)

        for _, row in leaks.iterrows():
            folium.CircleMarker(
                location=[row["Latitude"], row["Longitude"]],
                radius=14, color="#b91c1c", fill=True,
                fill_color="#ef4444", fill_opacity=0.95, weight=3,
                popup=folium.Popup(
                    f"<div style='font-family:Inter,sans-serif;min-width:200px'>"
                    f"<h4 style='color:#dc2626;margin:0'>⚠️ LEAK DETECTED</h4>"
                    f"<b>Zone:</b> {row['Zone']} | <b>Block:</b> {row['Block']}<br>"
                    f"<b>Pipe:</b> {row['Pipe']}<br>"
                    f"<b>Pressure:</b> {row['Pressure']:.1f} bar<br>"
                    f"<b>Flow Rate:</b> {row['Flow_Rate']:.1f} L/s</div>",
                    max_width=260
                ),
                tooltip=f"🔴 LEAK — {row['Location_Code']}"
            ).add_to(gis_layer)
        gis_layer.add_to(m)

    # ── Layer 2: Source tracing anomaly events (friend's P2 output) ──────────
    # Hex colours for proximity rings
    PROX_HEX = {
        "VERY_CLOSE (0-100m)":  ("#7f1d1d", "#dc2626"),   # dark red / bright red
        "CLOSE (100-300m)":     ("#92400e", "#f97316"),   # dark orange / bright orange
        "MODERATE (300-500m)": ("#713f12", "#eab308"),   # dark yellow / bright yellow
        "FAR (500m+)":          ("#14532d", "#22c55e"),   # dark green / bright green
    }
    if results_df is not None:
        trace_layer = folium.FeatureGroup(name="📍 Source Tracing Events (P2)", show=True)
        for _, row in results_df.iterrows():
            icon_color = PROXIMITY_COLORS.get(row["source_proximity"], "blue")
            border, fill = PROX_HEX.get(row["source_proximity"], ("#1e40af", "#3b82f6"))
            # Outer glow ring
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=28, color=border, fill=True,
                fill_color=fill, fill_opacity=0.18, weight=2, dash_array="6"
            ).add_to(trace_layer)
            # Inner solid dot
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=12, color=border, fill=True,
                fill_color=fill, fill_opacity=0.9, weight=3,
                popup=folium.Popup(
                    f"<div style='font-family:Inter,sans-serif;min-width:220px'>"
                    f"<h4 style='margin:0 0 6px'>📍 Anomaly Event</h4>"
                    f"<b>Time:</b> {row['timestamp']}<br>"
                    f"<b>Zone:</b> {row['nearest_zone']} | <b>Block:</b> {row['nearest_block']}<br>"
                    f"<b>Pipe:</b> {row['nearest_pipe']}<br>"
                    f"<b>Distance to sensor:</b> {row['distance_to_nearest_m']:.0f} m<br>"
                    f"<b style='color:{fill}'>Source: {row['source_proximity']}</b><br>"
                    f"<b>Confidence:</b> {row['source_confidence']:.0%}</div>",
                    max_width=290
                ),
                tooltip=(
                    f"P2 ▸ {row['source_proximity']} | "
                    f"Confidence: {row['source_confidence']:.0%}"
                )
            ).add_to(trace_layer)
        trace_layer.add_to(m)

    # ── Layer 3: Live EcoMed-AI prediction pin ────────────────────────────────
    if live_lat is not None and live_lon is not None:
        live_layer = folium.FeatureGroup(name="🧪 EcoMed-AI Live Reading", show=True)
        # Determine color based on safety label (which includes overrides)
        if "UNSAFE" in live_label:
            pin_color, pin_hex, pin_fill = "red", "#dc2626", "#f87171"
        elif "CAUTION" in live_label:
            pin_color, pin_hex, pin_fill = "orange", "#f59e0b", "#fbbf24"
        else:
            # Only green if explicitly SAFE and not UNSAFE
            pin_color, pin_hex, pin_fill = "green", "#16a34a", "#4ade80"
        anomaly_str = f"{live_anomaly*100:.0f}%" if live_anomaly is not None else "N/A"

        # Outer pulse ring
        folium.Circle(
            location=[live_lat, live_lon],
            radius=600, color=pin_hex, fill=True,
            fill_color=pin_fill, fill_opacity=0.12, weight=3, dash_array="8"
        ).add_to(live_layer)
        # Middle ring
        folium.Circle(
            location=[live_lat, live_lon],
            radius=300, color=pin_hex, fill=True,
            fill_color=pin_fill, fill_opacity=0.20, weight=2
        ).add_to(live_layer)
        # Centre pin
        folium.Marker(
            location=[live_lat, live_lon],
            icon=folium.Icon(color=pin_color, icon="flask", prefix="fa"),
            popup=folium.Popup(
                f"<div style='font-family:Inter,sans-serif;min-width:220px'>"
                f"<h4 style='color:{pin_hex};margin:0'>🧪 EcoMed-AI Reading</h4>"
                f"<b>Verdict:</b> {live_label}<br>"
                f"<b>Chemistry score:</b> {(live_prob or 0)*100:.1f}%<br>"
                f"<b>AquaSentinel anomaly risk:</b> {anomaly_str}<br>"
                f"<b>Location:</b> {live_lat:.4f}°N, {live_lon:.4f}°E</div>",
                max_width=270
            ),
            tooltip=f"🧪 Your reading: {live_label} ({(live_prob or 0)*100:.1f}%)"
        ).add_to(live_layer)
        live_layer.add_to(m)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:9999;
                background:rgba(255,255,255,0.95);color:#1e293b;
                padding:14px 18px;border-radius:12px;
                border:2px solid #e2e8f0;font-size:13px;
                font-family:Inter,sans-serif;line-height:2;
                box-shadow:0 4px 16px rgba(0,0,0,0.15)">
        <b style='font-size:14px'>📍 Source Proximity (P2)</b><br>
        <span style='color:#dc2626'>⬤</span> VERY CLOSE (&lt;100m)<br>
        <span style='color:#f97316'>⬤</span> CLOSE (100–300m)<br>
        <span style='color:#eab308'>⬤</span> MODERATE (300–500m)<br>
        <span style='color:#22c55e'>⬤</span> FAR (&gt;500m)<br>
        <hr style="border-color:#e2e8f0;margin:6px 0">
        <b style='font-size:14px'>🔧 Infrastructure</b><br>
        <span style='color:#22c55e'>⬤</span> Normal sensor<br>
        <span style='color:#ef4444'>⬤</span> Leak detected<br>
        <hr style="border-color:#e2e8f0;margin:6px 0">
        <b style='font-size:14px'>🧪 EcoMed-AI</b><br>
        <span style='color:#16a34a'>⬤</span> Safe reading<br>
        <span style='color:#dc2626'>⬤</span> Unsafe reading
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl(collapsed=False).add_to(m)
    return m

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — inputs
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 💧 EcoMed-AI")
st.sidebar.markdown("**Water Safety Intelligence**")
st.sidebar.markdown("---")

# Location presets — works globally, not just Dubai
LOCATION_PRESETS = {
    "🇧🇩 Dhaka, Bangladesh":     (23.8103,  90.4125),
    "🇦🇪 Dubai, UAE":            (25.2048,  55.2708),
    "🇬🇧 London, UK":            (51.5074,  -0.1278),
    "🇺🇸 New York, USA":         (40.7128, -74.0060),
    "🇮🇳 Mumbai, India":         (19.0760,  72.8777),
    "🇳🇬 Lagos, Nigeria":        (6.5244,   3.3792),
    "🇧🇷 São Paulo, Brazil":     (-23.5505, -46.6333),
    "🌐 Custom coordinates":     (None, None),
}

st.sidebar.markdown("### 📍 Sample Location")
preset = st.sidebar.selectbox("Choose location", list(LOCATION_PRESETS.keys()))
preset_lat, preset_lon = LOCATION_PRESETS[preset]

if preset_lat is None:
    sample_lat = st.sidebar.number_input("Latitude",  value=23.8103, format="%.4f")
    sample_lon = st.sidebar.number_input("Longitude", value=90.4125, format="%.4f")
else:
    sample_lat = st.sidebar.number_input("Latitude",  value=float(preset_lat), format="%.4f")
    sample_lon = st.sidebar.number_input("Longitude", value=float(preset_lon), format="%.4f")

st.sidebar.markdown("### 🧪 Chemistry Parameters")
aluminium  = st.sidebar.slider("Aluminium (mg/L)",    0.0,  5.0,  0.5,  0.01)
ammonia    = st.sidebar.slider("Ammonia (mg/L)",       0.0, 32.0,  1.5,  0.1)
arsenic    = st.sidebar.slider("Arsenic (mg/L)",       0.0,  1.0,  0.01, 0.001)
barium     = st.sidebar.slider("Barium (mg/L)",        0.0,  7.0,  0.7,  0.01)
cadmium    = st.sidebar.slider("Cadmium (mg/L)",       0.0,  0.2,  0.003,0.001)
chloramine = st.sidebar.slider("Chloramine (mg/L)",    0.0, 10.0,  2.0,  0.1)
chromium   = st.sidebar.slider("Chromium (mg/L)",      0.0,  1.0,  0.05, 0.001)
copper     = st.sidebar.slider("Copper (mg/L)",        0.0,  2.0,  2.0,  0.01)
flouride   = st.sidebar.slider("Fluoride (mg/L)",      0.0,  2.0,  1.5,  0.01)
bacteria   = st.sidebar.slider("Bacteria (CFU/mL)",    0.0,  1.0,  0.0,  0.01)
viruses    = st.sidebar.slider("Viruses (PFU/mL)",     0.0,  1.0,  0.0,  0.01)
lead       = st.sidebar.slider("Lead (mg/L)",          0.0,  0.2,  0.01, 0.001)
nitrates   = st.sidebar.slider("Nitrates (mg/L)",      0.0, 20.0,  5.0,  0.1)
nitrites   = st.sidebar.slider("Nitrites (mg/L)",      0.0,  3.0,  0.1,  0.01)
mercury    = st.sidebar.slider("Mercury (mg/L)",       0.0,  0.01, 0.001,0.0001)
perchlorate= st.sidebar.slider("Perchlorate (mg/L)",   0.0, 56.0,  4.0,  0.1)
radium     = st.sidebar.slider("Radium (pCi/L)",       0.0,  8.0,  0.1,  0.01)
selenium   = st.sidebar.slider("Selenium (mg/L)",      0.0,  0.1,  0.01, 0.001)
silver     = st.sidebar.slider("Silver (mg/L)",        0.0,  1.0,  0.1,  0.01)
uranium    = st.sidebar.slider("Uranium (mg/L)",       0.0,  0.1,  0.015,0.001)

# ─────────────────────────────────────────────────────────────────────────────
# Predict
# ─────────────────────────────────────────────────────────────────────────────
sample = dict(aluminium=aluminium, ammonia=ammonia, arsenic=arsenic,
              barium=barium, cadmium=cadmium, chloramine=chloramine,
              chromium=chromium, copper=copper, flouride=flouride,
              bacteria=bacteria, viruses=viruses, lead=lead,
              nitrates=nitrates, nitrites=nitrites, mercury=mercury,
              perchlorate=perchlorate, radium=radium, selenium=selenium,
              silver=silver, uranium=uranium)

X_raw = engineer_features(sample)
X_imp = pd.DataFrame(imputer.transform(X_raw), columns=X_raw.columns)
X_sc  = scaler.transform(X_imp)
prob  = float(model.predict_proba(X_sc)[0, 1])
anomaly_risk = get_anomaly_risk(chloramine, bacteria, nitrates)
heavy_load   = arsenic + cadmium + lead + mercury + chromium

# ── Global Verdict Calculation (ML + Safety Override) ──────────────────────────
# 1. Calculate Violation Count
_viol_count = sum([
    aluminium > 0.2, arsenic > 0.01, cadmium > 0.003,
    lead > 0.01, mercury > 0.006, chromium > 0.05,
    nitrates > 50, bacteria > 0, viruses > 0,
    radium > 5, uranium > 0.015
])

# 2. Initial ML Verdict
_verdict_base = ("SAFE" if prob >= inf["safety_threshold"]
                else ("CAUTION" if prob >= inf["caution_threshold"] else "UNSAFE"))

# 3. Apply Safety Override (Downgrade if violations exist)
if _verdict_base == "SAFE" and _viol_count > 0:
    _verdict_final = "CAUTION"  # Override
else:
    _verdict_final = _verdict_base

# 4. Set Global Display Label
if _verdict_final == "SAFE":
    live_label = "✅ SAFE"
elif _verdict_final == "CAUTION":
    live_label = "⚠️ CAUTION"
else:
    live_label = "🚨 UNSAFE"
    
# Store for use in tabs
_verdict_str = _verdict_final

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("# 💧 EcoMed-AI — Water Safety Intelligence")
st.markdown("*Chemistry analysis · Anomaly detection · Source tracing · Live map*")
st.markdown("---")

tab_dashboard, tab_map, tab_who = st.tabs(["🔬 Safety Dashboard", "🗺️ Live Source Map", "📋 WHO Limits"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: Safety Dashboard
# ══════════════════════════════════════════════════════════════════════════════
with tab_dashboard:
    # Calculate Verdict & Safety Override EARLY (so UI can use it)
    _viol_count = sum([
        aluminium > 0.2, arsenic > 0.01, cadmium > 0.003,
        lead > 0.01, mercury > 0.006, chromium > 0.05,
        nitrates > 50, bacteria > 0, viruses > 0,
        radium > 5, uranium > 0.015
    ])
    _verdict_str = ("SAFE" if prob >= inf["safety_threshold"]
                    else ("CAUTION" if prob >= inf["caution_threshold"] else "UNSAFE"))

    # 🚨 SAFETY OVERRIDE: If WHO limits are violated, never show "SAFE"
    if _verdict_str == "SAFE" and _viol_count > 0:
        _verdict_str = "CAUTION"  # Downgrade to ensure users are warned

    # Build contaminant context string (needed for AI later)
    _contam_parts = []
    if arsenic   > 0.01:  _contam_parts.append(f"arsenic ({arsenic:.3f} mg/L — limit 0.01)")
    if lead      > 0.01:  _contam_parts.append(f"lead ({lead:.3f} mg/L — limit 0.01)")
    if bacteria  > 0:     _contam_parts.append(f"bacteria ({bacteria:.2f} CFU/mL)")
    if viruses   > 0:     _contam_parts.append(f"viruses ({viruses:.2f} PFU/mL)")
    if mercury   > 0.006: _contam_parts.append(f"mercury ({mercury:.4f} mg/L — limit 0.006)")
    if nitrates  > 50:    _contam_parts.append(f"nitrates ({nitrates:.1f} mg/L — limit 50)")
    if cadmium   > 0.003: _contam_parts.append(f"cadmium ({cadmium:.3f} mg/L — limit 0.003)")
    if aluminium > 0.2:   _contam_parts.append(f"aluminium ({aluminium:.2f} mg/L — limit 0.2)")
    if uranium   > 0.015: _contam_parts.append(f"uranium ({uranium:.3f} mg/L — limit 0.015)")
    if radium    > 5:     _contam_parts.append(f"radium ({radium:.1f} pCi/L — limit 5)")
    _contam_str = ", ".join(_contam_parts) if _contam_parts else "No major contaminants detected"

    col_verdict, col_signals, col_interp = st.columns([2, 1.5, 1.5])

    with col_verdict:
        st.markdown("### 🔬 Safety Verdict")
        # Use _verdict_str (which includes Safety Override logic) for the UI
        if _verdict_str == "SAFE":
            st.markdown(f"""<div class='verdict-safe'>
                ✅ WATER IS SAFE<br>
                <span style='font-size:1rem;font-weight:400'>
                General Water Profile: {prob*100:.1f}%
                </span></div>""", unsafe_allow_html=True)
        elif _verdict_str == "CAUTION":
            st.markdown(f"""<div class='verdict-caution'>
                ⚠️ CAUTION — MARGINAL<br>
                <span style='font-size:1rem;font-weight:400'>
                General Water Profile: {prob*100:.1f}% · Safety Override Active
                </span></div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class='verdict-unsafe'>
                🚨 UNSAFE — DO NOT CONSUME<br>
                <span style='font-size:1rem;font-weight:400'>
                General Water Profile: {prob*100:.1f}% · Multiple risk factors
                </span></div>""", unsafe_allow_html=True)

        st.markdown("")
        st.progress(prob, text=f"Safety probability: {prob*100:.1f}%")

    with col_signals:
        st.markdown("### 📡 System Signals")

        ecomed_color = "#a6e3a1" if prob >= 0.5 else "#f38ba8"
        st.markdown(f"""<div class='signal-card'>
            <div class='signal-label'>EcoMed-AI Chemistry</div>
            <div class='signal-value' style='color:{ecomed_color}'>{prob*100:.1f}%</div>
            <div style='font-size:0.8rem;color:#a6adc8'>Random Forest · 25 features</div>
        </div>""", unsafe_allow_html=True)

        if anomaly_risk is not None:
            aqua_color = "#f38ba8" if anomaly_risk > 0.6 else "#a6e3a1"
            aqua_label = "HIGH RISK" if anomaly_risk > 0.6 else ("MODERATE" if anomaly_risk > 0.3 else "LOW RISK")
            st.markdown(f"""<div class='signal-card'>
                <div class='signal-label'>AquaSentinel Anomaly (P1)</div>
                <div class='signal-value' style='color:{aqua_color}'>{anomaly_risk*100:.1f}%</div>
                <div style='font-size:0.8rem;color:#a6adc8'>Temporal anomaly risk · {aqua_label}</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class='signal-card'>
                <div class='signal-label'>AquaSentinel (P1)</div>
                <div class='signal-value' style='color:#6c7086'>N/A</div>
                <div style='font-size:0.8rem;color:#a6adc8'>Model not loaded</div>
            </div>""", unsafe_allow_html=True)

        # Call Friend's API (Live Source Tracing)
        _dom_contam = "chemical"
        if bacteria > 0: _dom_contam = "bacterial"
        elif arsenic > 0.01: _dom_contam = "arsenic_poisoning"
        
        # Use nitrates as a generic 'level' proxy if heavy_load is low
        _trace_val = float(heavy_load) if heavy_load > 0 else float(nitrates)
        _friend_res = _friend_api_trace(sample_lat, sample_lon, _dom_contam, _trace_val)

        if _friend_res and "predicted_source" in _friend_res:
            src_label = "📍 TRACE ACTIVE"
            src_color = "#f38ba8"
            _causes = _friend_res["predicted_source"].get("possible_causes", ["unknown"])
            _cause_str = _causes[0].replace("_", " ").title() if _causes else "Unknown Source"
            _detail = f"ID: {_cause_str}"
        elif heavy_load > 0.1:
            src_label, src_color = "VERY CLOSE (<100m)", "#f38ba8"
            _detail = f"Heavy load: {heavy_load:.3f}"
        elif heavy_load > 0.05:
            src_label, src_color = "CLOSE (100–300m)", "#fab387"
            _detail = f"Heavy load: {heavy_load:.3f}"
        elif heavy_load > 0.01:
            src_label, src_color = "MODERATE (300m–1km)", "#f9e2af"
            _detail = f"Heavy load: {heavy_load:.3f}"
        else:
            src_label, src_color = "DISTANT (>1km)", "#a6e3a1"
            _detail = "No active trace"

        st.markdown(f"""<div class='signal-card'>
            <div class='signal-label'>Source Tracing (AquaSentinel)</div>
            <div class='signal-value' style='color:{src_color};font-size:1.1rem'>{src_label}</div>
            <div style='font-size:0.8rem;color:#a6adc8'>{_detail}</div>
        </div>""", unsafe_allow_html=True)

    with col_interp:
        st.markdown("### 🧠 Interpretation")
        parts = []
        if _verdict_str == "SAFE":
            parts.append("✅ **EcoMed-AI**: Chemistry is **SAFE**")
        elif _verdict_str == "CAUTION":
            parts.append("⚠️ **EcoMed-AI**: Chemistry is **CAUTION**")
        else:
            parts.append("🚨 **EcoMed-AI**: Chemistry is **UNSAFE**")

        if anomaly_risk is not None:
            if anomaly_risk > 0.6:
                parts.append(f"⚠️ **AquaSentinel**: HIGH temporal anomaly ({anomaly_risk*100:.0f}%)")
            elif anomaly_risk > 0.3:
                parts.append(f"⚠️ **AquaSentinel**: Moderate anomaly ({anomaly_risk*100:.0f}%)")
            else:
                parts.append(f"✅ **AquaSentinel**: Low anomaly risk ({anomaly_risk*100:.0f}%)")

        parts.append(f"📍 **Source Tracing**: {src_label}")

        violations = sum([
            aluminium > 0.2, arsenic > 0.01, cadmium > 0.003,
            lead > 0.01, mercury > 0.006, chromium > 0.05,
            nitrates > 50, bacteria > 0, viruses > 0,
            radium > 5, uranium > 0.015
        ])
        if violations:
            parts.append(f"🔬 **{violations} WHO violation(s)** detected")

        for p in parts:
            st.markdown(f"- {p}")

        st.markdown("---")
        st.markdown(f"**📍 Sample location**")
        st.markdown(f"`{sample_lat:.4f}°N, {sample_lon:.4f}°E`")
        st.caption("→ See Map tab to view on the live source map")

# ── Gemini AI Explanation (full-width below the 3 columns) ───────────────────
@st.cache_data(show_spinner=False)
def _get_gemini_explanation(verdict: str, score: float, contaminants_str: str, violations: int) -> str:
    """Direct Gemini call — used as fallback when Vertex is unavailable."""
    if not GEMINI_ENABLED:
        return "⚠️ Gemini not available."
    try:
        prompt = f"""You are EcoMed-AI, a public health water safety assistant.

A water quality analysis just returned:
- Verdict: {verdict}
- Safety Score: {score:.2f} out of 1.0  (1.0 = perfectly safe)
- Key contaminants detected: {contaminants_str}
- WHO/EPA limit violations: {violations}

Your task:
1. Explain the result to a non-expert in simple, plain language (no jargon).
2. If UNSAFE or CAUTION, name the dangerous contaminants and why they are harmful.
3. Give exactly 2 practical, low-cost actions they can take RIGHT NOW (e.g., boil, filter, report).
4. Tone: calm but urgent. Use short sentences. Max 4 sentences total.
"""
        response = _gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Gemini explanation unavailable: {e}"

with tab_dashboard:
    st.markdown("---")
    st.markdown("### 🤖 Gemini AI Health Recommendation")

    # Build contaminant context string
    _contam_parts = []
    if arsenic   > 0.01:  _contam_parts.append(f"arsenic ({arsenic:.3f} mg/L — limit 0.01)")
    if lead      > 0.01:  _contam_parts.append(f"lead ({lead:.3f} mg/L — limit 0.01)")
    if bacteria  > 0:     _contam_parts.append(f"bacteria ({bacteria:.2f} CFU/mL)")
    if viruses   > 0:     _contam_parts.append(f"viruses ({viruses:.2f} PFU/mL)")
    if mercury   > 0.006: _contam_parts.append(f"mercury ({mercury:.4f} mg/L — limit 0.006)")
    if nitrates  > 50:    _contam_parts.append(f"nitrates ({nitrates:.1f} mg/L — limit 50)")
    if cadmium   > 0.003: _contam_parts.append(f"cadmium ({cadmium:.3f} mg/L — limit 0.003)")
    if aluminium > 0.2:   _contam_parts.append(f"aluminium ({aluminium:.2f} mg/L — limit 0.2)")
    if uranium   > 0.015: _contam_parts.append(f"uranium ({uranium:.3f} mg/L — limit 0.015)")
    if radium    > 5:     _contam_parts.append(f"radium ({radium:.1f} pCi/L — limit 5)")
    _contam_str = ", ".join(_contam_parts) if _contam_parts else "No major contaminants detected"

    _viol_count = sum([
        aluminium > 0.2, arsenic > 0.01, cadmium > 0.003,
        lead > 0.01, mercury > 0.006, chromium > 0.05,
        nitrates > 50, bacteria > 0, viruses > 0,
        radium > 5, uranium > 0.015
    ])

    _verdict_str = ("SAFE" if prob >= inf["safety_threshold"]
                    else ("CAUTION" if prob >= inf["caution_threshold"] else "UNSAFE"))

    # 🚨 SAFETY OVERRIDE: If WHO limits are violated, never show "SAFE"
    if _verdict_str == "SAFE" and _viol_count > 0:
        _verdict_str = "CAUTION"  # Downgrade to ensure users are warned

    with st.spinner("🤖 Consulting AI health advisor..."):
        # ── Try Vertex AI endpoint first (ML + Gemini combined) ──────────────
        import json as _json
        _features = {
            "aluminium": aluminium, "ammonia": ammonia, "arsenic": arsenic,
            "barium": barium, "cadmium": cadmium, "chloramine": chloramine,
            "chromium": chromium, "copper": copper, "flouride": flouride,
            "bacteria": bacteria, "viruses": viruses, "lead": lead,
            "nitrates": nitrates, "nitrites": nitrites, "mercury": mercury,
            "perchlorate": perchlorate, "radium": radium, "selenium": selenium,
            "silver": silver, "uranium": uranium,
        }
        _vertex_result = _ecomed_api_predict(_json.dumps(_features))

        if _vertex_result:
            # TRUST THE API: If API gives a verdict, use it!
            # The API (ecomed-api) already has the Safety Override logic inside it.
            api_verdict = _vertex_result.get("verdict", _verdict_str)
            _ai_text    = _vertex_result.get("ai_explanation", "AI analysis unavailable")
            _ai_source  = "🌩️ EcoMed-AI API · Gemini 2.5 Flash"
            
            # Map API verdict to UI colors
            if api_verdict == "SAFE":
                _verdict_str = "SAFE"
                _card_color, _border_color, _icon = "#0f5132", "#198754", "✅"
            elif api_verdict == "CAUTION":
                _verdict_str = "CAUTION"
                _card_color, _border_color, _icon = "#664d03", "#ffc107", "⚠️"
            else:
                _verdict_str = "UNSAFE"
                _card_color, _border_color, _icon = "#58151c", "#dc3545", "🚨"
        elif GEMINI_ENABLED:
            # Fallback to local logic + Gemini
            _ai_text   = _get_gemini_explanation(_verdict_str, prob, _contam_str, _viol_count)
            _ai_source = "🔑 Direct · Gemini 2.5 Flash"
        else:
            _ai_text   = "⚠️ AI explanation unavailable."
            _ai_source = "unavailable"

    # Colour card by verdict
    _card_color   = "#0f5132" if _verdict_str == "SAFE" else ("#664d03" if _verdict_str == "CAUTION" else "#58151c")
    _border_color = "#198754" if _verdict_str == "SAFE" else ("#ffc107" if _verdict_str == "CAUTION" else "#dc3545")
    _icon         = "✅" if _verdict_str == "SAFE" else ("⚠️" if _verdict_str == "CAUTION" else "🚨")

    st.markdown(f"""
    <div style="background:{_card_color};border-left:5px solid {_border_color};
                border-radius:12px;padding:20px 24px;margin-top:8px;
                box-shadow:0 4px 20px rgba(0,0,0,0.3);">
        <div style="font-size:0.75rem;color:#adb5bd;text-transform:uppercase;
                    letter-spacing:1.5px;margin-bottom:8px;">{_ai_source} · AI Health Advisory</div>
        <div style="font-size:1.05rem;color:#f8f9fa;line-height:1.7;">{_icon} {_ai_text}</div>
        <div style="font-size:0.7rem;color:#6c757d;margin-top:12px;">Based on: {_contam_str}</div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: Live Source Map
# ══════════════════════════════════════════════════════════════════════════════
with tab_map:
    st.markdown("### 🗺️ Live Water Source Tracing Map")
    st.markdown(
        "Combines **friend's P2 source tracing events** (coloured pins) · "
        "**GIS infrastructure sensors** (green/red dots) · "
        "**your live EcoMed-AI reading** (flask icon)"
    )

    col_info, col_stats = st.columns([3, 1])
    with col_info:
        st.info(
            f"📍 **Your reading** is pinned at ({sample_lat:.4f}, {sample_lon:.4f}) "
            f"— verdict: **{live_label}** (Base Quality: {prob*100:.1f}%)"
        )
    with col_stats:
        if results_df is not None:
            st.metric("P2 Anomaly Events", len(results_df))
        if gis_df is not None:
            n_leaks = int(gis_df["Leakage_Flag"].sum())
            st.metric("Infrastructure Leaks", n_leaks)

    # Build and render the map
    fmap = build_map(
        live_lat=sample_lat,
        live_lon=sample_lon,
        live_label=live_label,
        live_prob=prob,
        live_anomaly=anomaly_risk,
        location_name=preset
    )
    st_folium(fmap, width="100%", height=600, returned_objects=[])

    # Show P2 results table below map (Only for Dubai)
    if "Dubai" in preset and results_df is not None:
        with st.expander("📊 Source Tracing Results Table (Friend's P2 Output)"):
            st.dataframe(results_df, use_container_width=True)
    elif "Bangladesh" in preset and bd_df is not None:
        with st.expander("📊 Bangladesh Groundwater Data (Arsenic Survey)"):
             st.dataframe(bd_df.head(100), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: WHO Limits
# ══════════════════════════════════════════════════════════════════════════════
with tab_who:
    st.markdown("### 📋 WHO Drinking Water Parameter Check")
    checks = [
        ("Aluminium",   aluminium,   0.2,   "mg/L", "WHO"),
        ("Arsenic",     arsenic,     0.01,  "mg/L", "WHO"),
        ("Cadmium",     cadmium,     0.003, "mg/L", "WHO"),
        ("Lead",        lead,        0.01,  "mg/L", "WHO"),
        ("Mercury",     mercury,     0.006, "mg/L", "WHO"),
        ("Chromium",    chromium,    0.05,  "mg/L", "WHO"),
        ("Nitrates",    nitrates,    50.0,  "mg/L", "WHO"),
        ("Nitrites",    nitrites,    3.0,   "mg/L", "WHO"),
        ("Bacteria",    bacteria,    0.0,   "CFU/mL","WHO"),
        ("Viruses",     viruses,     0.0,   "PFU/mL","WHO"),
        ("Radium",      radium,      5.0,   "pCi/L", "EPA"),
        ("Uranium",     uranium,     0.015, "mg/L",  "WHO"),
        ("Fluoride",    flouride,    1.5,   "mg/L",  "WHO"),
        ("Chloramine",  chloramine,  5.0,   "mg/L",  "WHO"),
        ("Perchlorate", perchlorate, 15.0,  "mg/L",  "EPA"),
        ("Selenium",    selenium,    0.04,  "mg/L",  "WHO"),
        ("Silver",      silver,      0.1,   "mg/L",  "EPA"),
        ("Copper",      copper,      2.0,   "mg/L",  "WHO"),
        ("Ammonia",     ammonia,     1.5,   "mg/L",  "WHO"),
        ("Barium",      barium,      0.7,   "mg/L",  "WHO"),
    ]

    violations = 0
    col1, col2 = st.columns(2)
    for i, (name, val, limit, unit, source) in enumerate(checks):
        col = col1 if i % 2 == 0 else col2
        with col:
            if val > limit:
                st.error(f"❌ **{name}**: {val:.4f} > {limit} {unit} ({source} limit)")
                violations += 1
            else:
                st.success(f"✅ **{name}**: {val:.4f} {unit} — within {source} limit")

    st.markdown("---")
    if violations == 0:
        st.success(f"✅ All {len(checks)} parameters are within WHO/EPA limits")
    else:
        st.error(f"🚨 **{violations} out of {len(checks)} parameters exceed WHO/EPA limits**")

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#6c7086;font-size:0.8rem'>"
    "EcoMed-AI · Integrated Water Safety System · "
    "Chemistry (EcoMed-AI) + Anomaly Detection (AquaSentinel P1) + Source Tracing (P2)"
    "</div>",
    unsafe_allow_html=True
)
