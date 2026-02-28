"""
generate_bangladesh_data.py
============================
Generates a realistic Bangladesh groundwater quality dataset based on:
  - British Geological Survey (BGS) Bangladesh Arsenic Survey (2001)
  - WHO/UNICEF JMP Bangladesh Water Quality Reports
  - DPHE (Dept of Public Health Engineering) published statistics

Key real-world facts encoded:
  - ~35% of Bangladesh wells exceed WHO arsenic limit (0.01 mg/L)
  - High-arsenic districts: Comilla, Chandpur, Munshiganj, Faridpur, Satkhira
  - Low-arsenic districts: Sylhet, Chittagong Hill Tracts, Rangpur
  - Bacteria contamination: ~40% of sources (open wells, ponds)
  - Nitrate elevated near agricultural zones (Rajshahi, Bogura)
  - Iron elevated in most groundwater (not in our 20 features but noted)

Output: data/raw/bangladesh_water.csv  (20 features + is_safe + lat + lon + district)
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
N = 2000  # samples

# ─────────────────────────────────────────────────────────────────────────────
# Bangladesh district profiles (lat, lon, arsenic_risk, bacteria_risk)
# Based on BGS 2001 survey and DPHE reports
# ─────────────────────────────────────────────────────────────────────────────
DISTRICTS = [
    # (name,          lat,     lon,    arsenic_mean, arsenic_std, bacteria_prob, nitrate_elev)
    ("Dhaka",         23.810,  90.412, 0.040,        0.035,       0.30,          0.15),
    ("Comilla",       23.461,  91.188, 0.080,        0.060,       0.35,          0.20),
    ("Chandpur",      23.232,  90.669, 0.090,        0.070,       0.40,          0.18),
    ("Munshiganj",    23.542,  90.530, 0.075,        0.055,       0.35,          0.15),
    ("Faridpur",      23.607,  89.843, 0.065,        0.050,       0.38,          0.22),
    ("Satkhira",      22.718,  89.072, 0.055,        0.045,       0.45,          0.12),
    ("Rajshahi",      24.374,  88.601, 0.015,        0.012,       0.25,          0.35),  # high nitrate
    ("Bogura",        24.851,  89.370, 0.012,        0.010,       0.28,          0.38),  # high nitrate
    ("Sylhet",        24.899,  91.872, 0.005,        0.004,       0.20,          0.08),  # low arsenic
    ("Chittagong",    22.356,  91.783, 0.008,        0.006,       0.22,          0.10),  # low arsenic
    ("Rangpur",       25.746,  89.251, 0.010,        0.008,       0.30,          0.28),
    ("Khulna",        22.845,  89.540, 0.045,        0.040,       0.42,          0.14),
    ("Barisal",       22.701,  90.370, 0.035,        0.030,       0.48,          0.12),  # high bacteria (ponds)
    ("Mymensingh",    24.746,  90.407, 0.020,        0.018,       0.32,          0.20),
    ("Jessore",       23.167,  89.213, 0.050,        0.042,       0.36,          0.18),
]

rows = []
district_weights = [120, 100, 80, 80, 80, 80, 120, 100, 120, 120, 100, 100, 100, 100, 100]
total = sum(district_weights)
district_counts = [int(N * w / total) for w in district_weights]
district_counts[-1] += N - sum(district_counts)  # fix rounding

for (name, lat, lon, as_mean, as_std, bact_prob, nit_elev), count in zip(DISTRICTS, district_counts):
    for _ in range(count):
        # Spatial jitter (within ~30km of district centre)
        r_lat = lat + np.random.uniform(-0.25, 0.25)
        r_lon = lon + np.random.uniform(-0.25, 0.25)

        # ── Arsenic (the defining feature of Bangladesh groundwater) ──────────
        arsenic = max(0.0, np.random.lognormal(
            mean=np.log(max(as_mean, 0.001)), sigma=0.8
        ))
        arsenic = min(arsenic, 1.0)

        # ── Bacteria (open wells, ponds, shallow tubewells) ───────────────────
        bacteria = 1.0 if np.random.random() < bact_prob else 0.0
        viruses  = 1.0 if np.random.random() < bact_prob * 0.4 else 0.0

        # ── Nitrates (agricultural runoff — Rajshahi/Bogura belt) ─────────────
        nitrate_base = 8.0 if nit_elev > 0.3 else 3.0
        nitrates = max(0.0, np.random.normal(nitrate_base + nit_elev * 20, 4.0))
        nitrites = max(0.0, np.random.normal(0.15, 0.1))

        # ── Heavy metals (lower than arsenic but present) ─────────────────────
        lead     = max(0.0, np.random.lognormal(-5.0, 0.8))   # ~0.007 mean
        cadmium  = max(0.0, np.random.lognormal(-7.0, 0.7))   # ~0.001 mean
        mercury  = max(0.0, np.random.lognormal(-9.0, 0.6))   # ~0.0001 mean
        chromium = max(0.0, np.random.lognormal(-5.5, 0.7))   # ~0.004 mean

        # ── Minerals (typical South Asian groundwater) ────────────────────────
        aluminium   = max(0.0, np.random.lognormal(-1.5, 0.8))  # ~0.22 mean
        barium      = max(0.0, np.random.normal(0.3, 0.15))
        copper      = max(0.0, np.random.normal(0.4, 0.2))
        flouride    = max(0.0, np.random.normal(0.6, 0.3))      # generally low in BD
        selenium    = max(0.0, np.random.lognormal(-5.5, 0.6))
        silver      = max(0.0, np.random.lognormal(-6.0, 0.5))
        uranium     = max(0.0, np.random.lognormal(-6.5, 0.6))
        radium      = max(0.0, np.random.normal(0.3, 0.2))
        perchlorate = max(0.0, np.random.normal(2.0, 1.0))
        ammonia     = max(0.0, np.random.normal(0.8, 0.5))
        chloramine  = max(0.0, np.random.normal(0.5, 0.3))  # low — minimal treatment

        # ── Safety label (WHO-based) ──────────────────────────────────────────
        unsafe = (
            arsenic    > 0.01   or   # WHO limit
            bacteria   > 0.0    or   # must be zero
            viruses    > 0.0    or   # must be zero
            lead       > 0.01   or
            cadmium    > 0.003  or
            mercury    > 0.006  or
            chromium   > 0.05   or
            nitrates   > 50.0   or
            nitrites   > 3.0    or
            aluminium  > 0.2    or
            flouride   > 1.5    or
            uranium    > 0.015  or
            radium     > 5.0
        )
        is_safe = 0 if unsafe else 1

        rows.append({
            "district":    name,
            "latitude":    round(r_lat, 6),
            "longitude":   round(r_lon, 6),
            "aluminium":   round(aluminium,   4),
            "ammonia":     round(ammonia,     4),
            "arsenic":     round(arsenic,     5),
            "barium":      round(barium,      4),
            "cadmium":     round(cadmium,     5),
            "chloramine":  round(chloramine,  4),
            "chromium":    round(chromium,    5),
            "copper":      round(copper,      4),
            "flouride":    round(flouride,    4),
            "bacteria":    bacteria,
            "viruses":     viruses,
            "lead":        round(lead,        5),
            "nitrates":    round(nitrates,    3),
            "nitrites":    round(nitrites,    4),
            "mercury":     round(mercury,     6),
            "perchlorate": round(perchlorate, 3),
            "radium":      round(radium,      4),
            "selenium":    round(selenium,    5),
            "silver":      round(silver,      5),
            "uranium":     round(uranium,     5),
            "is_safe":     is_safe,
        })

df = pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────────────────────
# Stats report
# ─────────────────────────────────────────────────────────────────────────────
print(f"Generated {len(df)} samples")
print(f"Safe: {df['is_safe'].sum()} ({df['is_safe'].mean()*100:.1f}%)")
print(f"Unsafe: {(df['is_safe']==0).sum()} ({(df['is_safe']==0).mean()*100:.1f}%)")
print()
print("Arsenic violations (>0.01 mg/L):", (df['arsenic'] > 0.01).sum(),
      f"({(df['arsenic']>0.01).mean()*100:.1f}%)")
print("Bacteria violations:", (df['bacteria'] > 0).sum(),
      f"({(df['bacteria']>0).mean()*100:.1f}%)")
print("Lead violations:", (df['lead'] > 0.01).sum())
print()
print("By district (unsafe %):")
print(df.groupby("district")["is_safe"].apply(lambda x: f"{(1-x.mean())*100:.0f}% unsafe").to_string())

# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────
out = Path("data/raw/bangladesh_water.csv")
df.to_csv(out, index=False)
print(f"\nSaved to {out}")
