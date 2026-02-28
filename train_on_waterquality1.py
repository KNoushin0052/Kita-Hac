"""
train_on_waterquality1.py
=========================
Trains EcoMed-AI on the superior waterQuality1.csv dataset.

Why this dataset is better than water_potability.csv:
  - 7,996 samples (vs 3,276)
  - 20 features (vs 9) — includes heavy metals, bacteria, viruses
  - Strong feature-target correlations (aluminium: 0.33 vs max 0.03)
  - Accuracy ceiling: 94.3% (vs 64.1%)
  - ROC-AUC ceiling: 0.98 (vs ~0.67)

Leak-free protocol:
  1. Drop 3 corrupted rows
  2. Split FIRST (stratified)
  3. Fit imputer on train ONLY
  4. Fit scaler on train ONLY
  5. Train with class_weight='balanced' (11% safe vs 89% unsafe)
  6. Evaluate with 5-fold CV + train-test gap check

Run: python train_on_waterquality1.py
Output: data/processed/wq1_model/
"""

import json, os, numpy as np, pandas as pd, joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report,
                             roc_auc_score, confusion_matrix, f1_score)

SEED = 42
np.random.seed(SEED)

print("=" * 70)
print("🚀 TRAINING ON waterQuality1.csv  (Superior Dataset)")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load & clean
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_csv("data/raw/waterQuality1.csv")
print(f"\n📊 Raw shape: {df.shape}")

# Drop the 3 corrupted rows (#NUM! in ammonia and is_safe)
df = df[df['is_safe'] != '#NUM!'].copy()
df['ammonia'] = pd.to_numeric(df['ammonia'], errors='coerce')
df['is_safe']  = df['is_safe'].astype(int)
print(f"   After dropping 3 corrupted rows: {df.shape}")
print(f"   Class balance: Safe={df.is_safe.sum()} ({df.is_safe.mean()*100:.1f}%) | "
      f"Unsafe={len(df)-df.is_safe.sum()} ({(1-df.is_safe.mean())*100:.1f}%)")

X = df.drop('is_safe', axis=1)
y = df['is_safe']

# ─────────────────────────────────────────────────────────────────────────────
# 2. Feature engineering (domain-specific for heavy metals / pathogens)
# ─────────────────────────────────────────────────────────────────────────────
X = X.copy()
# Heavy metal composite (WHO priority contaminants)
X['heavy_metal_load']  = X['arsenic'] + X['cadmium'] + X['lead'] + X['mercury'] + X['chromium']
# Pathogen risk
X['pathogen_risk']     = X['bacteria'] * X['viruses']
# Disinfection byproduct proxy
X['disinfect_proxy']   = X['chloramine'] * X['nitrates']
# Radioactivity composite
X['radio_composite']   = X['radium'] + X['uranium']
# Mineral excess
X['mineral_excess']    = X['barium'] + X['aluminium'] + X['silver']

print(f"\n🔧 Feature engineering: {X.shape[1]} features "
      f"(20 original + 5 engineered)")

# ─────────────────────────────────────────────────────────────────────────────
# 3. ✅ Split FIRST
# ─────────────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y
)
print(f"\n✂️  Split: {len(X_train)} train | {len(X_test)} test")

# ─────────────────────────────────────────────────────────────────────────────
# 4. ✅ Impute on train ONLY
# ─────────────────────────────────────────────────────────────────────────────
imputer = SimpleImputer(strategy='median')
X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test_imp  = pd.DataFrame(imputer.transform(X_test),      columns=X_test.columns)

# ─────────────────────────────────────────────────────────────────────────────
# 5. ✅ Scale on train ONLY
# ─────────────────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_imp)
X_test_sc  = scaler.transform(X_test_imp)

# ─────────────────────────────────────────────────────────────────────────────
# 6. Train — RandomForest with balanced class weights
# ─────────────────────────────────────────────────────────────────────────────
rf = RandomForestClassifier(
    n_estimators    = 300,
    max_depth       = 10,
    min_samples_leaf= 10,
    min_samples_split=20,
    max_features    = 'sqrt',
    class_weight    = 'balanced',   # critical: 11% safe vs 89% unsafe
    random_state    = SEED,
    n_jobs          = -1
)
rf.fit(X_train_sc, y_train)
print(f"\n🌲 Trained RandomForest: 300 trees, depth=10, class_weight=balanced")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Evaluate
# ─────────────────────────────────────────────────────────────────────────────
train_acc  = accuracy_score(y_train, rf.predict(X_train_sc))
test_acc   = accuracy_score(y_test,  rf.predict(X_test_sc))
test_f1    = f1_score(y_test, rf.predict(X_test_sc))
test_roc   = roc_auc_score(y_test, rf.predict_proba(X_test_sc)[:, 1])
gap        = train_acc - test_acc

print(f"\n{'Metric':<25} | {'Value':<10}")
print("-" * 38)
print(f"{'Train Accuracy':<25} | {train_acc*100:.2f}%")
print(f"{'Test Accuracy':<25} | {test_acc*100:.2f}%")
print(f"{'Overfitting Gap':<25} | {gap*100:.2f}%")
print(f"{'F1 Score (Safe class)':<25} | {test_f1*100:.2f}%")
print(f"{'ROC-AUC':<25} | {test_roc:.4f}")
print("-" * 38)

if gap * 100 > 5:
    print(f"⚠️  Gap {gap*100:.1f}% > 5% — slight overfitting")
else:
    print(f"✅  Gap {gap*100:.1f}% < 5% — excellent generalization")

# ─────────────────────────────────────────────────────────────────────────────
# 8. Cross-validation (train set only)
# ─────────────────────────────────────────────────────────────────────────────
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_acc = cross_val_score(rf, X_train_sc, y_train, cv=skf, scoring='accuracy', n_jobs=-1)
cv_f1  = cross_val_score(rf, X_train_sc, y_train, cv=skf, scoring='f1',       n_jobs=-1)
cv_roc = cross_val_score(rf, X_train_sc, y_train, cv=skf, scoring='roc_auc',  n_jobs=-1)

print(f"\n📊 5-Fold CV (train set):")
print(f"   Accuracy: {cv_acc.mean()*100:.2f}% ± {cv_acc.std()*100:.2f}%")
print(f"   F1:       {cv_f1.mean()*100:.2f}%  ± {cv_f1.std()*100:.2f}%")
print(f"   ROC-AUC:  {cv_roc.mean():.4f}  ± {cv_roc.std():.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 9. Classification report + confusion matrix
# ─────────────────────────────────────────────────────────────────────────────
y_pred = rf.predict(X_test_sc)
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Unsafe (0)", "Safe (1)"]))

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print(f"Confusion Matrix:")
print(f"  True Unsafe (TN):  {tn}  | False Safe (FP):   {fp}")
print(f"  False Unsafe (FN): {fn}  | True Safe (TP):    {tp}")
print(f"  Unsafe Recall (safety): {tn/(tn+fp)*100:.1f}%  ← critical for public health")

# ─────────────────────────────────────────────────────────────────────────────
# 10. Feature importance
# ─────────────────────────────────────────────────────────────────────────────
importances = pd.Series(rf.feature_importances_, index=X_train_imp.columns)
importances = importances.sort_values(ascending=False)
print(f"\n🔥 Top 10 Feature Importances:")
print(importances.head(10).to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 11. Save
# ─────────────────────────────────────────────────────────────────────────────
out_dir = Path("data/processed/wq1_model")
out_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(rf,      out_dir / "model.pkl")
joblib.dump(scaler,  out_dir / "scaler.pkl")
joblib.dump(imputer, out_dir / "imputer.pkl")

feature_meta = {
    "original_features": list(df.drop('is_safe', axis=1).columns),
    "engineered_features": ["heavy_metal_load", "pathogen_risk",
                             "disinfect_proxy", "radio_composite", "mineral_excess"],
    "all_features": list(X_train_imp.columns),
    "target": "is_safe",
    "dataset": "waterQuality1.csv"
}
with open(out_dir / "feature_names.json", "w") as f:
    json.dump(feature_meta, f, indent=2)

summary = {
    "dataset":       "waterQuality1.csv",
    "n_samples":     len(df),
    "n_features":    X_train_imp.shape[1],
    "train_accuracy": round(train_acc, 4),
    "test_accuracy":  round(test_acc,  4),
    "overfitting_gap":round(gap,       4),
    "f1_score":       round(test_f1,   4),
    "roc_auc":        round(test_roc,  4),
    "cv_accuracy_mean": round(float(cv_acc.mean()), 4),
    "cv_accuracy_std":  round(float(cv_acc.std()),  4),
    "cv_roc_mean":      round(float(cv_roc.mean()), 4),
    "unsafe_recall_pct": round(tn/(tn+fp)*100, 2),
}
with open(out_dir / "performance_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n✅ Saved to: {out_dir}/")
print("   Files: model.pkl, scaler.pkl, imputer.pkl,")
print("          feature_names.json, performance_summary.json")
print("\n" + "=" * 70)
print("🎯 TRAINING COMPLETE")
print("=" * 70)
