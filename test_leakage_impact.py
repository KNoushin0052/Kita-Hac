import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv("G:/usr/EcoMed-AI/data/raw/waterQuality1.csv")
df = df.apply(pd.to_numeric, errors='coerce').dropna()

# Baseline (all features)
X = df.drop('is_safe', axis=1)
y = df['is_safe']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
base_acc = accuracy_score(y_test, rf.predict(X_test))
print(f"Full Feature Accuracy: {base_acc:.4f}")

# Removing top 5 correlated features (to see if accuracy collapses)
top_corr = ['aluminium', 'cadmium', 'chloramine', 'chromium', 'arsenic']
X_hard = X.drop(columns=top_corr)
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X_hard, y, test_size=0.2, random_state=42)

rf_hard = RandomForestClassifier(n_estimators=100, random_state=42)
rf_hard.fit(X_train_h, y_train_h)
hard_acc = accuracy_score(y_test_h, rf_hard.predict(X_test_h))
print(f"Without Top 5 Features Accuracy: {hard_acc:.4f}")
print(f"Dataset Baseline (Most Frequent): {y.value_counts(normalize=True).max():.4f}")
