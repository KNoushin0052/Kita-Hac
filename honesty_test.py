import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data 
df = pd.read_csv("G:/usr/EcoMed-AI/data/raw/waterQuality1.csv")
df = df.apply(pd.to_numeric, errors='coerce').dropna()

# Full Features
X = df.drop('is_safe', axis=1)
y = df['is_safe']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SEVERE Regularization (to reach "honest" levels)
rf_hon = RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_leaf=50, random_state=42)
rf_hon.fit(X_train, y_train)
hon_acc = accuracy_score(y_test, rf_hon.predict(X_test))

print(f"Severely Regularized Accuracy (Honest Mode): {hon_acc:.4f}")
print(f"Previous 95% Model Accuracy: ~0.9481")
