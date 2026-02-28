import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data 
df = pd.read_csv("G:/usr/EcoMed-AI/data/raw/waterQuality1.csv")
df = df.apply(pd.to_numeric, errors='coerce').dropna()

# RAW Features ONLY (No engineering)
X = df.drop('is_safe', axis=1)
y = df['is_safe']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normal RF
rf_raw = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
rf_raw.fit(X_train, y_train)
raw_acc = accuracy_score(y_test, rf_raw.predict(X_test))

print(f"Raw Feature Accuracy (Realistic): {raw_acc:.4f}")
