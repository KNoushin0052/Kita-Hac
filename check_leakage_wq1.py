import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv("G:/usr/EcoMed-AI/data/raw/waterQuality1.csv")

# Ensure target is numeric
df['is_safe'] = pd.to_numeric(df['is_safe'], errors='coerce')

# Drop rows where target is NaN (if any)
df = df.dropna(subset=['is_safe'])

# Check correlation with target
correlations = df.corr()['is_safe'].abs().sort_values(ascending=False)

print("Correlation with 'is_safe':")
print(correlations)

# Check for features that are constant or almost perfectly 1/0
for col in df.columns:
    if col != 'is_safe':
        n_unique = df[col].nunique()
        if n_unique < 10:
             print(f"\nPotential leaky categorical feature: {col}")
             print(df.groupby(col)['is_safe'].mean())
