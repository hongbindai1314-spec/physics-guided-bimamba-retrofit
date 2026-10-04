#!/usr/bin/env python
"""Evaluate the archived Ridge screening surrogate on the first 5,000 model-input rows."""
from pathlib import Path
import pandas as pd, joblib

ROOT = Path(__file__).resolve().parents[1]

df = pd.read_csv(ROOT / 'data/processed/model_55d.csv.gz', nrows=5000)
X = df[[c for c in df.columns if c.startswith('feat_')]]
model = joblib.load(ROOT / 'results/models/ridge_surrogate.joblib')
print(model.predict(X.iloc[:-1])[:10])
