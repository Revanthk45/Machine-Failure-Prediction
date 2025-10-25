import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from joblib import dump

CSV_PATH = 'device_failure.csv'
MODEL_OUT = 'device_failure_model.joblib'

print('Loading', CSV_PATH)
try:
    df = pd.read_csv(CSV_PATH)
except Exception as e:
    raise SystemExit('Failed to load CSV: ' + str(e))

if 'failure' not in df.columns:
    raise SystemExit('CSV does not contain "failure" column; cannot train device model.')

# drop rows with NaNs in target
df = df.dropna(subset=['failure']).copy()

# Select first 5 numeric feature columns (excluding target)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'failure' in numeric_cols:
    numeric_cols.remove('failure')

if len(numeric_cols) < 5:
    # fallback: use all numeric columns
    features = numeric_cols
else:
    features = numeric_cols[:5]

print('Using features for device model:', features)

X = df[features]
y = df['failure']

# simple pipeline: imputer + scaler + RF
pipe = make_pipeline(
    SimpleImputer(strategy='median'),
    StandardScaler(),
    RandomForestClassifier(n_estimators=50, random_state=42)
)

print('Training device model...')
pipe.fit(X, y)

print('Saving model to', MODEL_OUT)
dump(pipe, MODEL_OUT)
print('Done')
