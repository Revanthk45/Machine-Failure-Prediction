import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from joblib import dump

CSV_PATH = 'factory_data (classification).csv'
MODEL_OUT = 'machine_failure_model.joblib'

print('Loading', CSV_PATH)
try:
    df = pd.read_csv(CSV_PATH)
except Exception as e:
    raise SystemExit('Failed to load CSV: ' + str(e))

# Minimal cleaning similar to notebook
if 'Unique ID' in df.columns:
    df = df.drop(['Unique ID'], axis=1)

# rename column if present
if 'Tool Wear (min)' in df.columns:
    df = df.rename(columns={'Tool Wear (min)': 'Tool Lifespan (min)'})

# drop rows where key numeric columns are zero
numeric_check_cols = ['Ambient T (C)', 'Process T (C)', 'Rotation Speed (rpm)', 'Torque (Nm)', 'Tool Lifespan (min)']
for c in numeric_check_cols:
    if c not in df.columns:
        raise SystemExit(f'Missing expected column: {c}')

# remove rows with zeros in those columns
for c in numeric_check_cols:
    df = df[df[c] != 0]

# Feature engineering
# Product Code: first char of Product ID if exists
if 'Product ID' in df.columns:
    df['Product Code'] = df['Product ID'].astype(str).str[0]
elif 'Product Code' not in df.columns:
    # create a placeholder categorical if missing
    df['Product Code'] = 'X'

# new features
df['T Difference Squared (C^2)'] = (df['Ambient T (C)'] - df['Process T (C)']) ** 2
# avoid division by zero
df['Tool Lifespan/Temp Increase^2 (min/C^2)'] = df['Tool Lifespan (min)'] / (df['T Difference Squared (C^2)'].replace(0, np.nan))
df['Tool Lifespan/Temp Increase^2 (min/C^2)'] = df['Tool Lifespan/Temp Increase^2 (min/C^2)'].fillna(df['Tool Lifespan (min)'])

# Horsepower
df['Horsepower (HP)'] = (df['Rotation Speed (rpm)'] * df['Torque (Nm)']) / 5252.0

# target
if 'Machine Status' not in df.columns:
    raise SystemExit('Missing target column Machine Status')

# select features consistent with notebook
feature_cols = [
    'Quality', 'Product Code', 'Process T (C)', 'Ambient T (C)',
    'T Difference Squared (C^2)', 'Tool Lifespan (min)',
    'Tool Lifespan/Temp Increase^2 (min/C^2)', 'Rotation Speed (rpm)',
    'Torque (Nm)', 'Horsepower (HP)'
]

for c in feature_cols:
    if c not in df.columns:
        # if Quality missing, try Quality if different naming
        df[c] = 0

X = df[feature_cols].copy()
y = df['Machine Status'].astype(int)

# encode categorical columns
cat_cols = ['Quality', 'Product Code']
for c in cat_cols:
    le = LabelEncoder()
    X[c] = le.fit_transform(X[c].astype(str))

# simple pipeline
pipe = make_pipeline(
    SimpleImputer(strategy='median'),
    RobustScaler(),
    GradientBoostingClassifier(random_state=42, learning_rate=0.1, n_estimators=200, max_depth=10, min_samples_split=600, min_samples_leaf=10)
)

print('Training machine failure model...')
pipe.fit(X, y)

print('Saving model to', MODEL_OUT)
dump(pipe, MODEL_OUT)
