"""
train.py — Mobile Price Prediction model trainer
=================================================
Replicates the training pipeline from Group 13.ipynb exactly.
Called automatically by the Streamlit apps when mobile_price_model.pkl
is not found (e.g. on first boot in Streamlit Cloud).
"""

import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# All paths are anchored to the directory this file lives in,
# so they resolve correctly regardless of the working directory
# (local terminal, Streamlit Cloud, etc.)
_HERE      = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_HERE, "mobile_price_model.pkl")
TRAIN_CSV  = os.path.join(_HERE, "Mobile_Price_Prediction_train.csv")


def train_and_save():
    df = pd.read_csv(TRAIN_CSV)

    # Encode ordinal feature mobile_wt (string → integer) if needed
    # pd.api.types.is_string_dtype covers both legacy 'object' and pandas 3.x ArrowStringArray
    if pd.api.types.is_string_dtype(df["mobile_wt"]):
        df["mobile_wt"] = df["mobile_wt"].map({"Low": 1, "Med": 2, "High": 3})

    # Encode ALL remaining string columns (catches Yes/No in any column)
    le = LabelEncoder()
    for col in df.columns:
        if col == "price_range":
            continue
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = le.fit_transform(df[col].astype(str))

    # Feature engineering — matches derived features used in the app
    df["screen_area"]      = df["px_height"] * df["px_width"]
    df["battery_per_core"] = df["battery_power"] / df["n_cores"]

    X = df.drop("price_range", axis=1)
    y = df["price_range"]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)
    return model


def load_or_train():
    """Return the model, training it first if the pkl file is missing."""
    if not os.path.exists(MODEL_PATH):
        train_and_save()
    return joblib.load(MODEL_PATH)
