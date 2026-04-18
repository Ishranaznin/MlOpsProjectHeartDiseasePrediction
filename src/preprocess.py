"""
Stage 1: Data Preprocessing
All configuration is read from params.yaml — no hardcoded values.
"""

import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def preprocess(params: dict):
    cfg      = params["data"]
    raw_path = cfg["raw_path"]
    out_dir  = cfg["processed_path"]
    target   = cfg["target_column"]
    test_sz  = cfg["test_size"]
    seed     = cfg["random_state"]

    os.makedirs(out_dir, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────
    df = pd.read_csv(raw_path)
    print(f"[preprocess] Loaded {df.shape[0]} rows, {df.shape[1]} columns from '{raw_path}'")

    # ── Validate ──────────────────────────────────────────────────────────
    assert df.isnull().sum().sum() == 0, "Dataset contains null values!"
    assert target in df.columns, f"'{target}' column missing!"

    # ── Split features / target ───────────────────────────────────────────
    X = df.drop(columns=[target])
    y = df[target]
    feature_names = list(X.columns)

    print(f"[preprocess] Features ({len(feature_names)}): {feature_names}")
    print(f"[preprocess] Class distribution: {y.value_counts().to_dict()}")

    # ── Train / test split ────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_sz, random_state=seed, stratify=y
    )
    print(f"[preprocess] Train={X_train.shape[0]}, Test={X_test.shape[0]}  "
          f"(test_size={test_sz}, random_state={seed})")

    # ── Scale ─────────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names)
    X_test_s  = pd.DataFrame(scaler.transform(X_test),      columns=feature_names)

    # ── Save ──────────────────────────────────────────────────────────────
    X_train_s.to_csv(os.path.join(out_dir, "X_train.csv"), index=False)
    X_test_s.to_csv( os.path.join(out_dir, "X_test.csv"),  index=False)
    y_train.reset_index(drop=True).to_csv(os.path.join(out_dir, "y_train.csv"), index=False)
    y_test.reset_index(drop=True).to_csv( os.path.join(out_dir, "y_test.csv"),  index=False)
    joblib.dump(scaler, os.path.join(out_dir, "scaler.pkl"))

    print(f"[preprocess] Saved processed data and scaler to '{out_dir}'")


if __name__ == "__main__":
    preprocess(load_params())
