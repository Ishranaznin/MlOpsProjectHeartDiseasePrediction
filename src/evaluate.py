"""
Stage 3: Model Evaluation
All configuration read from params.yaml — no hardcoded values.
"""

import os
import sys
import json
import yaml
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, roc_auc_score
)
import sys, os as _os; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__))); from model_utils import load_model_bundle


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def evaluate():
    params    = load_params()
    data_cfg  = params["data"]
    eval_cfg  = params["evaluate"]

    processed_dir  = data_cfg["processed_path"]
    class_names    = data_cfg["class_names"]
    model_path     = eval_cfg["model_path"]
    output_path    = eval_cfg["output_path"]
    pass_threshold = eval_cfg["pass_threshold"]

    X_test = pd.read_csv(os.path.join(processed_dir, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(processed_dir, "y_test.csv")).squeeze()

    try:
        model, model_name, sklearn_ver = load_model_bundle(model_path)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"[evaluate] ERROR: {e}")
        sys.exit(1)

    print(f"[evaluate] Model      : {model_name}  (sklearn {sklearn_ver})")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None
    cm  = confusion_matrix(y_test, y_pred).tolist()

    print("\n" + "="*55)
    print("EVALUATION REPORT")
    print("="*55)
    print(classification_report(y_test, y_pred, target_names=class_names))
    print(f"Confusion Matrix : {cm}")
    print(f"Accuracy         : {acc:.4f}")
    print(f"F1 Score         : {f1:.4f}")
    if auc:
        print(f"ROC-AUC          : {auc:.4f}")
    print(f"Pass Threshold   : {pass_threshold}  (from params.yaml)")

    result = {
        "accuracy":         round(acc, 4),
        "f1_score":         round(f1, 4),
        "roc_auc":          round(auc, 4) if auc else None,
        "confusion_matrix": cm,
        "pass_threshold":   pass_threshold,
        "passed":           bool(f1 >= pass_threshold),
    }
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as fp:
        json.dump(result, fp, indent=2)
    print(f"\n[evaluate] Results saved → '{output_path}'")

    if f1 < pass_threshold:
        print(f"[evaluate] FAIL  F1={f1:.4f} < threshold={pass_threshold}")
        sys.exit(1)
    else:
        print(f"[evaluate] PASS  F1={f1:.4f} >= threshold={pass_threshold}")
        sys.exit(0)


if __name__ == "__main__":
    evaluate()
