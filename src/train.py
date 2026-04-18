"""
Stage 2: Model Training with MLflow Experiment Tracking
All configuration read from params.yaml.
"""

import os
import json
import yaml
import joblib
import sklearn
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def load_data(processed_dir: str):
    X_train = pd.read_csv(os.path.join(processed_dir, "X_train.csv"))
    X_test  = pd.read_csv(os.path.join(processed_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(processed_dir, "y_train.csv")).squeeze()
    y_test  = pd.read_csv(os.path.join(processed_dir, "y_test.csv")).squeeze()
    return X_train, X_test, y_train, y_test


def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
    m = {
        "accuracy":  round(accuracy_score(y_true, y_pred),                    4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0),  4),
        "recall":    round(recall_score(y_true, y_pred,    zero_division=0),   4),
        "f1_score":  round(f1_score(y_true, y_pred,        zero_division=0),   4),
    }
    if y_prob is not None:
        m["roc_auc"] = round(roc_auc_score(y_true, y_prob), 4)
    return m


def build_estimators(params: dict) -> dict:
    """Construct (estimator, param_grid) pairs entirely from params.yaml."""
    lr_p = params["logistic_regression"]
    rf_p = params["random_forest"]
    ab_p = params["adaboost"]

    return {
        "LogisticRegression": (
            LogisticRegression(
                C=lr_p["C"],
                max_iter=lr_p["max_iter"],
                random_state=lr_p["random_state"],
            ),
            lr_p["param_grid"],
        ),
        "RandomForest": (
            RandomForestClassifier(
                n_estimators=rf_p["n_estimators"],
                max_depth=rf_p["max_depth"],
                min_samples_split=rf_p["min_samples_split"],
                random_state=rf_p["random_state"],
            ),
            rf_p["param_grid"],
        ),
        "AdaBoost": (
            AdaBoostClassifier(
                n_estimators=ab_p["n_estimators"],
                learning_rate=ab_p["learning_rate"],
                random_state=ab_p["random_state"],
            ),
            ab_p["param_grid"],
        ),
    }


def train_and_log(
    model_name, estimator, param_grid,
    X_train, X_test, y_train, y_test,
    experiment_name, cv_folds, cv_scoring,
) -> tuple:
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=model_name) as run:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        gs = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=cv,
            scoring=cv_scoring,
            n_jobs=-1,
        )
        gs.fit(X_train, y_train)
        best = gs.best_estimator_

        mlflow.log_param("model_type",   model_name)
        mlflow.log_param("cv_folds",     cv_folds)
        mlflow.log_param("cv_scoring",   cv_scoring)
        mlflow.log_param("sklearn_version", sklearn.__version__)
        mlflow.log_params(gs.best_params_)

        y_pred = best.predict(X_test)
        y_prob = best.predict_proba(X_test)[:, 1] if hasattr(best, "predict_proba") else None
        metrics = compute_metrics(y_test, y_pred, y_prob)

        mlflow.log_metrics(metrics)
        mlflow.log_metric("cv_best_f1", round(gs.best_score_, 4))
        mlflow.sklearn.log_model(best, artifact_path="model")

        print(f"\n{'='*55}")
        print(f"  Model   : {model_name}")
        print(f"  Params  : {gs.best_params_}")
        print(f"  CV F1   : {gs.best_score_:.4f}")
        print(f"  Test    : {metrics}")
        print(f"  Run ID  : {run.info.run_id}")

        return run.info.run_id, metrics, best


def train_all(params: dict):
    data_cfg   = params["data"]
    train_cfg  = params["training"]
    mlflow_cfg = params["mlflow"]

    processed_dir   = data_cfg["processed_path"]
    champion_metric = train_cfg["champion_metric"]
    cv_folds        = train_cfg["cv_folds"]
    cv_scoring      = train_cfg["cv_scoring"]
    model_out       = train_cfg["model_output_path"]
    results_out     = train_cfg["results_output_path"]
    experiment_name = mlflow_cfg["experiment_name"]
    registry_name   = mlflow_cfg["model_registry_name"]

    X_train, X_test, y_train, y_test = load_data(processed_dir)
    print(f"[train] Train={X_train.shape}, Test={X_test.shape}")
    print(f"[train] sklearn version : {sklearn.__version__}")
    print(f"[train] Champion metric : {champion_metric}")

    models = build_estimators(params)
    results = {}

    for name, (est, grid) in models.items():
        run_id, metrics, best_model = train_and_log(
            name, est, grid,
            X_train, X_test, y_train, y_test,
            experiment_name, cv_folds, cv_scoring,
        )
        results[name] = {"run_id": run_id, "metrics": metrics, "model": best_model}

    # Select champion
    champion_name = max(results, key=lambda k: results[k]["metrics"].get(champion_metric, 0))
    champion      = results[champion_name]
    champ_score   = champion["metrics"].get(champion_metric, 0)

    print(f"\n{'='*55}")
    print(f"  Champion : {champion_name}  ({champion_metric}={champ_score})")

    # Save model + metadata bundle so version is always co-located with pkl
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    bundle = {
        "model":          champion["model"],
        "sklearn_version": sklearn.__version__,
        "model_name":     champion_name,
    }
    joblib.dump(bundle, model_out)
    print(f"[train] Champion bundle saved → '{model_out}'")

    # Save results summary
    summary = {
        "champion":         champion_name,
        "champion_run_id":  champion["run_id"],
        "champion_metric":  champion_metric,
        "champion_score":   champ_score,
        "model_path": model_out,
        "metrics_path": results_out,
        "champion_metrics": champion["metrics"],
        "sklearn_version":  sklearn.__version__,
        "all_results": {
            k: {"run_id": v["run_id"], "metrics": v["metrics"]}
            for k, v in results.items()
        },
        "status": "candidate"
    }
    os.makedirs(os.path.dirname(results_out), exist_ok=True)
    with open(results_out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[train] Summary saved  → '{results_out}'")

    # MLflow registry
    try:
        client    = MlflowClient()
        model_uri = f"runs:/{champion['run_id']}/model"
        mv        = mlflow.register_model(model_uri=model_uri, name=registry_name)
        client.set_registered_model_tag(registry_name, "champion", champion_name)
        client.set_model_version_tag(registry_name, mv.version, "stage", "production")
        print(f"[train] Registered '{registry_name}' version {mv.version}")
    except Exception as e:
        print(f"[train] Registry step skipped: {e}")

    print("\n[train] Done!")
    
    os.makedirs("models", exist_ok=True)
    # Simulate a simple registry
    registry_path = "models/results_summary.json"
    
    

    if os.path.exists(registry_path):
        with open(registry_path, "r") as f:
            current_champion = json.load(f)
        # Replace champion only if the new model has better F1
        if summary["champion_score"] > current_champion.get("champion_score", 0):
            summary["status"] = "champion"
            with open(registry_path, "w") as f:
                json.dump(summary, f, indent=4)
            print("New model promoted to champion.")
        else:
            print("Current champion retained.")
    else:
        summary["status"] = "champion"
        with open(registry_path, "w") as f:
            json.dump(summary, f, indent=4)
        print("No champion existed. New model set as champion.")
        print("Training complete.")
        print("Metrics:", metrics)   
    return summary


if __name__ == "__main__":
    train_all(load_params())
