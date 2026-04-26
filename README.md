# Heart Disease MLOps Pipeline

> End-to-end MLOps system for binary heart disease classification — covering data versioning, experiment tracking, model registry, containerised deployment, monitoring prediction, Kubernetes orchestration, and CI/CD automation.

**Dataset:** UCI Heart Disease · 270 patients · 13 clinical features  
**Problem:** Binary classification (heart disease: yes / no)  
**Champion Model:** Logistic Regression · F1 = 0.84 · ROC-AUC = 0.911

---

## Project Structure

```
heart-disease-mlops/
├── data/
│   ├── raw/                  # Source dataset (DVC-tracked)
│   └── processed/            # Scaled train/test splits + scaler.pkl
├── src/
│   ├── preprocess.py         # Stage 1 — data cleaning & splitting
│   ├── train.py              # Stage 2 — training + MLflow logging
│   ├── evaluate.py           # Stage 3 — evaluation & quality gate
│   ├── app.py                # FastAPI inference server
│   ├── streamlit_app.py      # Streamlit UI
│   └── model_utils.py        # Shared model loading utilities
├── models/
│   ├── champion_model.pkl    # Best model bundle (DVC-tracked)
│   ├── results_summary.json  # All model metrics
│   └── evaluation.json       # Final evaluation output
├── k8s/
│   ├── deployment.yaml       # Kubernetes Deployment
│   ├── service.yaml          # NodePort Service
│   ├── prometheus-deployment.yaml
│   └── grafana-deployment.yaml
├── monitoring/
│   ├── prometheus.yml        # Prometheus scrape config
│   └── grafana/provisioning/ # Auto-provisioned dashboards & datasources
├── .github/workflows/
│   └── mlops_pipeline.yml    # CI/CD — train → evaluate → Docker push
├── dvc.yaml                  # DVC pipeline definition
├── params.yaml               # Single source of truth for all config
├── Dockerfile
└── docker-compose.yml
```

---

## Prerequisites

- Python 3.11
- Docker & Docker Compose
- Minikube (for Kubernetes)
- Git + DVC

---

## Quickstart — Local

```bash
# 1. Clone and install dependencies
git clone <repo-url>
cd heart-disease-mlops
pip install -r requirements.txt

# 2. Run the full ML pipeline (preprocess → train → evaluate)
dvc repro

# 3. Launch the Streamlit app locally
streamlit run src/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501), fill in the patient details in the sidebar, and click **Predict Risk**.

---

## Pipeline Stages (DVC)

The pipeline is defined in `dvc.yaml` and driven entirely by `params.yaml` — no hardcoded values anywhere.

| Stage | Script | Key Outputs |
|-------|--------|-------------|
| `preprocess` | `src/preprocess.py` | `data/processed/` splits + `scaler.pkl` |
| `train` | `src/train.py` | `models/champion_model.pkl`, `results_summary.json` |
| `evaluate` | `src/evaluate.py` | `models/evaluation.json` (fails if F1 < 0.75) |

To reproduce from scratch:

```bash
dvc repro
```

To run individual stages:

```bash
python src/preprocess.py
python src/train.py
python src/evaluate.py
```

---

## Experiment Tracking (MLflow)

All training runs are logged automatically. Three models are trained with GridSearchCV (5-fold StratifiedKFold):

- Logistic Regression
- Random Forest
- AdaBoost

To view the experiment dashboard:

```bash
mlflow ui --port 5000
```

Open [http://localhost:5000](http://localhost:5000) to compare runs, parameters, and metrics. The champion model is selected by `f1_score` (configurable via `params.yaml → training.champion_metric`) and registered in the MLflow model registry as **HeartDiseaseChampion** with tag `stage=production`.

---

## Model Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Logistic Regression** ★ | 0.8519 | 0.8077 | 0.8750 | **0.8400** | 0.9111 |
| Random Forest | 0.8148 | 0.7692 | 0.8333 | 0.8000 | 0.8847 |
| AdaBoost | 0.7407 | 0.6923 | 0.7500 | 0.7200 | 0.8556 |

★ Champion · Pass threshold: F1 ≥ 0.75

---

## Docker — Full Stack (Compose)

Runs the Streamlit app, Prometheus, and Grafana together:

```bash
docker compose up --build
```

| Service | URL | Notes |
|---------|-----|-------|
| Streamlit UI | http://localhost:8501 | Main prediction interface |
| Prometheus | http://localhost:9090 | Metrics scraping |
| Grafana | http://localhost:3000 | Dashboards (admin / admin) |

To stop:

```bash
docker compose down
```

> Grafana auto-provisions dashboards and the Prometheus datasource on startup via the `monitoring/grafana/provisioning/` mount.

---

## Kubernetes (Minikube)

### Start the cluster and deploy

```bash
minikube start

# Pull image and apply all manifests
kubectl apply -f k8s/
```

### Verify deployment

```bash
kubectl get pods
kubectl get svc
```

### Access the app

```bash
minikube service heart-disease-streamlit-service --url
```

If the URL is not reachable directly, use port-forward:

```bash
kubectl port-forward svc/heart-disease-streamlit-service 8501:8501
```

Open [http://localhost:8501](http://localhost:8501).

### Access Prometheus and Grafana

```bash
# Prometheus
kubectl port-forward svc/prometheus 9090:9090

# Grafana
kubectl port-forward svc/grafana 3000:3000
```

> **Note:** In Kubernetes, Grafana provisioning files are embedded in ConfigMaps rather than mounted from the host. The datasource ConfigMap sets `uid: prometheus` and all dashboard panel targets reference the datasource as `{type: prometheus, uid: prometheus}` to comply with how newer Grafana versions resolve datasources.

---

## CI/CD (GitHub Actions)

Every push to `master` triggers the full pipeline automatically:

```
Push to master
      ↓
Install dependencies
      ↓
Run preprocessing → training → evaluation
      ↓
Quality gate (F1 ≥ 0.75 — pipeline fails if not met)
      ↓
Build Docker image with champion model
      ↓
Push to Docker Hub (ishranaznin/heart-disease-api:latest)
```

The workflow is defined in `.github/workflows/mlops_pipeline.yml`. The Docker build job has an explicit `needs: train` dependency, so a failing model blocks image publication. Secrets required: `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`.

---

## Monitoring

Every prediction made through the Streamlit UI increments a Prometheus counter `predictions_total`. Prometheus scrapes the metrics endpoint on port `8001` every 15 seconds.

To query prediction rate in the Prometheus UI ([http://localhost:9090](http://localhost:9090)):

```promql
rate(predictions_total[5m]) * 60
```

The Grafana dashboard (**Dashboards → Heart Disease MLOps**) shows:

- **Total predictions** — running count since container start
- **Prediction rate** — predictions per minute over time

---

## Configuration

All pipeline behaviour is controlled through `params.yaml`. Key sections:

| Section | Controls |
|---------|----------|
| `data` | Raw/processed paths, test split size, target column |
| `logistic_regression / random_forest / adaboost` | Model hyperparameters and GridSearchCV grids |
| `training` | CV folds, scoring metric, champion selection metric |
| `evaluate` | Pass threshold (default F1 ≥ 0.75) |
| `mlflow` | Experiment name, registry name, tracking URI |
| `api` | FastAPI host, port, model/scaler paths |

---

## Docker Image

The pre-built image is available on Docker Hub:

```bash
docker pull ishranaznin/heart-disease-api:latest
```
