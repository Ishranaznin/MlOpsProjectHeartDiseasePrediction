# Heart Disease MLOps Pipeline

An end-to-end MLOps pipeline for heart disease classification using the Statlog Heart Disease dataset.

## Problem Statement
Binary classification: predict whether a patient has heart disease (`target=1`) or not (`target=0`).  
Dataset: 270 samples, 13 features, 2 classes (150 negative, 120 positive).

## Project Structure
```
heart-disease-mlops/
├── data/
│   ├── raw/               # Raw dataset (DVC tracked)
│   └── processed/         # Preprocessed data (DVC tracked)
├── src/
│   ├── preprocess.py      # Data preprocessing
│   ├── train.py           # Model training + MLflow logging
│   ├── evaluate.py        # Model evaluation
│   └── app.py             # FastAPI inference server
├── models/                # Saved model artifacts
├── k8s/                   # Kubernetes manifests
│   ├── deployment.yaml
│   └── service.yaml
├── .github/workflows/     # GitHub Actions CI/CD
│   └── mlops_pipeline.yml
├── dvc.yaml               # DVC pipeline definition
├── params.yaml            # Hyperparameters
├── Dockerfile
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Clone & Install
```bash
git clone <your-repo-url>
cd heart-disease-mlops
pip install -r requirements.txt
```

### 2. Run DVC Pipeline
```bash
dvc init
dvc repro
```

### 3. Start MLflow UI
```bash
mlflow ui --port 5000
# Open http://localhost:5000
```

### 4. Build & Run Docker
```bash
docker build -t heart-disease-api .
docker run -p 8000:8000 heart-disease-api
# API docs: http://localhost:8000/docs
```

### 5. Deploy on Kubernetes (Minikube)
```bash
minikube start
kubectl apply -f k8s/
kubectl get services heart-disease-service
```

## API Usage
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,
       "restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,
       "ca":0,"thal":1}'
```

## Models Trained
- Logistic Regression
- Random Forest
- AdaBoost

Champion model is selected based on F1-score and automatically registered in MLflow.

# Always retrain after unzipping — never use the pre-built pkl
pip install -r requirements.txt
python src/preprocess.py
python src/train.py        # saves pkl with YOUR sklearn version
python src/evaluate.py

# Then run Streamlit
streamlit run src/streamlit_app.py
