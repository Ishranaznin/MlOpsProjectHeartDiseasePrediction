"""Generate the 4-page MLOps project report as PDF."""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

W, H = letter

def build_report(output_path):
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=0.85*inch,
        rightMargin=0.85*inch,
        topMargin=0.9*inch,
        bottomMargin=0.9*inch,
    )

    styles = getSampleStyleSheet()
    base = styles["Normal"]

    # Custom styles
    title_style = ParagraphStyle("ReportTitle", parent=base,
        fontSize=18, fontName="Helvetica-Bold", textColor=colors.HexColor("#1a237e"),
        alignment=TA_CENTER, spaceAfter=4)
    subtitle_style = ParagraphStyle("Subtitle", parent=base,
        fontSize=11, fontName="Helvetica", textColor=colors.HexColor("#555555"),
        alignment=TA_CENTER, spaceAfter=14)
    h1 = ParagraphStyle("H1", parent=base,
        fontSize=13, fontName="Helvetica-Bold", textColor=colors.HexColor("#1a237e"),
        spaceBefore=12, spaceAfter=4, borderPad=2)
    h2 = ParagraphStyle("H2", parent=base,
        fontSize=11, fontName="Helvetica-Bold", textColor=colors.HexColor("#283593"),
        spaceBefore=8, spaceAfter=3)
    body = ParagraphStyle("Body", parent=base,
        fontSize=10, fontName="Helvetica", leading=14, alignment=TA_JUSTIFY,
        spaceAfter=5)
    bullet = ParagraphStyle("Bullet", parent=base,
        fontSize=10, fontName="Helvetica", leading=13,
        leftIndent=16, spaceAfter=2, bulletIndent=6)
    mono = ParagraphStyle("Mono", parent=base,
        fontSize=8.5, fontName="Courier", leading=12,
        leftIndent=12, textColor=colors.HexColor("#212121"), spaceAfter=4)
    caption = ParagraphStyle("Caption", parent=base,
        fontSize=9, fontName="Helvetica-Oblique",
        textColor=colors.HexColor("#666666"), alignment=TA_CENTER, spaceAfter=4)

    divider = HRFlowable(width="100%", thickness=1,
                         color=colors.HexColor("#c5cae9"), spaceAfter=8, spaceBefore=4)

    def section(title):
        return [Paragraph(title, h1), divider]

    def sub(title):
        return Paragraph(title, h2)

    def p(text):
        return Paragraph(text, body)

    def b(text):
        return Paragraph(f"• {text}", bullet)

    story = []

    # ── PAGE 1 ──────────────────────────────────────────────────────────────
    story += [
        Spacer(1, 0.2*inch),
        Paragraph("End-to-End MLOps Pipeline", title_style),
        Paragraph("Heart Disease Classification — Group Project Report", subtitle_style),
        divider,
    ]

    story += section("1. Problem Description")
    story += [
        p("This project implements a complete MLOps pipeline for binary classification of heart "
          "disease using the Statlog Heart Disease dataset. The clinical objective is to predict "
          "whether a patient has heart disease (target = 1) or not (target = 0) based on 13 "
          "physiological and diagnostic features."),
        p("The dataset contains <b>270 patient records</b> with a mild class imbalance: "
          "150 negative cases (55.6%) and 120 positive cases (44.4%). Features include age, "
          "sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, "
          "resting ECG, maximum heart rate, exercise-induced angina, ST depression, slope, "
          "number of major vessels, and thalassemia type."),
        sub("1.1 ML Objective"),
        p("Train and compare multiple classifiers, track experiments systematically, package "
          "the best model into a production-ready inference API, and automate the entire "
          "workflow with CI/CD — matching real-world MLOps practices."),
        sub("1.2 Evaluation Metrics"),
        p("Given the clinical domain, <b>F1-score and ROC-AUC</b> are the primary metrics. "
          "Missing a positive case (false negative) is clinically costlier than a false alarm, "
          "making recall important. A minimum F1 threshold of 0.75 was set as the CI/CD quality gate."),
        Spacer(1, 8),
    ]

    story += section("2. System Architecture")
    story += [
        p("The pipeline follows a clear separation between training (offline) and inference "
          "(online), a core MLOps principle. The architecture consists of five interconnected layers:"),
        b("<b>Data Layer:</b> Raw CSV tracked and versioned by DVC. Preprocessing (StandardScaler, "
          "stratified 80/20 train-test split) produces versioned outputs in data/processed/."),
        b("<b>Training Layer:</b> Three scikit-learn models trained with GridSearchCV "
          "(5-fold stratified CV). All parameters and metrics logged to MLflow."),
        b("<b>Registry Layer:</b> Champion model selected by highest test F1-score, registered "
          "in MLflow Model Registry as 'HeartDiseaseChampion'."),
        b("<b>Serving Layer:</b> FastAPI application loads the champion model on startup. "
          "Exposes /predict (single) and /predict_batch endpoints with Pydantic validation."),
        b("<b>Orchestration Layer:</b> Docker container deployed via Kubernetes (2 replicas "
          "with liveness/readiness probes), exposed as a NodePort service on port 30080."),
        Spacer(1, 6),
        p("The DVC pipeline (dvc.yaml) defines three stages — preprocess → train → evaluate — "
          "with explicit dependencies and outputs, ensuring full reproducibility. Re-running "
          "<i>dvc repro</i> only re-executes stages whose inputs have changed."),
    ]

    story.append(PageBreak())

    # ── PAGE 2 ──────────────────────────────────────────────────────────────
    story += section("3. Tools Used and Justification")

    tools_data = [
        ["Tool", "Role", "Justification"],
        ["DVC 3.x", "Data & pipeline versioning",
         "Git-native, supports remote storage (S3/GCS), reproducible stages"],
        ["MLflow 2.x", "Experiment tracking + Model Registry",
         "Open-source, UI for run comparison, built-in model lifecycle management"],
        ["scikit-learn 1.4", "ML models + preprocessing",
         "Production-grade implementations, GridSearchCV for tuning"],
        ["FastAPI 0.111", "Inference REST API",
         "Async-capable, auto OpenAPI docs, Pydantic validation"],
        ["Docker", "Containerization",
         "Reproducible runtime, multi-stage build for minimal image size"],
        ["Kubernetes", "Container orchestration",
         "Declarative deployment, auto-restart, horizontal scaling ready"],
        ["GitHub Actions", "CI/CD automation",
         "Native to GitHub, matrix builds, artifact passing between jobs"],
        ["pytest", "Automated testing",
         "Validates API contracts before Docker build in CI pipeline"],
    ]
    t = Table(tools_data, colWidths=[1.1*inch, 1.5*inch, 3.5*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1a237e")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 8.5),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#f5f5f5"), colors.white]),
        ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#bdbdbd")),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("PADDING", (0,0), (-1,-1), 5),
        ("FONTNAME", (0,1), (0,-1), "Helvetica-Bold"),
    ]))
    story += [t, Spacer(1, 12)]

    story += section("4. Model Results")
    story += [
        p("Three models were trained with 5-fold stratified cross-validation GridSearchCV. "
          "Results on the held-out test set (54 samples):"),
    ]

    results_data = [
        ["Model", "Best Params", "Accuracy", "F1-Score", "ROC-AUC", "Champion"],
        ["Logistic Regression", "C=0.01, solver=liblinear", "85.2%", "0.840", "0.911", "✓"],
        ["Random Forest", "n_est=100, max_depth=5", "81.5%", "0.800", "0.885", ""],
        ["AdaBoost", "n_est=50, lr=1.5", "79.6%", "0.784", "0.882", ""],
    ]
    r = Table(results_data, colWidths=[1.35*inch, 1.85*inch, 0.75*inch, 0.75*inch, 0.75*inch, 0.65*inch])
    r.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#283593")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 8.5),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#e8eaf6"), colors.white]),
        ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#bdbdbd")),
        ("ALIGN", (2,0), (-1,-1), "CENTER"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("PADDING", (0,0), (-1,-1), 5),
        ("FONTNAME", (0,1), (0,1), "Helvetica-Bold"),
        ("BACKGROUND", (5,1), (5,1), colors.HexColor("#c8e6c9")),
    ]))
    story += [r, Spacer(1, 8),
        p("<b>Logistic Regression</b> achieved the best test F1 (0.840) and ROC-AUC (0.911), "
          "outperforming the ensemble methods on this relatively small dataset. "
          "The champion model was registered in MLflow Model Registry as 'HeartDiseaseChampion' "
          "version 1, tagged as 'production'."),
    ]

    story.append(PageBreak())

    # ── PAGE 3 ──────────────────────────────────────────────────────────────
    story += section("5. MLOps Pipeline Implementation")

    story += [
        sub("5.1 DVC Pipeline (dvc.yaml)"),
        p("Three stages are declared with explicit deps/outs, enabling cached, incremental re-runs. "
          "The scaler (scaler.pkl) is an output of <i>preprocess</i> and a dependency of "
          "inference — ensuring the same transformation is used in training and serving."),
        Paragraph("dvc repro    # runs only changed stages", mono),
        Paragraph("dvc dag      # visualises the DAG", mono),

        sub("5.2 MLflow Experiment Tracking"),
        p("Each model run logs: model type, best hyperparameters, CV F1, and five test metrics "
          "(accuracy, precision, recall, F1, ROC-AUC). The MLflow UI at localhost:5000 allows "
          "visual run comparison. The champion is programmatically registered via MlflowClient."),

        sub("5.3 FastAPI Inference API"),
        p("The API exposes four endpoints. Input validation is enforced by Pydantic range constraints "
          "(e.g., age 1–120, chol 100–600) — invalid inputs return HTTP 422 automatically. "
          "The /predict_batch endpoint allows efficient bulk scoring."),
    ]

    api_data = [
        ["Endpoint", "Method", "Description"],
        ["/health", "GET", "Liveness check — returns model type"],
        ["/model_info", "GET", "Returns champion metrics from results_summary.json"],
        ["/predict", "POST", "Single patient prediction with probabilities"],
        ["/predict_batch", "POST", "Batch predictions for a list of patients"],
    ]
    at = Table(api_data, colWidths=[1.4*inch, 0.7*inch, 4.0*inch])
    at.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#37474f")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 8.5),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#eceff1"), colors.white]),
        ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#bdbdbd")),
        ("FONTNAME", (0,1), (0,-1), "Courier"),
        ("PADDING", (0,0), (-1,-1), 5),
    ]))
    story += [at, Spacer(1, 8)]

    story += [
        sub("5.4 Docker Containerization"),
        p("A multi-stage Dockerfile is used: the builder stage installs Python dependencies, "
          "the runtime stage copies only the installed packages and application files. "
          "The container runs as a non-root user (appuser) for security. A HEALTHCHECK "
          "polls /health every 30 seconds."),
        Paragraph("docker build -t heart-disease-api .", mono),
        Paragraph("docker run -p 8000:8000 heart-disease-api", mono),

        sub("5.5 Kubernetes Deployment"),
        p("The k8s/ directory contains two manifests. The Deployment configures 2 replicas "
          "with resource limits (256Mi RAM, 250m CPU per pod), liveness probes (polling /health "
          "every 20s), and readiness probes (every 10s). The NodePort Service exposes port 30080 "
          "externally on the Minikube node."),
        Paragraph("minikube start", mono),
        Paragraph("kubectl apply -f k8s/", mono),
        Paragraph("curl http://$(minikube ip):30080/health", mono),

        sub("5.6 GitHub Actions CI/CD"),
        p("The workflow (.github/workflows/mlops_pipeline.yml) has three jobs running on every "
          "push to main or PR. Jobs run sequentially with artifact passing:"),
        b("<b>train:</b> Installs deps → runs preprocess.py → train.py → evaluate.py. "
          "Fails the pipeline if F1 < 0.75. Uploads model artifacts."),
        b("<b>docker:</b> Downloads artifacts → builds Docker image → smoke-tests container. "
          "Blocked until 'train' passes."),
        b("<b>test:</b> Downloads artifacts → runs pytest. All 6 API tests must pass."),
    ]

    story.append(PageBreak())

    # ── PAGE 4 ──────────────────────────────────────────────────────────────
    story += section("6. Repository Structure")

    repo_text = (
        "heart-disease-mlops/\n"
        "├── data/raw/heart_disease.csv      ← DVC tracked\n"
        "├── data/processed/                 ← DVC outputs\n"
        "│   ├── X_train.csv, X_test.csv\n"
        "│   ├── y_train.csv, y_test.csv\n"
        "│   └── scaler.pkl\n"
        "├── src/\n"
        "│   ├── preprocess.py               ← Stage 1\n"
        "│   ├── train.py                    ← Stage 2 (MLflow)\n"
        "│   ├── evaluate.py                 ← Stage 3 (CI gate)\n"
        "│   └── app.py                      ← FastAPI server\n"
        "├── models/\n"
        "│   ├── champion_model.pkl\n"
        "│   ├── results_summary.json\n"
        "│   └── evaluation.json\n"
        "├── k8s/deployment.yaml\n"
        "├── k8s/service.yaml\n"
        "├── .github/workflows/mlops_pipeline.yml\n"
        "├── tests/test_api.py               ← 6 pytest tests\n"
        "├── dvc.yaml                        ← Pipeline DAG\n"
        "├── params.yaml                     ← All hyperparameters\n"
        "├── Dockerfile\n"
        "└── requirements.txt"
    )
    story.append(Paragraph(repo_text.replace("\n", "<br/>"), mono))
    story.append(Spacer(1, 8))

    story += section("7. Challenges Faced")
    story += [
        sub("7.1 DVC + MLflow Integration"),
        p("DVC manages data/model artifacts on disk while MLflow tracks experiment metadata "
          "and model binaries in mlruns/. Avoiding double-tracking of the champion model "
          "required careful configuration: DVC owns data/processed/ and models/champion_model.pkl, "
          "while MLflow owns experiment runs and the model registry."),

        sub("7.2 Scaler in Training vs. Inference"),
        p("A common MLOps pitfall is applying different preprocessing at training and inference "
          "time (training-serving skew). This was addressed by saving the fitted StandardScaler "
          "as scaler.pkl via DVC and loading the same artifact in the FastAPI server on startup."),

        sub("7.3 FastAPI Lifespan Events"),
        p("The deprecated @app.on_event('startup') pattern does not trigger in FastAPI's "
          "TestClient unless a context manager is used. The app was updated to use the modern "
          "@asynccontextmanager lifespan pattern, and tests were fixed to use 'with TestClient(app)'."),

        sub("7.4 Kubernetes Local Image Access"),
        p("Minikube runs its own Docker daemon. To use a locally built image without pushing "
          "to a registry, the image must be built inside Minikube's Docker context "
          "('eval $(minikube docker-env)') and imagePullPolicy set to 'IfNotPresent'."),
    ]

    story += section("8. MLOps Best Practices Demonstrated")
    story += [
        b("<b>Reproducibility:</b> dvc repro re-creates any artifact from scratch given the same params.yaml."),
        b("<b>Version Control:</b> Code in Git, data/models in DVC, experiments in MLflow — all linked by commit hash."),
        b("<b>Automated Pipelines:</b> No manual steps — GitHub Actions orchestrates preprocessing, training, evaluation, and Docker build end-to-end."),
        b("<b>Training/Inference Separation:</b> Training code (src/train.py) is never imported by the serving code (src/app.py). Only the serialized artifact crosses the boundary."),
        b("<b>Quality Gate:</b> evaluate.py exits with code 1 if F1 < 0.75, blocking the Docker build job in CI."),
        b("<b>Documentation:</b> README with quickstart, inline docstrings, auto-generated OpenAPI docs at /docs."),
        Spacer(1, 10),
        HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#9e9e9e"), spaceAfter=6),
        Paragraph("End of Report", caption),
    ]

    doc.build(story)
    print(f"Report saved to {output_path}")


if __name__ == "__main__":
    import os
    os.makedirs("reports", exist_ok=True)
    build_report("reports/MLOps_Project_Report.pdf")
