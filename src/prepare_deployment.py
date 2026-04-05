import json
import shutil
from pathlib import Path

registry_path = Path("models/results_summary.json")
deployment_dir = Path("deployment")


deployment_dir.mkdir(parents=True, exist_ok=True)
if not registry_path.exists():
    raise FileNotFoundError("models/results_summary.json not found")

with open(registry_path, "r") as f:
    champion = json.load(f)
model_source = Path(champion["model_path"])

if not model_source.exists():
    raise FileNotFoundError(f"Champion model not found at {model_source}")

# Copy champion model into deployment folder
shutil.copy2(model_source, deployment_dir / "model.pkl")

# Copy API app into deployment folder
shutil.copy2("src/app.py", deployment_dir / "app.py")

print("Deployment files prepared successfully.")
print(f"Champion model copied from: {model_source}")
print("Files ready inside deployment/")