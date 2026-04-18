"""
run.py — Single entrypoint for the entire project.

What it does automatically:
  1. Checks which pipeline stages are stale (missing outputs or changed inputs)
  2. Runs only the stages that need to run (preprocess → train → evaluate)
  3. Launches the Streamlit UI

Usage:
    python run.py              # auto-pipeline + launch Streamlit
    python run.py --pipeline   # only run pipeline, no Streamlit
    python run.py --ui         # only launch Streamlit (pipeline must have run before)
    python run.py --force      # force re-run full pipeline even if outputs exist
"""

import os
import sys
import argparse
import subprocess
import hashlib
import json
import yaml

# ── Resolve project root regardless of where script is called from ───────────
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(ROOT, "src")

def load_params():
    with open(os.path.join(ROOT, "params.yaml")) as f:
        return yaml.safe_load(f)

# ── Helpers ───────────────────────────────────────────────────────────────────
def file_hash(path):
    """MD5 of a file — used to detect if inputs changed."""
    if not os.path.exists(path):
        return None
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()

def all_exist(*paths):
    return all(os.path.exists(p) for p in paths)

def run_stage(label, script, check_outputs):
    """Run a pipeline stage script and print status."""
    print(f"\n{'='*55}")
    print(f"  RUNNING: {label}")
    print(f"{'='*55}")
    result = subprocess.run(
        [sys.executable, os.path.join(SRC, script)],
        cwd=ROOT,
    )
    if result.returncode != 0:
        print(f"\n[run.py]   {label} failed. Fix the error above and retry.")
        sys.exit(1)
    print(f"[run.py]  {label} complete.")

def stage_is_stale(outputs, inputs, cache_file):
    """
    Returns True if any output is missing OR any input has changed since
    last run (tracked via a small JSON cache file).
    This is a lightweight DVC-style staleness check.
    """
    # Missing outputs → always stale
    if not all_exist(*outputs):
        return True

    # Load previous input hashes
    if not os.path.exists(cache_file):
        return True
    with open(cache_file) as f:
        cached = json.load(f)

    for inp in inputs:
        current = file_hash(inp)
        if cached.get(inp) != current:
            return True
    return False

def save_hashes(inputs, cache_file):
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    hashes = {inp: file_hash(inp) for inp in inputs}
    with open(cache_file, "w") as f:
        json.dump(hashes, f, indent=2)

# ── Pipeline stages ───────────────────────────────────────────────────────────
def run_pipeline(force=False):
    params = load_params()
    data   = params["data"]
    train  = params["training"]
    ev     = params["evaluate"]

    raw_csv       = os.path.join(ROOT, data["raw_path"])
    processed_dir = os.path.join(ROOT, data["processed_path"])
    model_path    = os.path.join(ROOT, train["model_output_path"])
    results_path  = os.path.join(ROOT, train["results_output_path"])
    eval_path     = os.path.join(ROOT, ev["output_path"])
    params_file   = os.path.join(ROOT, "params.yaml")

    cache_dir = os.path.join(ROOT, ".pipeline_cache")

    # ── Stage 1: Preprocess ──────────────────────────────────────────────
    preprocess_outputs = [
        os.path.join(processed_dir, "X_train.csv"),
        os.path.join(processed_dir, "X_test.csv"),
        os.path.join(processed_dir, "y_train.csv"),
        os.path.join(processed_dir, "y_test.csv"),
        os.path.join(processed_dir, "scaler.pkl"),
    ]
    preprocess_inputs = [raw_csv, params_file]
    preprocess_cache  = os.path.join(cache_dir, "preprocess.json")

    if force or stage_is_stale(preprocess_outputs, preprocess_inputs, preprocess_cache):
        run_stage("Preprocessing", "preprocess.py", preprocess_outputs)
        save_hashes(preprocess_inputs, preprocess_cache)
    else:
        print("[run.py] ⏭   Preprocess — up to date, skipping.")

    # ── Stage 2: Train ───────────────────────────────────────────────────
    train_outputs = [model_path, results_path]
    train_inputs  = preprocess_outputs + [params_file]
    train_cache   = os.path.join(cache_dir, "train.json")

    if force or stage_is_stale(train_outputs, train_inputs, train_cache):
        run_stage("Training", "train.py", train_outputs)
        save_hashes(train_inputs, train_cache)
    else:
        print("[run.py] ⏭   Training   — up to date, skipping.")

    # ── Stage 3: Evaluate ────────────────────────────────────────────────
    eval_outputs = [eval_path]
    eval_inputs  = [model_path] + preprocess_outputs
    eval_cache   = os.path.join(cache_dir, "evaluate.json")

    if force or stage_is_stale(eval_outputs, eval_inputs, eval_cache):
        run_stage("Evaluation", "evaluate.py", eval_outputs)
        save_hashes(eval_inputs, eval_cache)
    else:
        print("[run.py] ⏭   Evaluation — up to date, skipping.")

    print(f"\n[run.py] ✅  Pipeline complete. All artifacts are ready.\n")

# ── Launch Streamlit ──────────────────────────────────────────────────────────
def launch_streamlit():
    params    = load_params()
    port      = params.get("streamlit", {}).get("port", 8501)
    ui_script = os.path.join(SRC, "streamlit_app.py")

    print(f"[run.py] 🚀  Launching Streamlit on http://localhost:{port}")
    print(f"[run.py]     Press Ctrl+C to stop.\n")

    subprocess.run(
        ["streamlit", "run", ui_script,
         "--server.port", str(port),
         "--server.headless", "false"],
        cwd=ROOT,
    )

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Heart Disease MLOps entrypoint")
    parser.add_argument("--pipeline", action="store_true", help="Run pipeline only, no UI")
    parser.add_argument("--ui",       action="store_true", help="Launch UI only, skip pipeline")
    parser.add_argument("--force",    action="store_true", help="Force re-run all pipeline stages")
    args = parser.parse_args()

    if args.ui:
        # User explicitly wants just the UI
        launch_streamlit()
    elif args.pipeline:
        # User wants just the pipeline
        run_pipeline(force=args.force)
    else:
        # Default: run pipeline (smart skip) then launch UI
        run_pipeline(force=args.force)
        launch_streamlit()
