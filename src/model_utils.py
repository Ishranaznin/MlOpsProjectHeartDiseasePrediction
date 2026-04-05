"""
Shared model loading utility used by evaluate.py, app.py, and streamlit_app.py.
Handles the sklearn version bundle format and provides clear error messages.
"""

import os
import sys
import sklearn
import joblib
import warnings
from packaging.version import Version


def load_model_bundle(model_path: str):
    """
    Load a model saved by train.py.
    Returns (model, model_name) tuple.
    Raises RuntimeError with a clear message on version mismatch or missing file.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at '{model_path}'.\n"
            f"Run:  python src/preprocess.py && python src/train.py"
        )

    bundle = joblib.load(model_path)

    # Support both bundle dict format and raw model (backward compat)
    if isinstance(bundle, dict) and "model" in bundle:
        model        = bundle["model"]
        model_name   = bundle.get("model_name", type(model).__name__)
        saved_ver    = bundle.get("sklearn_version", "unknown")
        current_ver  = sklearn.__version__

        if saved_ver != "unknown" and saved_ver != current_ver:
            # Only warn (not crash) for minor version differences
            saved_parts   = saved_ver.split(".")
            current_parts = current_ver.split(".")

            if saved_parts[0] != current_parts[0]:
                # Major version mismatch — hard error
                raise RuntimeError(
                    f"sklearn version mismatch!\n"
                    f"  Model was trained with : sklearn {saved_ver}\n"
                    f"  Your environment has  : sklearn {current_ver}\n\n"
                    f"Fix: retrain the model in your environment:\n"
                    f"  python src/preprocess.py\n"
                    f"  python src/train.py"
                )
            else:
                # Minor mismatch — warn but continue
                warnings.warn(
                    f"sklearn version mismatch (minor): trained={saved_ver}, "
                    f"current={current_ver}. Retraining is recommended.",
                    UserWarning, stacklevel=2
                )
    else:
        # Raw model (old format or direct joblib.dump)
        model      = bundle
        model_name = type(model).__name__
        saved_ver  = "unknown"

    return model, model_name, saved_ver
