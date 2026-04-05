"""
Streamlit UI for Heart Disease Risk Predictor.
Loads model directly (no API server needed).
All labels, options, defaults from params.yaml.

Run:
    streamlit run src/streamlit_app.py
"""

import sys
import os
import yaml
import json
import sklearn
import joblib
import warnings
import numpy as np
import pandas as pd
import streamlit as st

# ── Ensure src/ is on path so model_utils imports cleanly ────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from model_utils import load_model_bundle

# ── params.yaml path: works whether you run from project root or src/ ────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
PARAMS_PATH = os.path.join(_ROOT, "params.yaml")


@st.cache_data(show_spinner=False)
def load_params():
    with open(PARAMS_PATH) as f:
        return yaml.safe_load(f)


PARAMS      = load_params()
ST_CFG      = PARAMS["streamlit"]
FEAT_INFO   = ST_CFG["feature_info"]
CLASS_NAMES = PARAMS["data"]["class_names"]
MODEL_PATH  = os.path.join(_ROOT, PARAMS["evaluate"]["model_path"])
SCALER_PATH = os.path.join(_ROOT, PARAMS["data"]["processed_path"], "scaler.pkl")
RESULTS_PATH= os.path.join(_ROOT, PARAMS["training"]["results_output_path"])
EVAL_PATH   = os.path.join(_ROOT, PARAMS["evaluate"]["output_path"])

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=ST_CFG["title"],
    page_icon=ST_CFG["page_icon"],
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.result-high { background:#ffebee; border-left:5px solid #c62828;
               padding:16px 20px; border-radius:8px; }
.result-low  { background:#e8f5e9; border-left:5px solid #2e7d32;
               padding:16px 20px; border-radius:8px; }
.result-title{ font-size:1.3rem; font-weight:600; margin-bottom:4px; }
.result-sub  { font-size:0.88rem; color:#444; }
.metric-box  { background:#f5f5f5; border-radius:8px;
               padding:12px; text-align:center; }
.metric-num  { font-size:1.5rem; font-weight:600; color:#1a237e; }
.metric-lbl  { font-size:0.75rem; color:#666; }
</style>
""", unsafe_allow_html=True)


# ── Load model & scaler (cached across reruns) ────────────────────────────────
@st.cache_resource(show_spinner="Loading champion model…")
def get_model_and_scaler():
    model, model_name, saved_ver = load_model_bundle(MODEL_PATH)
    try:
        scaler = joblib.load(SCALER_PATH)
    except FileNotFoundError:
        scaler = None
    return model, model_name, saved_ver, scaler


@st.cache_data(show_spinner=False)
def get_results():
    out = {}
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            out["summary"] = json.load(f)
    if os.path.exists(EVAL_PATH):
        with open(EVAL_PATH) as f:
            out["evaluation"] = json.load(f)
    return out


# ── Header ────────────────────────────────────────────────────────────────────
st.title(f"{ST_CFG['page_icon']} {ST_CFG['title']}")
st.caption("Enter patient data in the sidebar → click **Predict** → see risk assessment.")
st.divider()

# ── Check model exists before anything else ───────────────────────────────────
if not os.path.exists(MODEL_PATH):
    st.error("### Model not found")
    st.markdown(f"""
The trained model was not found at:
```
{MODEL_PATH}
```
**Run the training pipeline first:**
```bash
python src/preprocess.py
python src/train.py
python src/evaluate.py
```
Then restart Streamlit.
""")
    st.stop()

# ── Load model — show version-mismatch guidance if it fails ──────────────────
try:
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        model, model_name, saved_ver, scaler = get_model_and_scaler()

    # Surface any version-mismatch warnings as Streamlit warnings
    for w in caught_warnings:
        if issubclass(w.category, UserWarning):
            st.warning(
                f"**sklearn version mismatch (minor):** "
                f"model trained with sklearn `{saved_ver}`, "
                f"current environment has sklearn `{sklearn.__version__}`. "
                f"Predictions may differ slightly. "
                f"Re-run `python src/train.py` to retrain in this environment."
            )

except RuntimeError as e:
    # Major version mismatch — clear actionable message
    st.error("### sklearn version mismatch")
    st.markdown(f"""
```
{e}
```

**How to fix:**  
Open a terminal in the project directory and retrain the model using **your** Python environment:
```bash
python src/preprocess.py
python src/train.py
python src/evaluate.py
```
Then restart Streamlit. The new model will be saved with sklearn `{sklearn.__version__}`.
""")
    st.stop()

except Exception as e:
    st.error(f"Failed to load model: `{e}`")
    st.stop()

# ── Sidebar — all widgets built from params.yaml feature_info ────────────────
st.sidebar.header("🩺 Patient Clinical Data")
st.sidebar.caption("All fields required · defaults are dataset averages")


def build_sidebar_inputs(feat_info: dict) -> dict:
    values = {}
    for feat, cfg in feat_info.items():
        label = cfg["label"]
        ftype = cfg["type"]

        if ftype == "select":
            raw_opts   = cfg["options"]
            int_keys   = [int(k) for k in raw_opts.keys()]
            str_labels = [str(raw_opts[k]) for k in raw_opts.keys()]
            default_v  = int(cfg["default"])
            default_ix = int_keys.index(default_v) if default_v in int_keys else 0
            chosen     = st.sidebar.selectbox(label, options=str_labels, index=default_ix, key=feat)
            values[feat] = int_keys[str_labels.index(chosen)]

        elif ftype == "float":
            values[feat] = st.sidebar.slider(
                label,
                min_value=float(cfg["min"]),
                max_value=float(cfg["max"]),
                value=float(cfg["default"]),
                step=float(cfg["step"]),
                key=feat,
            )
        else:  # int
            values[feat] = st.sidebar.slider(
                label,
                min_value=int(cfg["min"]),
                max_value=int(cfg["max"]),
                value=int(cfg["default"]),
                step=int(cfg["step"]),
                key=feat,
            )
    return values


input_vals  = build_sidebar_inputs(FEAT_INFO)
predict_btn = st.sidebar.button("🔍 Predict Risk", use_container_width=True, type="primary")
st.sidebar.divider()
st.sidebar.markdown(
    f"<small>Model: <b>{model_name}</b><br>"
    f"sklearn: <b>{saved_ver}</b><br>"
    f"Current sklearn: <b>{sklearn.__version__}</b></small>",
    unsafe_allow_html=True,
)

# ── Main layout ───────────────────────────────────────────────────────────────
left_col, right_col = st.columns([3, 2], gap="large")

with left_col:
    st.subheader("🎯 Prediction Result")
    result_slot = st.empty()

    if not predict_btn:
        result_slot.info("👈 Fill in the patient data in the sidebar, then click **Predict Risk**.")
    else:
        # Build feature vector in exact training column order
        feat_names = list(FEAT_INFO.keys())
        row_vals   = [input_vals[f] for f in feat_names]
        row_df     = pd.DataFrame([row_vals], columns=feat_names)
        X          = scaler.transform(row_df) if scaler is not None else row_df.values

        pred   = int(model.predict(X)[0])
        proba  = model.predict_proba(X)[0] if hasattr(model, "predict_proba") else [0.5, 0.5]
        p_dis  = round(float(proba[1]) * 100, 1)
        p_none = round(float(proba[0]) * 100, 1)
        label  = CLASS_NAMES[pred]

        if pred == 1:
            result_slot.markdown(
                f'<div class="result-high">'
                f'<div class="result-title">🔴 {label}</div>'
                f'<div class="result-sub">Elevated risk detected. '
                f'Further clinical evaluation is recommended.</div>'
                f'</div>', unsafe_allow_html=True
            )
        else:
            result_slot.markdown(
                f'<div class="result-low">'
                f'<div class="result-title">🟢 {label}</div>'
                f'<div class="result-sub">Low risk detected. '
                f'Continue routine monitoring.</div>'
                f'</div>', unsafe_allow_html=True
            )

        st.markdown("#### Probability Breakdown")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("❤️ Disease Risk",       f"{p_dis}%")
            st.progress(p_dis / 100)
        with c2:
            st.metric("✅ No Disease",         f"{p_none}%")
            st.progress(p_none / 100)

        st.caption(f"Model: **{model_name}** · sklearn {saved_ver}")

    # Input summary table
    st.markdown("#### 📋 Input Summary")
    summary_rows = []
    for feat, cfg in FEAT_INFO.items():
        val = input_vals[feat]
        if cfg["type"] == "select":
            int_keys  = [int(k) for k in cfg["options"].keys()]
            str_vals  = [str(cfg["options"][k]) for k in cfg["options"].keys()]
            display   = str_vals[int_keys.index(val)] if val in int_keys else val
        else:
            display = val
        summary_rows.append({"Feature": cfg["label"], "Value": display})

    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)


with right_col:
    st.subheader("📊 Model Performance")
    results = get_results()

    if "summary" in results:
        s = results["summary"]
        st.success(f"🏆 Champion: **{s['champion']}**  "
                   f"(selected by `{s.get('champion_metric','f1_score')}` from params.yaml)")

        if "all_results" in s:
            rows = []
            for name, info in s["all_results"].items():
                m = info["metrics"]
                rows.append({
                    "Model":    name,
                    "Accuracy": f"{m.get('accuracy',0)*100:.1f}%",
                    "F1":       f"{m.get('f1_score',0):.3f}",
                    "ROC-AUC":  f"{m.get('roc_auc',0):.3f}",
                    "🏆":       "✓" if name == s["champion"] else "",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if "evaluation" in results:
        e = results["evaluation"]
        st.markdown("#### Test-Set Metrics")
        m1, m2, m3 = st.columns(3)
        m1.metric("Accuracy", f"{e['accuracy']*100:.1f}%")
        m2.metric("F1 Score", f"{e['f1_score']:.3f}")
        m3.metric("ROC-AUC",  f"{e.get('roc_auc',0):.3f}")

        st.markdown("#### Confusion Matrix")
        cm     = e["confusion_matrix"]
        cm_df  = pd.DataFrame(
            cm,
            index  =[f"Actual: {CLASS_NAMES[i]}" for i in range(len(cm))],
            columns=[f"Pred: {CLASS_NAMES[j]}"   for j in range(len(cm[0]))],
        )
        st.dataframe(cm_df, use_container_width=True)

        gate  = e["pass_threshold"]
        f1val = e["f1_score"]
        if e["passed"]:
            st.success(f"✅ CI/CD Gate PASSED — F1={f1val:.3f} ≥ threshold={gate} (params.yaml)")
        else:
            st.error(f"❌ CI/CD Gate FAILED — F1={f1val:.3f} < threshold={gate} (params.yaml)")
    else:
        st.info("Run `python src/evaluate.py` to populate this panel.")

    # Feature reference expander
    with st.expander("📖 Feature Reference"):
        for feat, cfg in FEAT_INFO.items():
            if cfg["type"] == "select":
                opts = ", ".join(f"{k}={v}" for k, v in cfg["options"].items())
                st.markdown(f"**{cfg['label']}** — {opts}")
            else:
                st.markdown(
                    f"**{cfg['label']}** — "
                    f"range [{cfg['min']}, {cfg['max']}], default {cfg['default']}"
                )
