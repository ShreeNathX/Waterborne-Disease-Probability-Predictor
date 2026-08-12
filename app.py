"""
Streamlit app that loads model_artifacts.joblib and lets you input a water sample
and see:
 - model predicted probabilities per disease
 - which heuristic rules fired and their empirical confidences
 - a combined probability (model vs rule) via a simple weighting
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Waterborne Disease Predictor", layout="wide")

# Load artifacts
@st.cache_resource
def load_artifacts(path="model_artifacts.joblib"):
    artifacts = joblib.load(path)
    return artifacts

try:
    artifacts = load_artifacts("model_artifacts.joblib")
except Exception as e:
    st.error(f"Could not load model artifacts: {e}")
    st.stop()

clf = artifacts["model"]
le = artifacts["label_encoder"]
FEATURES = artifacts["features"]
defaults = artifacts["defaults"]
ranges = artifacts["ranges"]
rule_confidences = artifacts["rule_confidences"]
feature_importances = artifacts["feature_importances"]
class_names = list(le.classes_)

st.title("Waterborne Disease Probability Predictor")
st.write("Enter water sample values (lab / sensor) below and press **Predict**. The app shows model predictions (probabilities), which heuristic rules fired, and combined probabilities.")

# System/browser dependent theme.
# Streamlit's own theme can vary by installation, so these CSS rules explicitly
# follow the browser/OS preference through prefers-color-scheme.
st.markdown("""
<style>
    /* Keep the main content centered instead of using the sidebar for inputs. */
    .block-container {
        max-width: 1180px;
        margin: 0 auto;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    @media (prefers-color-scheme: dark) {
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stHeader"] {
            background-color: #0e1117 !important;
            color: #f5f7fa !important;
        }

        [data-testid="stSidebar"] {
            background-color: #11151c !important;
        }

        [data-testid="stMarkdownContainer"],
        [data-testid="stText"],
        label,
        h1, h2, h3, h4, h5, h6,
        p, span, div {
            color: #f5f7fa;
        }

        input, textarea, [data-baseweb="select"] > div {
            background-color: #1b212b !important;
            color: #f5f7fa !important;
        }

        input::placeholder {
            color: #aeb7c4 !important;
        }

        [data-testid="stDataFrame"],
        [data-testid="stTable"] {
            color: #f5f7fa !important;
        }
    }

    @media (prefers-color-scheme: light) {
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stHeader"] {
            background-color: #ffffff !important;
            color: #172033 !important;
        }

        [data-testid="stSidebar"] {
            background-color: #f7f8fa !important;
        }

        [data-testid="stMarkdownContainer"],
        [data-testid="stText"],
        label,
        h1, h2, h3, h4, h5, h6,
        p, span, div {
            color: #172033;
        }

        input, textarea, [data-baseweb="select"] > div {
            background-color: #ffffff !important;
            color: #172033 !important;
        }

        input::placeholder {
            color: #6b7280 !important;
        }
    }

    /* Center the input section and keep it readable on wide screens. */
    .input-section {
        max-width: 1050px;
        margin: 0 auto;
    }

    .input-title {
        text-align: center;
        margin-bottom: 0.5rem;
    }

    .input-help {
        text-align: center;
        opacity: 0.75;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="input-section">', unsafe_allow_html=True)
st.markdown('<h2 class="input-title">Water Sample Inputs</h2>', unsafe_allow_html=True)
st.markdown(
    '<p class="input-help">Enter the laboratory or sensor values for the water sample.</p>',
    unsafe_allow_html=True
)

input_vals = {}

# Use centered columns rather than the left sidebar.
input_columns = st.columns(3, gap="large")

for i, f in enumerate(FEATURES):
    r = ranges.get(f, {"min": 0.0, "max": 100.0})
    default = defaults.get(f, (r["min"] + r["max"]) / 2.0)
    default_val = float(default)

    step = max((r["max"] - r["min"]) / 100.0, 0.01)

    with input_columns[i % 3]:
        input_vals[f] = st.number_input(
            f,
            min_value=float(r["min"]),
            max_value=float(r["max"]),
            value=default_val,
            step=step,
            format="%.3f",
            key=f"input_{f}"
        )

alpha = st.slider(
    "Rule weight (alpha)",
    0.0, 1.0, 0.4, 0.05,
    help=(
        "Controls how strongly a fired heuristic rule influences the model "
        "probability. If a rule does not fire, the model probability is kept "
        "unchanged."
    )
)

st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# Heuristic rule implementations for single sample (mirrors training rules)
def cholera_rule_sample(s):
    return (s["E_coli"] >= 100) and (s["Total_Coliforms"] >= 500) and (s["Residual_Chlorine"] < 0.2) and (s["Turbidity"] >= 5)

def diarrhea_rule_sample(s):
    conds = [s["E_coli"] >= 50, s["Turbidity"] >= 3, s["Residual_Chlorine"] < 0.5]
    return sum(1 for c in conds if c) >= 2

def typhoid_rule_sample(s):
    return (s["Total_Coliforms"] >= 200) and (s["Enterococci"] >= 10) and (s["Nitrate"] >= 10)

def dysentery_rule_sample(s):
    return (s["E_coli"] >= 100) and (s["Enterococci"] >= 20) and (s["Residual_Chlorine"] < 0.3)

def hepatitis_rule_sample(s):
    return (s["Turbidity"] >= 10) and (s["Residual_Chlorine"] < 0.5)

SAMPLE_RULES = [
    ("Cholera_rule", "Cholera", cholera_rule_sample),
    ("Diarrhea_rule", "Diarrhea", diarrhea_rule_sample),
    ("Typhoid_rule", "Typhoid", typhoid_rule_sample),
    ("Dysentery_rule", "Dysentery", dysentery_rule_sample),
    ("HepatitisA_rule", "HepatitisA", hepatitis_rule_sample)
]

# Predict button
if st.button("Predict"):
    sample_df = pd.DataFrame([input_vals])[FEATURES]
    # Model probabilities
    probs = clf.predict_proba(sample_df)[0]  # array of len n_classes
    probs_dict = dict(zip(le.inverse_transform(np.arange(len(probs))), probs))  # mapping label->prob

    # Rule checks
    rule_matches = []
    per_disease_rule_conf = {c: 0.0 for c in class_names}  # default 0.0
    for rule_name, disease_label, rule_fn in SAMPLE_RULES:
        match = rule_fn(input_vals)
        info = rule_confidences.get(rule_name, {"confidence": None, "matches": 0})
        conf = info["confidence"] if info["confidence"] is not None else 0.0
        rule_matches.append({
            "rule": rule_name,
            "disease": disease_label,
            "match": bool(match),
            "empirical_confidence": conf,
            "n_matches_in_data": info.get("matches", 0)
        })
        # if rule matches, assign to disease the empirical confidence (choose max if multiple rules for same disease)
        if match and conf is not None:
            per_disease_rule_conf[disease_label] = max(per_disease_rule_conf.get(disease_label, 0.0), conf)

    # Combine model probabilities with heuristic evidence.
    #
    # IMPORTANT:
    # A rule that does NOT fire is not evidence of "0% probability".
    # Previously, non-fired rules were represented as 0.0 and blended directly
    # with the model probability, which could incorrectly reduce a strong model
    # prediction. We now blend a rule confidence only when that rule actually
    # fires; otherwise the model probability is retained unchanged.
    combined = {}
    for disease in class_names:
        model_p = float(probs_dict.get(disease, 0.0))

        matched_rule = next(
            (r for r in rule_matches if r["disease"] == disease and r["match"]),
            None
        )

        if matched_rule is not None:
            rule_p = float(matched_rule["empirical_confidence"])
            combined_p = (1.0 - alpha) * model_p + alpha * rule_p
            rule_status = "Fired"
        else:
            rule_p = None
            combined_p = model_p
            rule_status = "Not fired"

        combined[disease] = {
            "model_prob": model_p,
            "rule_prob": rule_p,
            "combined_prob": float(np.clip(combined_p, 0.0, 1.0)),
            "rule_status": rule_status
        }

    # Present results in two columns
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Predicted probabilities (model only)")
        df_model = pd.DataFrame([
            {"disease": d, "model_prob (%)": round(v["model_prob"]*100, 1)}
            for d, v in combined.items()
        ]).sort_values("model_prob (%)", ascending=False)
        st.table(df_model)

        st.subheader("Combined probabilities (model + rule, alpha={:.2f})".format(alpha))
        df_comb = pd.DataFrame([
            {
                "disease": d,
                "combined_prob (%)": round(v["combined_prob"] * 100, 1),
                "rule_status": v["rule_status"]
            }
            for d, v in combined.items()
        ]).sort_values("combined_prob (%)", ascending=False)
        st.table(df_comb)

        # Show bar chart for combined probabilities
        st.subheader("Combined probability visualization")
        viz_df = pd.DataFrame({k: v["combined_prob"] for k, v in combined.items()}, index=["prob"]).T
        st.bar_chart(viz_df["prob"])

        # Allow download of output
        out_df = pd.DataFrame([input_vals])
        for d, v in combined.items():
            out_df[f"model_prob_{d}"] = v["model_prob"]
            out_df[f"rule_conf_{d}"] = v["rule_prob"]
            out_df[f"rule_status_{d}"] = v["rule_status"]
            out_df[f"combined_prob_{d}"] = v["combined_prob"]

        csv = out_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download prediction CSV", csv, "prediction_result.csv", "text/csv")

    with col2:
        st.subheader("Rule matches")
        rm_df = pd.DataFrame(rule_matches)
        # show only the rules that matched first
        matched = rm_df[rm_df["match"] == True]
        if not matched.empty:
            st.write("Rules that fired (empirical confidence shown):")
            tmp = matched[["rule", "disease", "empirical_confidence", "n_matches_in_data"]].copy()
            tmp["empirical_confidence (%)"] = (tmp["empirical_confidence"]*100).round(1)
            st.table(tmp[["rule", "disease", "empirical_confidence (%)", "n_matches_in_data"]])
        else:
            st.info("No heuristic rules fired for this sample. Model-only prediction shown.")

        st.subheader("Heuristic rule confidences (from training data)")
        rc_df = pd.DataFrame([
            {"rule": k, "disease": v["disease"], "matches_in_dataset": v["matches"], "empirical_confidence (%)": (v["confidence"]*100 if v["confidence"] is not None else None)}
            for k, v in rule_confidences.items()
        ])
        st.table(rc_df)

        st.subheader("Top feature importances")
        fi_df = pd.DataFrame([
            {"feature": k, "importance": v} for k, v in feature_importances.items()
        ]).sort_values("importance", ascending=False)
        st.table(fi_df)

st.write("---")
st.write(
    "Notes: model probabilities come from the trained RandomForest. "
    "A heuristic rule is blended with the model only when that rule fires; "
    "a non-fired rule is treated as no additional evidence rather than 0% probability. "
    "Rule confidences are empirical precisions computed from the training data "
    "and should be calibrated further on real outbreak labels."
)