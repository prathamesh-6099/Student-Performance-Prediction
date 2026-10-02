"""
app/app.py
----------
Streamlit Interactive Dashboard for Explainable Student Performance Prediction.

Architecture:
1. ML Model (Random Forest / XGBoost / Ridge) predicts student final exam score.
2. SHAP (TreeExplainer) identifies exact positive & negative feature drivers.
3. Constrained LLM Layer (Gemini API with robust template fallback) translates drivers
   into plain language for Students or Teachers without hallucination.
4. Interactive Counterfactual What-If Sandbox with real-time score updates.
"""

import sys
from pathlib import Path

# Add project root and src to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "src"))

import json
from typing import Dict, Any, Tuple
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap

from src.explain import (
    generate_explanation_payload,
    get_shap_explainer,
    ACTIONABLE_FEATURES,
    NON_ACTIONABLE_FEATURES
)
from src.llm_layer import explain_in_words, template_fallback_explanation
from src.labels import convert_score_to_band, DEFAULT_FAIL_THRESHOLD, DEFAULT_PASS_THRESHOLD

# Page configuration
st.set_page_config(
    page_title="Explainable Student Performance AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Rich Modern Aesthetics)
st.markdown("""
<style>
    /* Global Typography & Palette */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 24px;
        border-radius: 12px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
    }
    
    .badge-pass {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    
    .badge-borderline {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeeba;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    
    .badge-fail {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    
    .card-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        margin-bottom: 18px;
    }
    
    .metric-value {
        font-size: 2.8rem;
        font-weight: 800;
        color: #1e293b;
        line-height: 1.1;
    }
    
    .metric-subtitle {
        font-size: 0.9rem;
        color: #64748b;
        margin-top: 4px;
    }
    
    .footer-text {
        text-align: center;
        color: #94a3b8;
        font-size: 0.85rem;
        padding: 20px 0;
        border-top: 1px solid #e2e8f0;
        margin-top: 40px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_project_artifacts():
    """
    Loads and caches model, preprocessor, and SHAP explainer artifacts once.
    """
    models_dir = PROJECT_ROOT / "models"
    model = joblib.load(models_dir / "best_model.joblib")
    preprocessor = joblib.load(models_dir / "preprocessor.joblib")
    feature_names = joblib.load(models_dir / "feature_names.joblib")
    explainer = get_shap_explainer(model)
    return model, preprocessor, feature_names, explainer


try:
    model, preprocessor, feature_names, explainer = load_project_artifacts()
except Exception as e:
    st.error(f"Failed to load model artifacts from 'models/'. Please run 'python3 src/train.py' first.\nError: {e}")
    st.stop()


# ---------------------------------------------------------
# Sidebar: Input Widgets for all 19 features
# ---------------------------------------------------------
st.sidebar.title("Student Profile Inputs")
st.sidebar.markdown("Configure student characteristics across academic, environmental, and behavioral dimensions.")

# Preset selector for fast demonstration
preset = st.sidebar.selectbox(
    "Load Example Profile:",
    ["Custom Input", "At-Risk Student (Low Attendance)", "Borderline Student", "High Achiever"]
)

default_vals = {
    "Study_Hours_per_Week": 24,
    "Attendance_Rate": 78.0,
    "Past_Exam_Scores": 72,
    "Sleep_Hours": 7,
    "Tutoring_Sessions": 1,
    "Physical_Activity": 3,
    "Gender": "Female",
    "Parental_Education_Level": "Bachelors",
    "Internet_Access_at_Home": "Yes",
    "Extracurricular_Activities": "Yes",
    "Parental_Involvement": "Medium",
    "Access_to_Resources": "Medium",
    "Motivation_Level": "Medium",
    "Family_Income": "Medium",
    "Teacher_Quality": "Medium",
    "School_Type": "Public",
    "Peer_Influence": "Neutral",
    "Learning_Disabilities": "No",
    "Distance_from_Home": "Near"
}

if preset == "At-Risk Student (Low Attendance)":
    default_vals.update({
        "Study_Hours_per_Week": 14,
        "Attendance_Rate": 54.0,
        "Past_Exam_Scores": 60,
        "Extracurricular_Activities": "No",
        "Parental_Involvement": "Low",
        "Motivation_Level": "Low"
    })
elif preset == "Borderline Student":
    default_vals.update({
        "Study_Hours_per_Week": 22,
        "Attendance_Rate": 72.0,
        "Past_Exam_Scores": 68,
        "Extracurricular_Activities": "No"
    })
elif preset == "High Achiever":
    default_vals.update({
        "Study_Hours_per_Week": 36,
        "Attendance_Rate": 98.0,
        "Past_Exam_Scores": 92,
        "Extracurricular_Activities": "Yes",
        "Parental_Involvement": "High",
        "Motivation_Level": "High"
    })

# Section 1: Core Academic Predictors
with st.sidebar.expander("📚 Academic & Study Habits", expanded=True):
    study_hours = st.slider(
        "Study Hours / Week",
        min_value=5, max_value=40,
        value=int(default_vals["Study_Hours_per_Week"]),
        help="Weekly self-study and homework hours."
    )
    attendance = st.slider(
        "Attendance Rate (%)",
        min_value=50.0, max_value=100.0,
        value=float(default_vals["Attendance_Rate"]),
        step=0.5,
        help="Percentage of total classes attended."
    )
    past_scores = st.slider(
        "Past Exam Score (0-100)",
        min_value=50, max_value=100,
        value=int(default_vals["Past_Exam_Scores"]),
        help="Average prior exam performance."
    )
    tutoring = st.slider(
        "Tutoring Sessions / Month",
        min_value=0, max_value=10,
        value=int(default_vals["Tutoring_Sessions"]),
        help="Number of external remedial or tutoring sessions."
    )

# Section 2: Environment & Family Background
with st.sidebar.expander("🏠 Background & Environment", expanded=False):
    parent_edu = st.selectbox(
        "Parental Education Level",
        ["High School", "College", "Bachelors", "Masters", "PhD"],
        index=["High School", "College", "Bachelors", "Masters", "PhD"].index(default_vals["Parental_Education_Level"])
    )
    internet = st.selectbox(
        "Internet Access at Home",
        ["Yes", "No"],
        index=["Yes", "No"].index(default_vals["Internet_Access_at_Home"])
    )
    extracurricular = st.selectbox(
        "Extracurricular Activities",
        ["Yes", "No"],
        index=["Yes", "No"].index(default_vals["Extracurricular_Activities"])
    )
    parent_involve = st.selectbox(
        "Parental Involvement",
        ["Low", "Medium", "High"],
        index=["Low", "Medium", "High"].index(default_vals["Parental_Involvement"])
    )
    resources = st.selectbox(
        "Access to Resources",
        ["Low", "Medium", "High"],
        index=["Low", "Medium", "High"].index(default_vals["Access_to_Resources"])
    )
    income = st.selectbox(
        "Family Income Level",
        ["Low", "Medium", "High"],
        index=["Low", "Medium", "High"].index(default_vals["Family_Income"])
    )
    school_type = st.selectbox(
        "School Type",
        ["Public", "Private"],
        index=["Public", "Private"].index(default_vals["School_Type"])
    )
    distance = st.selectbox(
        "Distance from Home",
        ["Near", "Moderate", "Far"],
        index=["Near", "Moderate", "Far"].index(default_vals["Distance_from_Home"])
    )

# Section 3: Personal & Behavioral
with st.sidebar.expander("👤 Personal & Wellness", expanded=False):
    gender = st.selectbox(
        "Gender",
        ["Female", "Male"],
        index=["Female", "Male"].index(default_vals["Gender"])
    )
    sleep = st.slider(
        "Daily Sleep Hours",
        min_value=4, max_value=12,
        value=int(default_vals["Sleep_Hours"])
    )
    physical = st.slider(
        "Physical Activity (Days/Week)",
        min_value=0, max_value=7,
        value=int(default_vals["Physical_Activity"])
    )
    motivation = st.selectbox(
        "Motivation Level",
        ["Low", "Medium", "High"],
        index=["Low", "Medium", "High"].index(default_vals["Motivation_Level"])
    )
    teacher_q = st.selectbox(
        "Teacher Quality",
        ["Low", "Medium", "High"],
        index=["Low", "Medium", "High"].index(default_vals["Teacher_Quality"])
    )
    peer_inf = st.selectbox(
        "Peer Influence",
        ["Negative", "Neutral", "Positive"],
        index=["Negative", "Neutral", "Positive"].index(default_vals["Peer_Influence"])
    )
    disability = st.selectbox(
        "Learning Disabilities",
        ["No", "Yes"],
        index=["No", "Yes"].index(default_vals["Learning_Disabilities"])
    )

# Compile current student dictionary
student_input = {
    "Study_Hours_per_Week": study_hours,
    "Hours_Studied": study_hours,
    "Attendance_Rate": attendance,
    "Attendance": attendance,
    "Past_Exam_Scores": past_scores,
    "Previous_Scores": past_scores,
    "Tutoring_Sessions": tutoring,
    "Sleep_Hours": sleep,
    "Physical_Activity": physical,
    "Gender": gender,
    "Parental_Education_Level": parent_edu,
    "Internet_Access_at_Home": internet,
    "Internet_Access": internet,
    "Extracurricular_Activities": extracurricular,
    "Parental_Involvement": parent_involve,
    "Access_to_Resources": resources,
    "Motivation_Level": motivation,
    "Family_Income": income,
    "Teacher_Quality": teacher_q,
    "School_Type": school_type,
    "Peer_Influence": peer_inf,
    "Learning_Disabilities": disability,
    "Distance_from_Home": distance
}

predict_button = st.sidebar.button("⚡ Predict & Explain", type="primary", use_container_width=True)

# ---------------------------------------------------------
# Main Page Header
# ---------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1 style="margin: 0; font-size: 2rem; font-weight: 700;">🎓 Explainable Student Performance Prediction</h1>
    <p style="margin: 8px 0 0 0; opacity: 0.9; font-size: 1.05rem;">
        Glass-box ML prediction powered by <b>Random Forest</b>, local <b>SHAP</b> feature attributions, and a grounded <b>Gemini LLM</b> explanation layer.
    </p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Prediction & Explanation Logic
# ---------------------------------------------------------
if "last_payload" not in st.session_state:
    st.session_state.last_payload = None

if predict_button or st.session_state.last_payload is not None:
    if predict_button or st.session_state.last_payload is None:
        with st.spinner("Calculating ML prediction and computing local SHAP values..."):
            payload = generate_explanation_payload(
                student_dict=student_input,
                model=model,
                preprocessor=preprocessor,
                feature_names=feature_names,
                explainer=explainer
            )
            st.session_state.last_payload = payload
            st.session_state.student_input = student_input
    else:
        payload = st.session_state.last_payload

    score = payload["predicted_score"]
    band = payload["band"]

    # 1. Prediction KPI Card & Performance Badge
    col1, col2, col3 = st.columns([1.5, 2, 2.5])

    with col1:
        st.markdown('<div class="card-box">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{score:.1f}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-subtitle">Predicted Exam Score (out of 100)</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card-box">', unsafe_allow_html=True)
        if band == "Pass":
            badge_html = '<span class="badge-pass">✅ PASS (≥ 60)</span>'
            desc = "Demonstrates strong foundational readiness."
        elif band == "Borderline":
            badge_html = '<span class="badge-borderline">⚠️ BORDERLINE (50–59)</span>'
            desc = "Near the passing margin; high coaching potential."
        else:
            badge_html = '<span class="badge-fail">🚨 AT RISK (&lt; 50)</span>'
            desc = "Immediate academic intervention advised."

        st.markdown(f'<div style="margin-top: 6px;">{badge_html}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-subtitle" style="margin-top: 14px;">{desc}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="card-box">', unsafe_allow_html=True)
        st.markdown("<b>Audience Explanation Perspective</b>", unsafe_allow_html=True)
        audience_choice = st.radio(
            "Tailor natural language tone for:",
            ["Student", "Teacher"],
            horizontal=True,
            index=0,
            key="audience_toggle"
        )
        st.markdown(f'<div class="metric-subtitle">Selected mode: <b>{audience_choice} Perspective</b></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # 2. LLM Explanation Layer in a Card
    st.markdown('<div class="card-box">', unsafe_allow_html=True)
    st.subheader(f"💬 Natural Language Advisor Briefing ({audience_choice} View)")

    with st.spinner("Generating plain-English explanation via LLM layer..."):
        try:
            explanation_text = explain_in_words(payload, audience=audience_choice.lower())
        except Exception as err:
            st.warning("Notice: Gemini API unavailable. Displaying deterministic template fallback.")
            explanation_text = template_fallback_explanation(payload, audience=audience_choice.lower())

    st.markdown(explanation_text)
    st.markdown('</div>', unsafe_allow_html=True)

    # 3. SHAP Feature Contributions Waterfall Chart
    st.markdown('<div class="card-box">', unsafe_allow_html=True)
    st.subheader("🔍 SHAP Local Attribution (Why this Score?)")
    st.markdown("Each bar indicates how much a specific factor nudged the prediction above (green) or below (red) the baseline average.")

    # Re-transform single student to pass directly into shap waterfall
    proc_single = pd.DataFrame(
        preprocessor.transform(pd.DataFrame([st.session_state.student_input])),
        columns=feature_names
    )
    shap_single = explainer(proc_single)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    shap.plots.waterfall(shap_single[0], max_display=7, show=False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.markdown('</div>', unsafe_allow_html=True)

    # 4. Interactive Live What-If Sandbox
    st.markdown('<div class="card-box">', unsafe_allow_html=True)
    st.subheader("🎯 Interactive What-If Simulator")
    st.markdown("Adjust key actionable levers to see real-time score improvements calculated live by the model.")

    wi_col1, wi_col2 = st.columns(2)
    with wi_col1:
        wi_att = st.slider(
            "Simulated Attendance Rate (%)",
            min_value=50.0, max_value=100.0,
            value=float(st.session_state.student_input["Attendance_Rate"]),
            step=1.0,
            key="wi_att_slider"
        )
        wi_study = st.slider(
            "Simulated Study Hours / Week",
            min_value=5, max_value=40,
            value=int(st.session_state.student_input["Study_Hours_per_Week"]),
            step=1,
            key="wi_study_slider"
        )
    with wi_col2:
        wi_extra = st.selectbox(
            "Simulated Extracurriculars",
            ["No", "Yes"],
            index=["No", "Yes"].index(st.session_state.student_input["Extracurricular_Activities"]),
            key="wi_extra_select"
        )
        wi_tutoring = st.slider(
            "Simulated Tutoring Sessions / Month",
            min_value=0, max_value=10,
            value=int(st.session_state.student_input["Tutoring_Sessions"]),
            key="wi_tutor_slider"
        )

    # Live what-if inference
    simulated_student = st.session_state.student_input.copy()
    simulated_student["Attendance_Rate"] = wi_att
    simulated_student["Attendance"] = wi_att
    simulated_student["Study_Hours_per_Week"] = wi_study
    simulated_student["Hours_Studied"] = wi_study
    simulated_student["Extracurricular_Activities"] = wi_extra
    simulated_student["Tutoring_Sessions"] = wi_tutoring

    sim_proc = pd.DataFrame(preprocessor.transform(pd.DataFrame([simulated_student])), columns=feature_names)
    sim_score = float(model.predict(sim_proc)[0])
    score_delta = sim_score - score
    sim_band = convert_score_to_band(sim_score)

    delta_color = "normal" if abs(score_delta) < 0.1 else ("inverse" if score_delta < 0 else "off")
    st.metric(
        label="Simulated Score",
        value=f"{sim_score:.1f} / 100 ({sim_band})",
        delta=f"{score_delta:+.2f} points from baseline"
    )

    if score < 60 and sim_score >= 60:
        st.success("🎉 These adjustments successfully elevate the student into the **PASS** band!")

    st.markdown('</div>', unsafe_allow_html=True)

    # Expandable Raw JSON inspection for technical grading
    with st.expander("🛠️ View Raw JSON Payload (XAI Hand-off to LLM)"):
        st.json(payload)

else:
    st.info("👈 Select student attributes in the left sidebar and click **'Predict & Explain'** to generate predictions.")


# ---------------------------------------------------------
# Footer: Ethical Disclaimer
# ---------------------------------------------------------
st.markdown("""
<div class="footer-text">
    <b>Pedagogical & Ethical Disclaimer</b>: All predictions, feature attributions, and natural language recommendations 
    are automated statistical estimates generated from historical student cohort data. They are designed to assist educators 
    and advisors in identifying positive behavioral interventions, not as definitive or punitive judgments.
</div>
""", unsafe_allow_html=True)
