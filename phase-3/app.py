import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="GrowMar Phase 3",
    page_icon="🌱",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
 background:#f5f7fb;
 color:#1f2937;
}
.block-container {
 max-width:1100px;
 padding:2rem 1rem;
}
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stWidgetLabel"] p {
 color:#1f2937 !important;
}
h1,h2,h3,h4 {
 color:#172b4d !important;
}
.hero {
 background:linear-gradient(120deg,#102a43,#176b59);
 padding:30px;
 border-radius:18px;
 color:white;
}
.hero h1,.hero p {
 color:white !important;
}
[data-testid="stMetric"] {
 background:white;
 padding:15px;
 border-radius:12px;
 border:1px solid #e5e7eb;
}
[data-testid="stMetricLabel"] p {
 color:#475569 !important;
}
[data-testid="stMetricValue"] {
 color:#102a43 !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>GrowMar Workforce Intelligence</h1>
<p>Phase 3 | AI and Machine Learning Engineering</p>
<p>Zenith Nova Technologies, part of Aura Workforce</p>
</div>
""", unsafe_allow_html=True)

st.info("Public project demonstration. Individual employee and candidate records are not displayed.")

st.header("Project Overview")
st.write("""
Phase 3 implements five HR analytics and machine-learning workstreams:
resume ranking, fairness auditing, attrition prediction,
employee risk explanations and salary modelling.
""")

st.header("Project Results")

c1, c2, c3 = st.columns(3)
c1.metric("Resume fairness audit", "250", "Twin comparisons")
c2.metric("Attrition ROC-AUC", "0.6793")
c3.metric("Attrition recall", "57.14%")

st.caption("Metrics are reproduced from the Phase 3 evaluation reports. They are not guarantees of future performance.")

tasks = {
"Task 11 — Resume Ranking":
"TF-IDF similarity, skill overlap, ranking evaluation and manual misrank review.",
"Task 12 — Fairness Audit":
"Counterfactual twin-resume testing, masking comparison and group selection gaps.",
"Task 13 — Attrition Model V2":
"Gradient boosting, cross-validation, threshold selection and probability calibration.",
"Task 14 — Employee Risk Explanations":
"Attrition risk scoring, risk bands and SHAP-based explanations for human review.",
"Task 15 — Salary Modelling":
"Salary regression, model comparison and gender and career-gap fairness evaluation."
}

st.header("Phase 3 Workstreams")
for title, description in tasks.items():
    with st.container(border=True):
        st.subheader(title)
        st.write(description)

st.header("Evaluation Highlights")
st.write("""
- Task 12: The report records 250 exact zero-gap twin-resume comparisons.
- Task 13: Held-out ROC-AUC 0.6793 and PR-AUC 0.3289.
- Task 15: The selected log-linear model reduced test MAE by approximately
  34.3% compared with ordinary linear regression.
- Fairness reports identify limitations and areas requiring further review.
""")

st.header("Technical Architecture")
st.code("""
HR Data
   |
Data Validation and Feature Engineering
   |
ML Workflows (Tasks 11–15)
   |
Evaluation and Fairness Audits
   |
Reports and Human Review
   |
Responsive Project Dashboard
""")

st.header("Technology Stack")
st.write("""
Python | pandas | NumPy | scikit-learn | XGBoost |
SHAP | Matplotlib | Docker | Streamlit
""")

st.header("Project Reports")

report_dir = Path("reports")
report_files = sorted(report_dir.glob("*.md")) if report_dir.exists() else []

if report_files:
    for report in report_files:
        with st.expander(report.stem.replace("_", " ").title()):
            st.markdown(report.read_text(encoding="utf-8"))
else:
    st.warning("Project reports are not available in this deployment.")

st.header("Limitations and Responsible Use")
st.warning("""
This is an educational and engineering demonstration.
Model predictions are uncertain decision-support outputs.
They must not be used as the sole basis for hiring, promotion,
salary or termination decisions. Fairness results are limited
to the evaluation samples described in the reports.
""")

st.caption("GrowMar Phase 3 | Responsive web demonstration")

