import os
import uuid
import requests
import streamlit as st
import pandas as pd


st.set_page_config(
    page_title="GrowMar | HR Intelligence",
    page_icon="📊",
    layout="wide",
)

API_KEY = os.getenv("GROWMAR_API_KEY", "").strip()
if not API_KEY:
    st.error("Server configuration error: GROWMAR_API_KEY is not configured.")
    st.stop()

st.markdown("""
<style>
    .block-container {padding-top: 1.8rem; padding-bottom: 2rem;}
    .hero {
        padding: 1.4rem 1.6rem;
        border-radius: 14px;
        background: linear-gradient(110deg, #142b4a, #245b73);
        color: white;
        margin-bottom: 1.2rem;
    }
    .hero h1 {color: white; margin: 0;}
    .hero p {color: #d8e8f2; margin: .4rem 0 0 0;}
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.25);
        padding: 12px; border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>GrowMar HR Intelligence</h1>
  <p>Workforce analytics · Predictive insights · Decision support</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("Connection")
    api_base = st.text_input(
        "API Base URL",
        value=os.getenv("API_BASE_URL", "http://localhost:8000"),
        help="For local testing, use http://localhost:8000",
    ).rstrip("/")
    st.caption("Connect to your running GrowMar Phase 4 API.")
    if st.button("Check API health", use_container_width=True):
        try:
            r = requests.get(f"{api_base}/v1/health", timeout=10)
            r.raise_for_status()
            st.success("API is reachable")
            st.json(r.json())
        except requests.RequestException as e:
            st.error(f"API connection failed: {e}")

    st.divider()
    st.caption("AI outputs are decision-support signals. Human review is required.")

def api_get(path, params=None):
    response = requests.get(
        f"{api_base}{path}", params=params,
        headers={"X-API-Key": API_KEY}, timeout=30
    )
    response.raise_for_status()
    return response.json()

def api_post(path, payload, headers=None):
    response = requests.post(
        f"{api_base}{path}",
        json=payload,
        headers={**(headers or {}), "X-API-Key": API_KEY},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()

def show_api_error(error):
    if isinstance(error, requests.HTTPError):
        try:
            st.error(f"API error ({error.response.status_code}): {error.response.text}")
        except Exception:
            st.error(str(error))
    else:
        st.error(f"Request failed: {error}")

tabs = st.tabs([
    "Attrition",
    "Attendance",
    "Payroll",
    "Salary",
    "Resume Screening",
    "Workforce Forecast",
])

# ---------------------------------------------------------
# 1. ATTRITION
# ---------------------------------------------------------
with tabs[0]:
    st.subheader("Employee Attrition Risk")
    st.write("Enter a synthetic or authorised employee feature profile.")

    fields = [
        ("department_id", "D01", "str"),
        ("role_id", "R002", "str"),
        ("job_level", 2, "int"),
        ("work_location", "Chennai", "str"),
        ("work_mode", "Hybrid", "str"),
        ("employment_type", "Full-time", "str"),
        ("highest_education", "B.Tech", "str"),
        ("tenure_months", 24, "int"),
        ("total_experience_years", 3.0, "float"),
        ("previous_employers_count", 1, "int"),
        ("commute_distance_km", 8.0, "float"),
        ("annual_fixed_pay", 600000, "int"),
        ("comp_ratio", 1.0, "float"),
        ("raises_received", 1, "int"),
        ("last_raise_pct", 8.0, "float"),
        ("months_since_last_raise", 12, "int"),
        ("promotions_received", 0, "int"),
        ("months_since_last_promotion", 24.0, "float"),
        ("internal_transfers", 0, "int"),
        ("last_performance_rating", 3.5, "float"),
        ("last_engagement_score", 3.0, "float"),
        ("manager_changes_12m", 0, "int"),
        ("training_hours_12m", 20.0, "float"),
        ("attendance_rate_3m", 0.95, "float"),
        ("late_arrivals_3m", 2, "int"),
        ("avg_overtime_hours_per_month_3m", 5.0, "float"),
        ("avg_hours_worked_3m", 160.0, "float"),
        ("wfh_share_3m", 0.4, "float"),
        ("unpaid_leave_days_6m", 0.0, "float"),
        ("paid_leave_days_6m", 3, "int"),
        ("dept_attrition_rate_prior_12m", 0.10, "float"),
    ]

    with st.form("attrition_form"):
        st.caption("These are editable example values, not actual employee records.")
        values = {}
        groups = [
            ("Employment profile", fields[:7]),
            ("Experience and compensation", fields[7:19]),
            ("Performance and work patterns", fields[19:31]),
        ]
        for group_name, group_fields in groups:
            with st.expander(group_name, expanded=(group_name == "Employment profile")):
                cols = st.columns(2)
                for i, (name, default, kind) in enumerate(group_fields):
                    with cols[i % 2]:
                        label = name.replace("_", " ").title()
                        if kind == "str":
                            values[name] = st.text_input(label, value=str(default), key=f"attr_{name}")
                        elif kind == "int":
                            values[name] = st.number_input(label, value=int(default), step=1, key=f"attr_{name}")
                        else:
                            values[name] = st.number_input(label, value=float(default), step=0.1, format="%.4f", key=f"attr_{name}")
        submitted = st.form_submit_button("Predict attrition risk", type="primary")

    if submitted:
        try:
            result = api_post("/v1/attrition/predict", {"features": values})
            st.markdown("#### Prediction result")
            c1, c2, c3 = st.columns(3)
            c1.metric("Risk probability", f"{float(result.get('attrition_probability', 0)):.1%}")
            c2.metric("Prediction", str(result.get("prediction_label", result.get("label", result.get("predicted_class", "N/A")))))
            c3.metric("Threshold", str(result.get("threshold", "N/A")))
            st.json(result)
            st.warning(result.get("notice", "For human review only. Do not use as the sole basis for employment decisions."))
        except requests.RequestException as e:
            show_api_error(e)

# ---------------------------------------------------------
# 2. ATTENDANCE
# ---------------------------------------------------------
with tabs[1]:
    st.subheader("Attendance Anomaly Detection")
    st.write("Enter monthly attendance summary values.")
    with st.form("attendance_form"):
        c1, c2, c3 = st.columns(3)
        scheduled_days = c1.number_input("Scheduled days", min_value=1, value=22)
        present_days = c2.number_input("Present days", min_value=0, value=20)
        wfh_days = c3.number_input("Work from home days", min_value=0, value=5)
        half_days_unpaid = c1.number_input("Unpaid half days", min_value=0, value=0)
        paid_leave_days = c2.number_input("Paid leave days", min_value=0, value=2)
        unpaid_absence_days = c3.number_input("Unpaid absence days", min_value=0, value=0)
        longest_unpaid_streak_days = c1.number_input("Longest unpaid absence streak", min_value=0, value=0)
        late_arrivals = c2.number_input("Late arrivals", min_value=0, value=2)
        avg_hours_worked = c3.number_input("Average hours worked", min_value=0.0, value=8.0)
        overtime_hours = c1.number_input("Overtime hours", min_value=0.0, value=4.0)
        attendance_rate = c2.number_input("Attendance rate", min_value=0.0, max_value=1.0, value=0.91, step=0.01)
        attendance_submit = st.form_submit_button("Check attendance", type="primary")

    if attendance_submit:
        payload = {
            "scheduled_days": scheduled_days,
            "present_days": present_days,
            "wfh_days": wfh_days,
            "half_days_unpaid": half_days_unpaid,
            "paid_leave_days": paid_leave_days,
            "unpaid_absence_days": unpaid_absence_days,
            "longest_unpaid_streak_days": longest_unpaid_streak_days,
            "late_arrivals": late_arrivals,
            "avg_hours_worked": avg_hours_worked,
            "overtime_hours": overtime_hours,
            "attendance_rate": attendance_rate,
        }
        try:
            result = api_post("/v1/anomaly/attendance", payload)
            st.markdown("#### Attendance result")
            st.metric("Anomaly score", result.get("anomaly_score", "N/A"))
            st.write("Predicted anomaly:", result.get("predicted_anomaly"))
            st.write("Protected by paid leave:", result.get("protected_by_paid_leave"))
            st.json(result)
            st.warning(result.get("notice", "Review anomalies in context; do not penalise protected leave."))
        except requests.RequestException as e:
            show_api_error(e)

# ---------------------------------------------------------
# 3. PAYROLL
# ---------------------------------------------------------
with tabs[2]:
    st.subheader("Payroll Anomaly Detection")
    st.write("Use synthetic or authorised payroll data. A unique payroll item ID is generated for each submission.")
    with st.form("payroll_form"):
        c1, c2 = st.columns(2)
        payroll_item_id = c1.text_input("Payroll item ID", value=f"DEMO-{uuid.uuid4().hex[:10].upper()}")
        employee_id = c2.text_input("Employee ID", value="DEMO-EMP-001")
        month = c1.text_input("Month (YYYY-MM-DD)", value="2026-10-01")
        run_id = c2.text_input("Payroll run ID", value="DEMO-RUN-001")
        monthly_fixed_gross = c1.number_input("Monthly fixed gross", min_value=0.0, value=50000.0)
        gross_pay = c2.number_input("Gross pay", min_value=0.0, value=52000.0)
        total_deductions = c1.number_input("Total deductions", min_value=0.0, value=5000.0)
        net_pay = c2.number_input("Net pay", min_value=0.0, value=47000.0)
        overtime_pay = c1.number_input("Overtime pay", min_value=0.0, value=2000.0)
        variable_pay = c2.number_input("Variable pay", min_value=0.0, value=0.0)
        lop_days = c1.number_input("LOP days", min_value=0.0, value=0.0)
        overtime_hours_paid = c2.number_input("Overtime hours paid", min_value=0.0, value=4.0)
        overtime_eligible = c1.selectbox("Overtime eligible", ["Y", "N"])
        pf_employee = c2.number_input("Employee PF", min_value=0.0, value=1800.0)
        payroll_submit = st.form_submit_button("Check payroll", type="primary")

    if payroll_submit:
        payload = {
            "payroll_item_id": payroll_item_id,
            "employee_id": employee_id,
            "month": month,
            "run_id": run_id,
            "monthly_fixed_gross": monthly_fixed_gross,
            "gross_pay": gross_pay,
            "total_deductions": total_deductions,
            "net_pay": net_pay,
            "overtime_pay": overtime_pay,
            "variable_pay": variable_pay,
            "lop_days": lop_days,
            "overtime_hours_paid": overtime_hours_paid,
            "overtime_eligible": overtime_eligible,
            "pf_employee": pf_employee,
        }
        try:
            result = api_post("/v1/anomaly/payroll", payload)
            st.markdown("#### Payroll result")
            st.metric("Anomaly score", result.get("anomaly_score", "N/A"))
            st.write("Model flag:", result.get("model_flag"))
            st.write("Rule flag:", result.get("rule_flag"))
            st.write("Anomaly flag:", result.get("anomaly_flag"))
            st.write("Reason codes:", result.get("reason_codes"))
            st.json(result)
            st.warning(result.get("notice", "Review payroll anomalies before taking action."))
        except requests.RequestException as e:
            show_api_error(e)

# ---------------------------------------------------------
# 4. SALARY
# ---------------------------------------------------------
with tabs[3]:
    st.subheader("Salary Prediction")
    st.write("Estimate annual fixed salary using job and experience characteristics.")
    with st.form("salary_form"):
        c1, c2 = st.columns(2)
        hire_year = c1.number_input("Hire year", min_value=1990, max_value=2035, value=2026)
        job_level = c2.number_input("Job level", min_value=1, max_value=10, value=2)
        prior_experience_years = c1.number_input("Prior experience (years)", min_value=0.0, value=2.0)
        career_gap_months = c2.number_input("Career gap (months)", min_value=0.0, value=0.0)
        skills_count = c1.number_input("Skills count", min_value=0, value=5)
        market_index = c2.number_input("Market index", min_value=0.0, max_value=1.0, value=0.8)
        department_id = c1.text_input("Department ID", value="D01")
        role_id = c2.text_input("Role ID", value="R002")
        work_location = c1.text_input("Work location", value="Chennai")
        highest_education = c2.text_input("Highest education", value="B.Tech Information Technology")
        salary_submit = st.form_submit_button("Predict salary", type="primary")

    if salary_submit:
        payload = {
            "hire_year": hire_year,
            "job_level": job_level,
            "prior_experience_years": prior_experience_years,
            "career_gap_months": career_gap_months,
            "skills_count": skills_count,
            "market_index": market_index,
            "department_id": department_id,
            "role_id": role_id,
            "work_location": work_location,
            "highest_education": highest_education,
        }
        try:
            result = api_post("/v1/salary/predict", payload)
            salary = result.get("estimated_annual_fixed_salary")
            if salary is not None:
                st.metric("Estimated annual fixed salary", f"₹{float(salary):,.0f}")
            st.json(result)
            st.warning(result.get("notice", "This is an estimate for decision support, not a salary commitment."))
        except requests.RequestException as e:
            show_api_error(e)

# ---------------------------------------------------------
# 5. RESUME SCREENING
# ---------------------------------------------------------
with tabs[4]:
    st.subheader("Resume Screening")
    st.write("Rank resumes for a selected job using the API's configured ranking method.")
    job_options = {
        "J01": "Software Engineer",
        "J02": "Senior Software Engineer",
        "J03": "Data Analyst",
        "J04": "ML Engineer",
        "J05": "Product Manager",
        "J06": "Customer Support Executive",
        "J07": "HR Executive",
        "J08": "Accountant",
        "J09": "Account Executive",
        "J10": "Digital Marketing Executive",
    }
    with st.form("resume_form"):
        selected_job = st.selectbox(
            "Job role",
            list(job_options.keys()),
            format_func=lambda x: f"{x} \u2014 {job_options[x]}",
            index=3,
        )
        top_n = st.slider("Number of candidates", min_value=1, max_value=50, value=5)
        resume_submit = st.form_submit_button("Rank candidates", type="primary")

    if resume_submit:
        try:
            result = api_post("/v1/resume/rank", {"job_id": selected_job, "top_n": top_n})
            st.write("Job:", result.get("job_title"))
            st.write("Ranking method:", result.get("ranking_method"))
            st.write("Candidates returned:", result.get("candidates_returned"))
            candidates = result.get("candidates", [])
            if candidates:
                st.dataframe(pd.DataFrame(candidates), use_container_width=True, hide_index=True)
            else:
                st.info("No candidates were returned.")
            st.json({k: v for k, v in result.items() if k != "candidates"})
            st.warning(result.get("notice", "Resume ranking is decision support. Human review is required."))
        except requests.RequestException as e:
            show_api_error(e)

# ---------------------------------------------------------
# 6. WORKFORCE FORECAST
# ---------------------------------------------------------
with tabs[5]:
    st.subheader("Workforce Forecast")
    st.write("Explore forecast outputs by metric and department.")
    metric_options = {
        "headcount_end": "Ending headcount",
        "exits": "Exits",
        "attrition_rate_monthly_pct": "Monthly attrition rate (%)",
    }
    with st.form("forecast_form"):
        selected_metric = st.selectbox(
            "Forecast metric",
            list(metric_options.keys()),
            format_func=lambda x: f"{x} \u2014 {metric_options[x]}",
        )
        selected_department = st.selectbox(
            "Department",
            ["ALL"] + [f"D{i:02d}" for i in range(1, 11)],
        )
        forecast_submit = st.form_submit_button("Load forecast", type="primary")

    if forecast_submit:
        try:
            result = api_get(
                "/v1/forecast",
                params={
                    "metric": selected_metric,
                    "department_id": selected_department,
                },
            )
            st.metric("Forecast rows returned", result.get("count", 0))
            st.write("Filters:", result.get("filters"))
            forecasts = result.get("forecasts", [])
            if forecasts:
                frame = pd.DataFrame(forecasts)
                st.dataframe(frame, use_container_width=True, hide_index=True)
                numeric_columns = frame.select_dtypes(include="number").columns.tolist()
                if numeric_columns:
                    st.line_chart(frame.set_index(frame.columns[0])[numeric_columns])
            else:
                st.info("No forecast rows returned for these filters.")
        except requests.RequestException as e:
            show_api_error(e)

st.divider()
st.caption(
    "GrowMar Phase 4 | Public synthetic-data demonstration. "
    "Zenith Nova is fictional. All datasets and outputs are synthetic. "
    "Not real employee assessments, hiring decisions, or legal advice. "
    "Human review is required for every decision."
)

