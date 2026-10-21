# CortexHR — Phase 1: Data Foundations and Baselines

## Project overview

CortexHR Phase 1 establishes a reproducible machine-learning foundation for the fictional Zenith Nova workforce dataset. It covers data-quality auditing, reusable feature engineering, employee-attrition baseline modelling, risk evaluation, and attendance-anomaly validation.

**Scope:** Week 1 tasks CX-01 to CX-05. Later-phase features and public deployment are not claimed as complete here.

## Phase 1 tasks and evidence

| Task | Description | Evidence |
|---|---|---|
| CX-01 | Data audit and leakage investigation | `reports/data-audit.md`, `reports/data-audit-overview.png` |
| CX-02 | Reusable preprocessing and feature pipeline | `src/features/`, `TASK2_EVIDENCE.md`, `tests/` |
| CX-03 | Dummy, Logistic Regression, and Decision Tree baselines | `reports/baseline-report.md`, `reports/baseline_benchmark_comparison.csv`, `reports/baseline_metrics.csv`, `reports/baseline_confusion_matrix.png`, `reports/baseline_roc_curve.png` |
| CX-04 | Attrition evaluation, calibration, and risk bands | `reports/attrition_evaluation.md`, `reports/attrition_evaluation_metrics.csv`, `reports/attrition_risk_band_counts.csv`, `reports/attrition_calibration_curve.png`, `reports/attrition_lift_chart.png` |
| CX-05 | Attendance anomaly detection and validation | `reports/attendance-anomaly-report.md`, `reports/attendance-validation-report.md`, `reports/attendance_validation_metrics.csv`, `reports/attendance_validation_by_type.csv`, `reports/attendance_validation_by_type.png` |


## Technology stack

- Python
- pandas and NumPy
- scikit-learn
- Matplotlib
- pytest
- joblib
- Git

Refer to `requirements.txt` for the project's declared Python dependencies.

## Prerequisites

- Python 3.12 or a compatible Python version
- Git
- Windows PowerShell or another terminal

## Run locally

From the repository root:

1. Create a virtual environment:

   ```powershell
   python -m venv .venv
   ```

2. Activate it in PowerShell:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

3. Install the declared dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

4. Run the automated tests:

   ```powershell
   python -m pytest -v
   ```

5. To regenerate the Phase 1 PDF report, run:

   ```powershell
   python src/create_phase1_report.py
   ```

   The report generator expects the evaluation evidence files referenced in its source code to exist.

## Results and validation

The current local test run completed with **5 passed tests**.

The baseline and evaluation reports document model metrics, the class-imbalance limitation, risk-band results, and attendance-anomaly catch rates. See the relevant files under `reports/` for the detailed results.

These results are experimental and should not be treated as proof of production readiness. The attendance validation data contains labelled anomaly examples without a confirmed normal comparison set, limiting precision and false-alarm evaluation.

## Architecture

The Phase 1 workflow is documented in [`ARCHITECTURE.md`](ARCHITECTURE.md), including a Mermaid diagram showing data auditing, feature engineering, preprocessing, baseline modelling, evaluation, and reporting.

The diagram covers the implemented Phase 1 workflow only. The unified API, hosted console, and public deployment remain later-phase deliverables.
## Data notice

The GrowMar project overview identifies the Zenith Nova workforce data as fictional training data. Do not interpret project results as findings about real employees or a real organization.

## Limitations and next steps

- Compare outputs with all official acceptance checks and answer keys provided for the phase.
- Improve and expand automated tests where needed.
- Complete repository security and Git-history checks.
- Add a verified architecture diagram and release notes.
- Continue with the later-phase tasks, including the unified API and public deployment, according to the project schedule.
- Document model limitations and fairness considerations before any real-world use.

## Report

The Phase 1 summary report is `AIML_Ashwitha_Phase1_Report.pdf`.

## Repository

GitHub: https://github.com/ashwithagnanasekaran/Growmar_phase1
