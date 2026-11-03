# GrowMar Phase 3 – AI/ML Engineering

## Overview

Implementation of GrowMar AI/ML Tasks 11–15.

## Tasks

- Task 11: Resume ranking and job-description matching.
- Task 12: Screener fairness, masking, and ranking-change audits.
- Task 13: Attrition prediction model v2.
- Task 14: Employee attrition risk scoring and SHAP explanations.
- Task 15: Salary prediction, salary-band recommendations, and fairness evaluation.

## Technology

- Python 3.12
- Pandas and NumPy
- Scikit-learn
- XGBoost
- SHAP
- Matplotlib
- Pytest
- Docker

## Setup

Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run tests

```powershell
python -m pytest -v
```

## Docker

Build the Docker image:

```powershell
docker build -t growmar-phase3:1.0.0 .
```

Run the test container with the local data folder mounted:

```powershell
docker run --rm --mount "type=bind,source=<LOCAL_DATA_PATH>,target=/app/data,readonly" growmar-phase3:1.0.0
```

## Data and model handling

- Raw datasets and trained model files are excluded from Git.
- Employee-level CSV outputs and generated plots are excluded from Git.
- Keep company data confidential and use it only for authorized work.

## Responsible use

- Model predictions are decision-support outputs, not final HR decisions.
- SHAP values describe model associations, not causal effects.
- Fairness metrics should be interpreted with the evaluation sample sizes and limitations.
