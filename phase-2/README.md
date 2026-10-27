# GrowMar Phase 2

## Project Overview

GrowMar Phase 2 contains payroll anomaly detection, payroll reason codes,
an API for anomaly detection and chatbot requests, chatbot intent
classification, and mock HR dialogue services.

## Repository contents and data handling

This repository copy intentionally excludes datasets, trained model files,
and generated reports while their approval for GitHub publication is being
confirmed.

The following directories are therefore not included in this copy:

- `data/`
- `models_store/`
- `reports/`

Some API operations, training commands, and tests require those files.
They may fail until approved artifacts are provided locally.

Do not add company datasets, employee records, trained models, or generated
reports to GitHub until their publication is explicitly approved.

## Requirements

- Python 3.12
- Docker Desktop (for container builds and tests)

## Local Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Automated Tests

Run tests locally:

```powershell
python -m pytest -v
```

**Verification status for this data-excluded copy:** the latest selected
test run reported 4 passed and 27 failed. Failures include missing data
files such as `data/raw/demo_users.csv` and
`data/raw/payroll_monthly_features.csv`, plus a missing generated CX-07
results file. These results do not establish that the full test suite
passes without the excluded artifacts.

The original Phase 2 workspace previously reported 41 local tests passed
and 41 Docker tests passed with its local artifacts present. Those results
have not been reproduced in this data-excluded integration copy.

## Run the API

The API entry point is `src/api.py`. Many operations depend on datasets
or trained models that are not included in this repository copy.

After the approved artifacts are made available locally, install the
requirements and verify the API and its endpoints before deployment.

## Docker

The Dockerfile copies application code and tests but intentionally does not
copy datasets, trained models, or generated reports. A successful image
build would not by itself demonstrate that model-dependent endpoints work.

Build:

```powershell
docker build -t growmar-phase2 .
```

Run the API container:

```powershell
docker run --rm -p 8001:8000 growmar-phase2
```

Verify health and model-dependent endpoints separately after the required
approved artifacts are available.

## Limitations and Review Status

- HR services use mock data and are for demonstration, not production HR access.
- Attendance anomaly detection uses rules.
- Money-at-risk values are estimates, not confirmed financial losses.
- Payroll anomaly flags require human review; they are not confirmed fraud.
- CX-07 reviewer feedback remains pending; no reviewer results are assumed.
- The chatbot candidate's reported 80.30% accuracy came from a test set
  used during model comparison, so it is not an independent final evaluation.
- A new independent test set is recommended for final performance evaluation.
- Production deployment, security review, and operational monitoring have
  not been established by the checks described here.
