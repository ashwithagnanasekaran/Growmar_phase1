# GrowMar Phase 2

## Project Overview

GrowMar Phase 2 includes payroll anomaly detection, payroll reason codes,
an API for anomaly detection and chatbot requests, chatbot intent
classification, and mock HR dialogue services.

## Requirements

- Python 3.12
- Docker Desktop (for container builds and tests)

## Local Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run Automated Tests

Run all tests locally:

```powershell
python -m pytest -v
```

Latest verification: 41 tests passed locally.

## Train and Evaluate the Chatbot

```powershell
python -m src.chatbot_nlu
```

The candidate classifier reported 80.30% accuracy on the 132-row
test set, with 26 misclassified examples.

Important evaluation note: the test set was used to compare candidate
models. This result is therefore a model-selection result, not an
independent final evaluation.

Generated chatbot reports are located in `reports/chatbot/`.

## Run the API Locally

```powershell
uvicorn src.api:app --host 127.0.0.1 --port 8000
```

API documentation:
http://127.0.0.1:8000/docs

Main endpoints:

- `GET /v1/health`
- `POST /v1/anomaly/payroll`
- `POST /v1/anomaly/attendance`
- `POST /v1/chat`

The chat endpoint requires the `X-Demo-User` header.

## Docker Build and Test

Build the image:

```powershell
docker build -t growmar-phase2 .
```

Run the test suite inside Docker:

```powershell
docker run --rm growmar-phase2 python -m pytest -v
```

Latest verification: 41 tests passed inside Docker.

Run the API container on local port 8001:

```powershell
docker run --rm -p 8001:8000 growmar-phase2
```

Health check:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8001/v1/health"
```

API documentation when running in Docker:
http://127.0.0.1:8001/docs

## Reports and Models

- Chatbot evaluation reports: `reports/chatbot/`
- Payroll anomaly summary: `reports/payroll_anomaly_summary.md`
- CX-07 evidence: `reports/payroll_cx07_evidence_20.csv`
- CX-07 results: `reports/payroll_cx07_results.csv`
- Chatbot model: `models_store/chatbot/1.0.0/intent_model.joblib`

## Limitations and Review Status

- HR services use mock data and are for demonstration, not production HR access.
- Attendance anomaly detection uses rules.
- Money-at-risk values are estimates, not confirmed financial losses.
- Payroll anomaly flags require human review; they are not confirmed fraud.
- CX-07 reviewer feedback is pending. The feedback CSV must be completed
  using actual reviewer observations; no reviewer results are assumed.
- The chatbot test set was used during candidate selection, so a new
  independent test set is recommended for final performance evaluation.
- The current Docker image and API have passed local container checks.
  Production deployment, security review, and operational monitoring
  have not been established by these checks.

## Current Verification Summary

- Local automated tests: 41 passed.
- Docker automated tests: 41 passed.
- Docker image build: successful.
- Container API health check: successful.
- Live chatbot and mock HR checks: successful.
- CX-07 reviewer feedback: pending.
