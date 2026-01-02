# 🚨 Time Series Anomaly Detection  
## Production-Ready Inference API (FastAPI + Docker)

A **production-grade anomaly detection system** for multivariate industrial time series data, built using **unsupervised learning (PCA on sliding windows)** and deployed as a **FastAPI-based inference service**.

The project focuses on **decision-level anomaly detection** — identifying **out-of-distribution behavior**, **persistent warnings**, and **critical anomalies** — instead of returning raw anomaly scores. This reflects how real industrial monitoring systems are designed and operated.

---

## 🎯 Project Objective

Industrial failures rarely appear as isolated spikes.  
They usually manifest as **persistent abnormal behavior over time**.

This system is designed to answer:

- Is a time series **out-of-distribution (OOD)**?
- Are anomalies **persistent** or transient?
- Should the system trigger a **warning** or a **critical alert**?
- Can multiple time series be **scored safely in batch** without failure propagation?

The result is a **robust, interpretable, and deployable anomaly decision pipeline**.

---

## 🏗️ System Architecture

CSV Sensor Data
│
▼
Sliding Window Segmentation
(window = 120, stride = 12)
│
▼
Feature Scaling
(trained on normal data)
│
▼
PCA Model
(normal behavior subspace)
│
▼
Reconstruction Error per Window
│
▼
Persistence Logic
(warning_k / critical_k)
│
▼
Final Decision
(OOD / WARNING / CRITICAL)

yaml
Copy code

---

## 📦 Repository Structure

ts-anomaly-detection/
├── src/tsad/
│ ├── api/ # FastAPI application
│ │ ├── routes/ # REST endpoints
│ │ ├── schemas/ # Pydantic request/response models
│ │ └── core/ # Settings, logging, middleware
│ ├── inference/ # Inference & decision logic
│ └── scripts/ # Training & evaluation pipeline
├── configs/
│ └── default.yaml # Model, window, threshold configuration
├── docker/
│ └── Dockerfile # Production Docker image
├── models/ # Trained PCA models
├── data/ # Local data (gitignored)
├── reports/ # Inference & evaluation outputs
├── requirements.txt
└── README.md

yaml
Copy code

---

## 🚀 API Endpoints

### ✅ Health Check

**GET `/health`**

```json
{
  "status": "ok",
  "app": "tsad-inference",
  "version": "0.1.0",
  "data_root": "/data",
  "reports_dir": "/app/reports"
}
🔍 Single File Decision
POST /decide

Request

json
Copy code
{
  "rel_path": "valve1/1.csv",
  "warning_k": 3,
  "critical_k": 5,
  "ood_margin": 0.0
}
Response

json
Copy code
{
  "ood": false,
  "warning": true,
  "critical": false,
  "scores": {
    "p50": 0.0517,
    "p95": 0.0899
  }
}
📦 Batch Decision (Production Feature)
Safely score multiple files in a single request.
A corrupted or invalid file does not break the batch.

POST /decide_many

Request

json
Copy code
{
  "rel_paths": [
    "anomaly-free/anomaly-free.csv",
    "valve1/1.csv",
    "valve1/3.csv",
    "valve2/1.csv"
  ],
  "warning_k": 3,
  "critical_k": 5,
  "ood_margin": 0.0
}
Response

json
Copy code
{
  "n_total": 4,
  "n_scored": 4,
  "n_errors": 0,
  "n_ood": 1,
  "n_warning": 3,
  "n_critical": 0,
  "n_ok": 0
}
🐳 Run with Docker (Recommended)
Build the Image
bash
Copy code
docker build -t tsad-inference:latest -f docker/Dockerfile .
Run the Service
bash
Copy code
docker run --rm -p 8000:8000 \
  -e TSAD_DATA_ROOT=/data \
  -e TSAD_REPORTS_DIR=/app/reports \
  -v $(pwd)/data/raw/skab_repo/SKAB-master/data:/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data/processed:/app/data/processed \
  tsad-inference:latest
Interactive API Documentation
Open your browser at:
👉 http://127.0.0.1:8000/docs

🧪 Dataset
SKAB — Skoltech Anomaly Benchmark

Realistic industrial multivariate sensor data

Normal and anomalous operating regimes

Widely used benchmark for time series anomaly detection

💡 Why This Project Stands Out
✔️ Production-style FastAPI inference service

✔️ Robust batch scoring with fault isolation

✔️ Decision logic based on anomaly persistence

✔️ Fully Dockerized deployment

✔️ Clean separation of training, inference, and API layers

✔️ Real industrial benchmark dataset

📌 Typical Use Cases
Industrial equipment monitoring

Predictive maintenance

Sensor drift and fault detection

Anomaly screening pipelines

MLOps and ML system design portfolios