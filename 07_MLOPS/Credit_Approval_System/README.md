# 🚀 Credit Approval MLOps Project

End-to-end MLOps system for predicting credit approval using FastAPI, ML pipelines, Docker, and CI automation.

---

## 📌 Project Overview

This project is a complete **end-to-end Machine Learning Operations (MLOps) system** that takes a credit approval dataset through the full ML lifecycle — from data ingestion to production deployment.

It demonstrates how real-world ML systems are built in industry using modular pipelines, experiment tracking, automated training, and API deployment.

The system includes:
- Data ingestion from MongoDB
- Data validation using schema checks (Pandera)
- Feature engineering & preprocessing pipelines (Scikit-learn)
- Model training with multiple ML algorithms and hyperparameter tuning (Optuna)
- Experiment tracking using MLflow and DagsHub
- Model serving through a FastAPI web application
- Containerization using Docker & Docker Compose
- Automated testing and CI pipeline using GitHub Actions

---

## 📌 Features

- FastAPI web application for real-time predictions
- MongoDB-based data ingestion pipeline
- Data validation with Pandera schema enforcement
- Feature engineering using Scikit-learn pipelines
- Model training with multiple algorithms (RandomForest, XGBoost, SVC, etc.)
- Hyperparameter optimization using Optuna
- MLflow + DagsHub experiment tracking
- Docker + Docker Compose deployment
- CI pipeline using GitHub Actions
- Pytest automated testing

---

## 🏗️ Architecture

```

Data → Ingestion → Validation → Transformation → Training → Model → FastAPI → Prediction

````

---

## ⚙️ Tech Stack

- Python 3.13
- FastAPI
- Scikit-learn
- XGBoost
- Pandas / NumPy
- Optuna
- MLflow
- DagsHub
- MongoDB
- Docker
- Docker Compose
- GitHub Actions
- Pytest

---

## 🚀 Run Locally

### 1. Install dependencies
```bash
uv sync --frozen --no-install-project
````

---

### 2. Run training pipeline

```bash
python main.py
```

---

### 3. Start FastAPI application

```bash
uvicorn app:app --reload
```

Open in browser:

```
http://127.0.0.1:8000
```

---

## 🐳 Run with Docker

### Build image

```bash
docker build -t credit-approval-app .
```

### Run with Docker Compose

```bash
docker compose up --build
```

---

## 🧪 Run Tests

Run all tests using pytest:

```bash
pytest
```

or

```bash
python -m pytest
```

---

## 🔁 CI Pipeline (GitHub Actions)

This project uses a **CI (Continuous Integration) pipeline**.

It automatically runs on every push to `main`.

### CI Stages:

#### 1. Python Testing Stage

* Install dependencies
* Run pytest unit tests
* Validate FastAPI routes and ML pipeline

#### 2. Docker Integration Stage

* Build Docker image
* Run container using Docker Compose
* Execute tests inside running container

---

## 🔐 Environment Variables

The following environment variables are required:

```
MONGO_USERNAME
MONGO_PASSWORD
MLFLOW_TRACKING_USERNAME
MLFLOW_TRACKING_PASSWORD
DAGSHUB_USER_TOKEN
```

These are injected securely using GitHub Secrets in CI/CD.

---

## 📊 ML Pipeline Details

### 1. Data Ingestion

* Reads data from MongoDB
* Saves raw dataset as artifact

### 2. Data Validation

* Uses Pandera schema validation
* Splits valid and invalid data

### 3. Data Transformation

* Handles missing values
* One-hot encoding for categorical features
* Feature scaling
* Train-test split

### 4. Model Training

* Trains multiple models:

  * Decision Tree
  * Random Forest
  * XGBoost
  * SVC
  * MLP
* Uses Optuna for hyperparameter tuning
* Tracks experiments using MLflow

### 5. Model Serving

* Best model is saved as artifact
* Loaded into FastAPI for prediction

---

## 🌐 API Endpoints

### GET /

Redirects to prediction page

### GET /predict

Returns prediction HTML form

### POST /predict

Returns credit approval prediction

### GET/POST /train

Triggers model training pipeline

---

## 🧠 Key Highlights

* Full ML lifecycle automation
* Production-style modular architecture
* CI pipeline with Docker integration testing
* Experiment tracking with MLflow + DagsHub
* Real-world FastAPI deployment pattern
