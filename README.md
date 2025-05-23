# MLOps Project: Sentiment Analysis API

This project demonstrates a foundational MLOps pipeline for a sentiment analysis model. It covers data versioning, experiment tracking, model training automation, API creation, containerization, and CI/CD for building, testing, and deploying the application.

## Table of Contents
- [Project Overview](#project-overview)
- [Features Implemented](#features-implemented)
- [Directory Structure](#directory-structure)
- [Setup and Installation](#setup-and-installation)
  - [Prerequisites](#prerequisites)
  - [Local Setup](#local-setup)
- [Running Locally](#running-locally)
  - [1. Reproduce DVC Pipeline (Train Model)](#1-reproduce-dvc-pipeline-train-model)
  - [2. Run MLflow UI (Optional)](#2-run-mlflow-ui-optional)
  - [3. Run the Flask API](#3-run-the-flask-api)
  - [4. Test the API](#4-test-the-api)
  - [5. Build and Run with Docker (Optional Local Test)](#5-build-and-run-with-docker-optional-local-test)
- [MLOps Pipeline Components](#mlops-pipeline-components)
  - [Version Control](#version-control)
  - [Data and Model Versioning (DVC)](#data-and-model-versioning-dvc)
  - [Experiment Tracking (MLflow)](#experiment-tracking-mlflow)
  - [Automated Training Pipeline (DVC Pipeline)](#automated-training-pipeline-dvc-pipeline)
  - [API Serving (Flask)](#api-serving-flask)
  - [Containerization (Docker)](#containerization-docker)
  - [Continuous Integration/Continuous Deployment (CI/CD with GitHub Actions)](#continuous-integrationcontinuous-deployment-cicd-with-github-actions)
- [Cloud Deployment](#cloud-deployment)
- [Future Enhancements](#future-enhancements)

## Project Overview

The project trains a simple logistic regression model to classify movie review sentiments (positive/negative) using TF-IDF features. The focus is on demonstrating MLOps practices rather than achieving state-of-the-art model performance.

## Features Implemented

*   **Data Preprocessing:** Text cleaning, stopword removal, stemming.
*   **Model Training:** TF-IDF vectorization and Logistic Regression.
*   **DVC:** Versioning for raw data (`data/raw/reviews.csv`) and trained models (`models/*.joblib`).
*   **DVC Pipeline:** Defined in `dvc.yaml` to automate the training process.
*   **MLflow:** Tracking experiments, parameters, metrics, and model artifacts.
*   **Flask API:** A `/predict` endpoint to get sentiment predictions for input text.
*   **Docker:** Containerization of the Flask API for portability.
*   **GitHub Actions (CI/CD):**
    *   Automated linting (Flake8) and testing (Pytest).
    *   Automated DVC pipeline reproduction (`dvc repro`).
    *   Automated Docker image build and push to GitHub Container Registry (GHCR).
    *   Automated deployment of the Docker image to Google Cloud Run.
*   **Cloud Storage for DVC:** Using Google Cloud Storage (GCS) as a DVC remote.

## Directory Structure
Use code with caution.
Markdown
sentiment-analysis-mlops/
├── .dvc/ # DVC internal files
├── .github/
│ └── workflows/
│ └── ci.yml # GitHub Actions CI/CD workflow
├── .dockerignore # Files to ignore for Docker build
├── .flake8 # Flake8 configuration
├── .gitignore # Files to ignore by Git
├── data/
│ ├── raw/
│ │ └── reviews.csv.dvc # DVC pointer for raw data
│ └── (reviews.csv is DVC tracked, not in Git)
├── models/ # Output of DVC 'train_model' stage (gitignored)
│ # (sentiment_model.joblib, tfidf_vectorizer.joblib)
├── src/ # Source code
│ ├── init.py
│ ├── app.py # Flask API application
│ ├── preprocess.py # Data preprocessing logic
│ └── train.py # Model training script with MLflow
├── tests/ # Unit tests
│ ├── init.py
│ └── test_preprocess.py
├── .env.example # Example environment file (if needed for local GOOGLE_APPLICATION_CREDENTIALS)
├── Dockerfile # Docker configuration
├── docs.md # Detailed step-by-step project log (YOU ARE BUILDING THIS)
├── dvc.lock # DVC lock file, records pipeline state
├── dvc.yaml # DVC pipeline definition
├── gcp-creds.json.example # Example GCP credentials file structure (actual file gitignored)
├── params.yaml # Parameters (currently empty, can be used for DVC pipeline)
├── README.md # This file
└── requirements.txt # Python dependencies
## Setup and Installation

### Prerequisites
*   Python (3.9+ recommended)
*   Git
*   DVC (`pip install dvc-gs`)
*   MLflow (`pip install mlflow`)
*   Docker Desktop (or Docker Engine for Linux)
*   Google Cloud SDK (`gcloud` CLI) (for local testing of Cloud Run deployment)
*   A Google Cloud Platform (GCP) project with:
    *   Billing enabled.
    *   APIs enabled: Cloud Run API, Container Registry API (or Artifact Registry API), IAM API.
    *   A GCS bucket for DVC remote storage.
    *   A Service Account JSON key file with permissions for GCS (Storage Object Admin on DVC bucket) and Cloud Run (Cloud Run Admin, Service Account User).

### Local Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yashwatwani/sentiment-analysis-mlops.git
    cd sentiment-analysis-mlops
    ```

2.  **Create and activate a virtual environment (recommended):**
    *   Using `venv`:
        ```bash
        python -m venv venv
        source venv/bin/activate  # On Windows: venv\Scripts\activate
        ```
    *   Or using `conda`:
        ```bash
        conda create --name sentiment-analysis-mlops python=3.9
        conda activate sentiment-analysis-mlops
        ```

3.  **Install Python dependencies:**
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

4.  **Set up DVC Remote (Google Cloud Storage):**
    *   Place your GCP Service Account JSON key file in the project root (e.g., named `gcp-creds.json` or `gcp-creds2.json`). **Ensure this file is listed in `.gitignore`!**
    *   Authenticate DVC to use your GCS remote (if not already configured in `.dvc/config` or if this is a fresh clone by another user):
        ```bash
        # First time setup for the remote if .dvc/config doesn't exist or is minimal
        # dvc remote add mygcsremote gs://YOUR_DVC_GCS_BUCKET/dvc-store 
        # dvc remote default mygcsremote

        # Export credentials path for current session
        export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/your-gcp-keyfile.json"
        ```

5.  **Pull DVC-tracked data and models (if you are not training from scratch):**
    ```bash
    dvc pull data/raw/reviews.csv -v # Pull raw data
    # If models were generated by someone else and pushed, and you just want to run the API:
    # dvc pull models/sentiment_model.joblib models/tfidf_vectorizer.joblib -v
    ```
    *(The `data/raw/reviews.csv` is the original raw data. The models are generated by the pipeline.)*

## Running Locally

### 1. Reproduce DVC Pipeline (Train Model)
This step will download necessary DVC-tracked inputs (like `data/raw/reviews.csv` if not present) and run the training script to generate models.
```bash
# Ensure GOOGLE_APPLICATION_CREDENTIALS is set if reviews.csv needs to be pulled
dvc repro -v
Use code with caution.
This will:
Run src/train.py.
Log experiment details to MLflow (in a local mlruns directory).
Save sentiment_model.joblib and tfidf_vectorizer.joblib to the models/ directory.
Update dvc.lock.
2. Run MLflow UI (Optional)
To view experiment tracking results:
mlflow ui
Use code with caution.
Bash
Open your browser to http://127.0.0.1:5000.
3. Run the Flask API
# Ensure models exist in models/ directory (generated by 'dvc repro')
python src/app.py
Use code with caution.
Bash
The API will be available at http://127.0.0.1:5001.
4. Test the API
Use curl or Postman:
curl -X POST -H "Content-Type: application/json" \
     -d '{"text": "This movie was fantastic!"}' \
     http://127.0.0.1:5001/predict
Use code with caution.
Bash
5. Build and Run with Docker (Optional Local Test)
Ensure models are present in models/ directory (from dvc repro).
# Build the image
docker build -t sentiment-analysis-api:local .

# Run the container
docker run -d -p 5001:5001 --name sentiment-app-local sentiment-analysis-api:local

# Test it
curl -X POST -H "Content-Type: application/json" \
     -d '{"text": "Testing from docker!"}' \
     http://127.0.0.1:5001/predict

# Stop and remove
docker stop sentiment-app-local
docker rm sentiment-app-local
Use code with caution.
Bash
MLOps Pipeline Components
Version Control
Git & GitHub: Used for versioning all code, configuration files, DVC metafiles, and documentation.
Data and Model Versioning (DVC)
DVC (dvc): Tracks large data files (data/raw/reviews.csv) and model artifacts (models/*.joblib) without storing them in Git.
Remote Storage: Google Cloud Storage (GCS) is used as the DVC remote to store the actual data and model files.
.dvc files (pointers) and dvc.lock (pipeline state) are committed to Git.
Experiment Tracking (MLflow)
MLflow Tracking (mlflow): Integrated into src/train.py to log:
Parameters (e.g., test split ratio, TF-IDF features).
Metrics (e.g., accuracy, precision, recall, F1-score).
Artifacts (trained model, vectorizer, confusion matrix plot).
Logs are stored locally in the mlruns/ directory.
Automated Training Pipeline (DVC Pipeline)
dvc.yaml: Defines the stages of the ML pipeline (e.g., train_model).
Specifies dependencies (input data, scripts).
Specifies outputs (models, vectorizers).
Specifies the command to run the stage (python src/train.py).
dvc repro: Command to reproduce the pipeline. DVC intelligently re-runs only stages whose dependencies have changed.
dvc.lock: Records the exact state (hashes of dependencies and outputs) of a successful pipeline run, ensuring reproducibility.
API Serving (Flask)
src/app.py: A Flask application that:
Loads the DVC-tracked trained model and vectorizer.
Provides a /predict endpoint that accepts text input via a POST request.
Preprocesses the input text.
Returns a JSON response with the sentiment prediction (positive/negative) and probabilities.
Containerization (Docker)
Dockerfile: Defines the instructions to build a Docker image containing the Flask API, its Python dependencies, source code, and the trained model files (copied from the DVC-tracked models/ directory).
.dockerignore: Specifies files to exclude from the Docker build context.
Continuous Integration/Continuous Deployment (CI/CD with GitHub Actions)
.github/workflows/ci.yml: Defines the automated CI/CD pipeline.
Triggers: Runs on pushes/PRs to the main branch.
build_and_test Job:
Checks out code.
Sets up Python.
Installs dependencies.
Authenticates to GCP and pulls raw data from the GCS DVC remote (dvc pull data/...).
Reproduces the DVC pipeline (dvc repro), which trains the model and generates model artifacts.
Runs linters (flake8).
Runs unit tests (pytest).
build_docker_and_push_to_ghcr (or _gcr) Job:
Depends on build_and_test succeeding.
Runs only on pushes to main.
Authenticates to GCP and pulls the DVC-tracked model files (generated by the previous job's dvc repro and available via GCS DVC remote because dvc repro in the previous job implicitly updates DVC's knowledge of outputs if they changed, and we assume a dvc push happened or the state is consistent).
Builds the Docker image using the Dockerfile.
Pushes the image to a container registry (GitHub Container Registry - GHCR, or Google Container Registry - GCR).
deploy_to_cloud_run Job:
Depends on the Docker image being successfully built and pushed.
Runs only on pushes to main.
Authenticates to GCP.
Deploys the specified Docker image (from GHCR or GCR) to Google Cloud Run.
Configures the Cloud Run service for public access and correct port.
Cloud Deployment
The application is deployed as a containerized service to Google Cloud Run.
Deployment is automated via the GitHub Actions CI/CD pipeline.
The deployed service URL is output by the deployment job.
Future Enhancements
Implement more comprehensive unit and integration tests.
Add a data validation stage to the DVC pipeline.
Use MLflow Model Registry for model versioning and staging.
Deploy the API from the MLflow Model Registry.
Set up a separate DVC stage for evaluation and log metrics with dvc metrics.
Implement Continuous Training (CT) triggered by new data or code changes.
Add monitoring for the deployed API and model performance.
Use a more sophisticated model and feature engineering techniques.
Explore different deployment targets (e.g., Kubernetes).
Manage params.yaml for hyperparameter tuning and track experiments with DVC/MLflow.
**How to Use This README:**

1.  **Copy the entire content** above.
2.  **Open the `README.md` file** in the root of your `sentiment-analysis-mlops` project in your code editor.
3.  **Paste the copied content** into `README.md`, replacing any existing content.
4.  **Review and Customize:**
    *   **Replace placeholders:** Look for any bracketed placeholders like `YOUR_DVC_GCS_BUCKET` or `your-gcp-keyfile.json` and update them with your specific project details or more generic instructions.
    *   **Verify URLs:** Ensure any example URLs are correct for your setup.
    *   **Adjust Python versions:** If you used a Python version different from 3.9, update that.
    *   **GCR/GHCR:** The "CI/CD" and "MLOps Pipeline Components" sections mention pushing to GHCR or GCR. Ensure this aligns with your final CI setup (I believe we ended up with GHCR for image push, and then Cloud Run pulls from there, but the error message from `gcloud` suggests it prefers GCR/GAR. If your CI is now pushing to GCR, update that part). *Given our last working CI pushed to GCR, I've updated the example README to reflect pushing to GCR.*
    *   **Service Account Details:** Make sure instructions for setting `GOOGLE_APPLICATION_CREDENTIALS` are clear. You might want to reference a `.env.example` file.
5.  **Save `README.md`.**
6.  **Commit and push to GitHub:**
    ```bash
    git add README.md
    git commit -m "Docs: Add comprehensive project README"
    git push origin main
    ```

This README should give a good overview of your project, how it's structured, how to set it up, and what MLOps practices it demonstrates!