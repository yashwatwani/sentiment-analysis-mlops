# MLOps Project: Sentiment Analysis

This document chronicles the step-by-step development of a sentiment analysis MLOps project.

## Table of Contents
- [Step 0: Project Setup and GitHub Repository](#step-0-project-setup-and-github-repository)

---

## Step 0: Project Setup and GitHub Repository

**Goal:** Create the GitHub repository, initial file structure, and your `docs.md` file.

**Explanation:**
Every good project starts with setting up version control and a basic structure. We'll create a public GitHub repository where all our code, configurations, and documentation will live. The `docs.md` file will be our main logbook for this learning journey.

**Process & Code:**

1.  **Create a GitHub Repository:**
    *   Go to [GitHub](https://github.com) and create a new public repository.
    *   Name it `sentiment-analysis-mlops` (or similar).
    *   Initialize it with a `README.md` file.
    *   Add a Python `.gitignore` template.

2.  **Clone the Repository Locally:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/sentiment-analysis-mlops.git # Replace YOUR_USERNAME
    cd sentiment-analysis-mlops
    ```

3.  **Create Initial Directory Structure:**
    Open your terminal in the `sentiment-analysis-mlops` directory and run:
    ```bash
    mkdir -p data/raw src tests notebooks
    touch src/__init__.py src/preprocess.py src/train.py src/app.py
    touch tests/test_model.py
    touch Dockerfile dvc.yaml params.yaml requirements.txt
    touch docs.md
    ```

4.  **Initialize `docs.md`:**
    This step was to create this file itself.

5.  **Populate `.gitignore`:**
    Open `.gitignore` and ensure it contains the following:

    ```gitignore
    # Byte-compiled / optimized / DLL files
    __pycache__/
    *.py[cod]
    *$py.class

    # C extensions
    *.so

    # Distribution / packaging
    .Python
    build/
    develop-eggs/
    dist/
    downloads/
    eggs/
    .eggs/
    lib/
    lib60/
    parts/
    sdist/
    var/
    wheels/
    pip-wheel-metadata/
    share/python-wheels/
    *.egg-info/
    .installed.cfg
    *.egg
    MANIFEST

    # PyInstaller
    #  Usually these files are written by a python script from a template
    #  before PyInstaller builds the exe, so as to inject date/other infos into it.
    *.manifest
    *.spec

    # Installer logs
    pip-log.txt
    pip-delete-this-directory.txt

    # Unit test / coverage reports
    htmlcov/
    .tox/
    .nox/
    .coverage
    .coverage.*
    .cache
    nosetests.xml
    coverage.xml
    *.cover
    *.py,cover
    .hypothesis/
    .pytest_cache/

    # Environments
    .env
    .venv
    env/
    venv/
    ENV/
    env.bak/
    venv.bak/

    # Spyder project settings
    .spyderproject
    .spyproject

    # Rope project settings
    .ropeproject

    # mkdocs documentation
    /site

    # Jupyter Notebook
    .ipynb_checkpoints
    notebooks/.ipynb_checkpoints/ # If notebooks are committed

    # Virtualenv
    venv/
    .venv/

    # DVC
    .dvc/cache
    data/raw # We'll let DVC track data, not Git directly for large files
    data/processed
    models/
    # *.dvc files SHOULD be committed to Git

    # MLflow
    mlruns/
    .mlflow_tracking_uri # if configured locally

    # IDE specific
    .idea/
    .vscode/
    *.DS_Store
    ```

6.  **Initial Commit to GitHub:**
    ```bash
    git add .
    git commit -m "Initial project structure and setup"
    ```
    *Sample Output:*
    ```
    [main 9d1d5a5] Initial project structure and setup
     11 files changed, 104 insertions(+)
     create mode 100644 .gitignore
     create mode 100644 Dockerfile
     create mode 100644 docs.md
     create mode 100644 dvc.yaml
     create mode 100644 params.yaml
     create mode 100644 requirements.txt
     create mode 100644 src/__init__.py
     create mode 100644 src/app.py
     create mode 100644 src/preprocess.py
     create mode 100644 src/train.py
     create mode 100644 tests/test_model.py
    ```
    ```bash
    git push origin main # Or your default branch name (e.g., master)
    ```
    *Sample Output (if successful):*
    ```
    Enumerating objects: 15, done.
    Counting objects: 100% (15/15), done.
    Delta compression using up to X threads
    Compressing objects: 100% (12/12), done.
    Writing objects: 100% (14/14), 2.50 KiB | 2.50 MiB/s, done.
    Total 14 (delta 0), reused 0 (delta 0), pack-reused 0
    To https://github.com/YOUR_USERNAME/sentiment-analysis-mlops.git
     * [new branch]      main -> main
    ```
    *Note: If you had issues with the push and corrected it, include the corrected command and its output.*

7.  **Commit `docs.md` update:**
    ```bash
    git add docs.md
    git commit -m "Docs: Update Step 0 - Project Setup"
    git push origin main
    ```

---

## Step 1: Creating a Basic Model Training Script

**Goal:** Develop a Python script to train a simple sentiment analysis model.

**Key Actions & Files:**
1.  **Data Acquisition:**
    *   Downloaded sentiment labelled sentences dataset (specifically `imdb_labelled.txt`).
    *   Saved as `data/raw/reviews.csv`. *Note: This file is gitignored; will be handled by DVC later.*

2.  **Dependencies:**
    *   Updated `requirements.txt` with `pandas`, `scikit-learn`, `nltk`.
    *   Set up a Python virtual environment and installed dependencies.

3.  **Preprocessing (`src/preprocess.py`):**
    *   Created a function `preprocess_text` to:
        *   Lowercase text.
        *   Remove numbers and punctuation.
        *   Remove stopwords (using NLTK).
        *   Apply Porter stemming.

4.  **Training Script (`src/train.py`):**
    *   Loads data from `data/raw/reviews.csv`.
    *   Applies `preprocess_text` to reviews.
    *   Splits data into training and testing sets (80/20).
    *   Uses `TfidfVectorizer` for feature extraction.
    *   Trains a `LogisticRegression` model.
    *   Evaluates the model (accuracy, classification report).
    *   Saves the trained model as `models/sentiment_model.joblib` and the vectorizer as `models/tfidf_vectorizer.joblib`. *Note: `models/` directory is gitignored; will be handled by DVC later.*

5.  **Execution & Output:**
    *   Ran `python src/train.py` successfully.
    *   Observed console output for training progress, evaluation metrics, and file saving confirmation.
    *   Model artifacts (`.joblib` files) were created in the `models/` directory.

6.  **Git Commits:**
    *   Committed `src/preprocess.py`, `src/train.py`, and `requirements.txt` to the repository.
    *   `data/` and `models/` directories (and their contents) were correctly ignored by Git as per `.gitignore`.

**Outcome:** A functional, local script that can train and save a sentiment analysis model and its associated vectorizer. The groundwork is laid for versioning data and models.


-----------

## Step 2: Data and Model Versioning with DVC

**Goal:** Integrate DVC to version control the dataset and trained model artifacts.

**Key Actions & Files:**
1.  **Installation & Setup:**
    *   Installed `dvc` (`pip install dvc`).
    *   Added `dvc` to `requirements.txt`.
    *   Initialized DVC in the project: `dvc init`. This created the `.dvc/` directory and updated relevant `.gitignore` files.
    *   (Recommended) Configured DVC to autostage metafiles: `dvc config core.autostage true`.
    *   Corrected main `.gitignore` to properly handle DVC metafiles (not ignore `*.dvc` files).

2.  **Tracking Raw Data:**
    *   Added `data/raw/reviews.csv` to DVC tracking: `dvc add data/raw/reviews.csv`.
    *   This created `data/raw/reviews.csv.dvc` (metafile) and ensured `reviews.csv` itself is gitignored (e.g., via `data/raw/.gitignore`).

3.  **Tracking Models:**
    *   Added trained model artifacts to DVC tracking: `dvc add models/sentiment_model.joblib models/tfidf_vectorizer.joblib`.
    *   This created `models/sentiment_model.joblib.dvc` and `models/tfidf_vectorizer.joblib.dvc`, and ensured the `.joblib` files are gitignored (e.g., via `models/.gitignore`).

4.  **Git Commits:**
    *   Committed the DVC metafiles (`*.dvc`), DVC configuration (`.dvc/config`), updated main `.gitignore`, new DVC-generated `.gitignore` files (in `data/raw/`, `models/`), and updated `requirements.txt` to Git.
    *   The actual large data and model files are *not* in Git history, only their DVC metafiles.

5.  **DVC Remote Storage (Local):**
    *   Created an external directory (e.g., `../sentiment-analysis-dvc-remote`) to act as DVC remote storage.
    *   Configured this as the default DVC remote: `dvc remote add -d mylocalremote ../sentiment-analysis-dvc-remote`.
    *   Committed the remote configuration update (`.dvc/config`) to Git.

6.  **Pushing to DVC Remote:**
    *   Pushed the DVC-tracked files (data and models) to the local remote: `dvc push`.

7.  **Demonstrated DVC Pull:**
    *   Simulated data loss by deleting a tracked file and restored it using `dvc pull` (or `dvc checkout` from local cache).

**Outcome:** The project now uses DVC to version control large data and model files. Git tracks the code and DVC metafiles, while DVC manages the actual artifacts and their storage, keeping the Git repository lean. This setup allows for reproducible ML experiments where data and model versions are tied to code versions.

-----------

## Step 3: Creating a DVC Pipeline

**Goal:** Define the model training process as a reproducible DVC pipeline stage.

**Key Actions & Files:**
1.  **Cleanup:**
    *   Removed manually added DVC metafiles for model outputs (`models/sentiment_model.joblib.dvc`, `models/tfidf_vectorizer.joblib.dvc`) because these outputs are now managed by the DVC pipeline via `dvc.lock`.
    *   Optionally removed the model files themselves to ensure a clean run by the pipeline.

2.  **Define Pipeline (`dvc.yaml`):**
    *   Created/Updated `dvc.yaml` to define the `train_model` stage:
        ```yaml
        stages:
          train_model:
            cmd: python src/train.py
            deps:
              - data/raw/reviews.csv
              - src/preprocess.py
              - src/train.py
            outs:
              - models/sentiment_model.joblib
              - models/tfidf_vectorizer.joblib
        ```
    *   `cmd`: Specifies the command to run the stage.
    *   `deps`: Lists data and code dependencies. Changes here trigger a re-run.
    *   `outs`: Lists outputs generated and cached by DVC for this stage. Information about these outputs (hashes, etc.) is stored in `dvc.lock`.

3.  **Run Pipeline (`dvc repro`):**
    *   Executed `dvc repro`. DVC ran the `train_model` stage.
    *   The script `src/train.py` executed, generating the model and vectorizer.
    *   DVC created/updated `dvc.lock`, which records the hashes of dependencies and outputs for the successful run. Individual `.dvc` files are no longer created for outputs of pipeline stages.

4.  **Git Commits:**
    *   Committed `dvc.yaml` (pipeline definition) and `dvc.lock` (pipeline state) to Git.
    *   Committed the deletion of the old, manually added model `.dvc` files.

5.  **Push DVC Outputs (`dvc push`):**
    *   Pushed the DVC-tracked outputs (model and vectorizer generated by the pipeline, as recorded in `dvc.lock`) to the DVC remote storage.

6.  **Reproducibility Test:**
    *   Running `dvc repro` again showed that the stage was skipped as dependencies hadn't changed.
    *   Simulated a change in a dependency (`src/train.py`).
    *   Running `dvc repro` again correctly detected the change and re-executed the `train_model` stage, updating `dvc.lock`.

**Outcome:** The project now has a DVC pipeline that automates the model training process. This pipeline is versioned (via `dvc.yaml` and `dvc.lock` in Git) and ensures reproducibility. Output artifacts are tracked via `dvc.lock` rather than individual `.dvc` files for pipeline outputs.

-------

## Step 4: Experiment Tracking with MLflow

**Goal:** Integrate MLflow to log parameters, metrics, and artifacts from training runs, enabling better experiment comparison.

**Key Actions & Files:**
1.  **Installation & Setup:**
    *   Installed `mlflow`, `matplotlib`, `seaborn` (`pip install ...`).
    *   Added them to `requirements.txt`.
    *   Added `mlruns/` (MLflow's local tracking data directory) to `.gitignore`.

2.  **Modified Training Script (`src/train.py`):**
    *   Imported `mlflow` and other necessary libraries.
    *   Set an MLflow experiment name: `mlflow.set_experiment("SentimentAnalysisIMDB")`.
    *   Wrapped the main training logic in `with mlflow.start_run() as run:`.
    *   **Logged Parameters:** Used `mlflow.log_param()` to log hyperparameters like `test_split_ratio`, `tfidf_max_features`.
    *   **Logged Metrics:** Used `mlflow.log_metric()` to log evaluation results like `accuracy`, and parsed `classification_report` to log class-wise precision, recall, and F1-score.
    *   **Logged Artifacts:**
        *   Created and logged a confusion matrix plot using `plt.savefig()` and `mlflow.log_artifact()`.
        *   Logged the trained scikit-learn model and TF-IDF vectorizer using `mlflow.sklearn.log_model()`. This saves them in MLflow's standard format.
    *   Ensured DVC outputs (models in the `models/` directory) are still saved for DVC pipeline tracking.

3.  **Pipeline Execution:**
    *   Ran `dvc repro`. DVC detected changes in `src/train.py` and re-executed the `train_model` stage.
    *   The modified `train.py` script executed, logging data to MLflow.
    *   `dvc.lock` was updated.

4.  **MLflow UI Exploration:**
    *   Started the MLflow UI: `mlflow ui`.
    *   Accessed `http://127.0.0.1:5000` in a browser.
    *   Viewed the logged experiment, run details, parameters, metrics, and artifacts (including the confusion matrix plot and the logged model/vectorizer).

5.  **Git Commits & DVC Push:**
    *   Committed changes to `src/train.py`, `requirements.txt`, `.gitignore`, and `dvc.lock`.
    *   Pushed Git commits to the remote repository.
    *   Ran `dvc push` to update DVC remote storage with the newly generated model artifacts (in `models/`).

**Outcome:** The project now uses MLflow for detailed experiment tracking. Each run of the DVC pipeline (triggered by `dvc repro`) that involves training will log its parameters, metrics, and key artifacts to MLflow, allowing for systematic comparison and management of experiments via the MLflow UI.

-----------

## Step 5: Creating a Simple Model Serving API with Flask

**Goal:** Develop a basic Flask API to serve the trained sentiment analysis model for real-time predictions.

**Key Actions & Files:**
1.  **Installation & Setup:**
    *   Installed `Flask` (`pip install Flask`).
    *   Added `Flask` to `requirements.txt`.

2.  **Flask Application (`src/app.py`):**
    *   Created a Flask application.
    *   **Model Loading:**
        *   Loads the `sentiment_model.joblib` and `tfidf_vectorizer.joblib` from the DVC-tracked `models/` directory when the app starts.
        *   Includes path handling to locate models relative to the project structure.
        *   Basic error handling for model loading.
    *   **Endpoints:**
        *   `/` (GET): A simple home endpoint to confirm the API is running.
        *   `/predict` (POST):
            *   Accepts a JSON payload: `{"text": "user review text"}`.
            *   Uses the imported `preprocess_text` function to clean the input.
            *   Vectorizes the text using the loaded TF-IDF vectorizer.
            *   Predicts sentiment using the loaded model.
            *   Returns a JSON response containing the input, processed text, predicted label (`positive`/`negative`), numeric prediction, and prediction probabilities.
            *   Includes input validation and error handling.
    *   The script can be run directly (`python src/app.py`) for development, starting a Flask debug server.

3.  **Model Availability:**
    *   Ensured that the model and vectorizer files (tracked by DVC and output by the `dvc repro` pipeline) are present in the `models/` directory. This might require `dvc pull` or `dvc repro` if they are missing.

4.  **API Testing:**
    *   Ran the Flask app locally: `python src/app.py`.
    *   Tested the `/predict` endpoint using `curl` with sample positive and negative reviews, and malformed requests.
    *   Verified that the API returns correct JSON responses with predictions and handles errors appropriately.

5.  **Git Commits:**
    *   Committed the new `src/app.py` and updated `requirements.txt` to Git.

**Outcome:** A functional Flask API is now available that can serve predictions from the trained sentiment analysis model. This API can be accessed locally and forms the basis for future deployment.

----------

## Step 6: Containerizing the Application with Docker

**Goal:** Package the Flask API application and its dependencies into a Docker container for portability and consistent deployment.

**Key Actions & Files:**
1.  **Docker Daemon Check:**
    *   Ensured the Docker daemon (via Docker Desktop on macOS/Windows or Docker Engine on Linux) was running before attempting Docker commands.

2.  **Model Availability:**
    *   Ensured DVC-tracked model files (`models/*.joblib`) were present in the local workspace (using `dvc pull` or `dvc repro`) before building the Docker image, as the `Dockerfile` copies them directly.

3.  **`Dockerfile` Creation:**
    *   Created a `Dockerfile` in the project root.
    *   **Base Image:** Started from `python:3.9-slim-buster`.
    *   **Dependencies:** Copied `requirements.txt` and ran `pip install --no-cache-dir -r requirements.txt` to install Python packages.
    *   **Application Code:** Copied the `src/` directory into the image.
    *   **Models:** Copied the `models/` directory (containing the `.joblib` files) into the image.
    *   **Port Exposure:** Exposed port `5001` (used by the Flask app).
    *   **Environment Variables:** Set `FLASK_ENV=production` and `FLASK_APP=src/app.py`.
    *   **Healthcheck:** Added a `HEALTHCHECK` instruction to verify if the app is responsive.
    *   **CMD:** Defined the command `CMD ["python", "src/app.py"]` to run the Flask application when the container starts.

4.  **`.dockerignore` File:**
    *   Created a `.dockerignore` file in the project root to exclude unnecessary files and directories (like `.git`, `.dvc`, `venv`, `mlruns`, `__pycache__`) from the Docker build context, optimizing build time and image size.

5.  **Docker Image Build:**
    *   Successfully built the Docker image using `docker build -t sentiment-analysis-api:latest .`.

6.  **Docker Container Run & Test:**
    *   Ran the built image as a container: `docker run -d -p 5001:5001 --name sentiment-app sentiment-analysis-api:latest`.
    *   Verified the container was running using `docker ps`.
    *   Tested the API endpoint (`/predict`) using `curl`, confirming it served predictions from within the container.

7.  **Container Management:**
    *   Viewed container logs using `docker logs sentiment-app` (if needed).
    *   Stopped and removed the container using `docker stop sentiment-app` and `docker rm sentiment-app`.

8.  **Git Commits:**
    *   Committed the new `Dockerfile` and `.dockerignore` to Git.

**Outcome:** The Flask application is now containerized. A Docker image can be built that packages the API, its dependencies, and the necessary model files. This image can be run consistently across different environments, which is a crucial step towards robust deployment.

--------

## Step 7: Continuous Integration (CI) with GitHub Actions (Initial Setup)

**Goal:** Establish a basic CI pipeline using GitHub Actions to automate checks when code is pushed or pull requests are made. This initial setup focuses on environment setup and attempting to run the DVC pipeline.

**Key Actions & Files:**
1.  **Workflow File Creation:**
    *   Created `.github/workflows/ci.yml` to define the CI workflow.

2.  **Workflow Definition (`ci.yml`):**
    *   **Name:** "MLOps CI Pipeline".
    *   **Triggers:** Runs on pushes and pull requests to the `main` branch, and allows manual dispatch.
    *   **Job (`build_and_test`):**
        *   Runs on `ubuntu-latest`.
        *   **Steps:**
            *   `Checkout repository`: Checks out the source code.
            *   `Set up Python`: Configures Python 3.9.
            *   `Install dependencies`: Installs packages from `requirements.txt`.
            *   `Set up DVC`: Placeholder step acknowledging the need for DVC remote configuration in CI. Due to the current local DVC remote, `dvc pull` is skipped.
            *   `Reproduce DVC pipeline`: Attempts `dvc repro`. *It's anticipated this step might fail in the CI environment because the actual `data/raw/reviews.csv` content is not accessible without a working `dvc pull` from a CI-accessible remote.*
            *   `Lint with Flake8 (Placeholder)`: Placeholder for future linting.
            *   `Test with Pytest (Placeholder)`: Placeholder for future tests.
    *   Comments included regarding the limitations of the current DVC setup in CI and how to address it with a cloud remote and secrets.

3.  **Git Commits:**
    *   Committed the new `.github/workflows/ci.yml` to Git.

4.  **Workflow Observation:**
    *   Pushed changes to GitHub, triggering the Actions workflow.
    *   Observed the workflow run in the "Actions" tab of the GitHub repository.
    *   Anticipated a potential failure in the `dvc repro` step due to the unavailability of the actual raw data file content on the CI runner (as the DVC remote is local to the development machine). This failure is educational, highlighting the need for a CI-accessible DVC remote for full pipeline reproducibility.

**Outcome:** An initial GitHub Actions CI workflow is in place. It automates environment setup and attempts to reproduce the DVC pipeline. This step highlights the challenges and requirements for integrating DVC with CI, particularly regarding remote data access. Future iterations will involve configuring a cloud-based DVC remote and adding more comprehensive testing and build steps (like Docker image building).

----------

## Step 8: Adding Linters and Basic Tests to CI

**Goal:** Enhance the CI pipeline by incorporating automated code linting with Flake8 and unit testing with Pytest to improve code quality and catch errors early.

**Key Actions & Files:**
1.  **Linting Setup (Flake8):**
    *   Added `flake8` to `requirements.txt`.
    *   Installed dependencies locally (`pip install -r requirements.txt`).
    *   Created a `.flake8` configuration file in the project root to customize linting rules (e.g., `max-line-length = 119`, ignored errors `E203, W503`).
    *   Ran `flake8 src/` locally to identify and fix all initial linting issues in `src/app.py`, `src/preprocess.py`, and `src/train.py`.
    *   Updated `.github/workflows/ci.yml`:
        *   The "Install dependencies" step now also installs `flake8`.
        *   Modified the "Lint with Flake8" step to execute `flake8 src/`. The CI job will fail if `flake8` reports any errors.

2.  **Testing Setup (Pytest):**
    *   Added `pytest` to `requirements.txt`.
    *   Installed dependencies locally.
    *   Created a test file `tests/test_preprocess.py`.
    *   Wrote unit tests for the `preprocess_text` function using Pytest's `@pytest.mark.parametrize` for multiple test cases, covering aspects like punctuation, case, stopwords, and empty strings.
    *   Ran `pytest tests/ -v` locally to ensure all tests pass.
    *   Updated `.github/workflows/ci.yml`:
        *   The "Install dependencies" step now also installs `pytest`.
        *   Modified the "Run Tests with Pytest" step to execute `pytest tests/ -v`. The CI job will fail if any tests fail.

3.  **CI Workflow Update:**
    *   The `Install dependencies` step in `ci.yml` now ensures `flake8` and `pytest` are available on the CI runner.
    *   The linting and testing steps are active and positioned after the DVC pipeline reproduction.

4.  **Git Commits:**
    *   Committed changes to `requirements.txt`, `tests/test_preprocess.py`, `.github/workflows/ci.yml`, `.flake8`, and source files (`src/*.py`) that were updated to fix linting issues.

5.  **Workflow Execution and Validation:**
    *   Pushed changes to GitHub, triggering the Actions workflow.
    *   Observed the workflow run in the "Actions" tab.
    *   Verified that the linting (`flake8 src/`) and testing (`pytest tests/ -v`) steps execute and pass in the CI environment.

**Outcome:** The CI pipeline is now more comprehensive. It not only checks for DVC pipeline reproducibility but also enforces code style consistency through linting and validates core functionality through automated unit tests. This leads to higher code quality, earlier bug detection, and a more reliable development process.

-------

## Step 9: Basic Continuous Deployment (CD) to Google Cloud Run

**Goal:** Automate the deployment of the containerized application to Google Cloud Run whenever changes are pushed to the `main` branch and the CI/CD pipeline succeeds.

**Key Actions Cloud Run**
```markdown
## Step 10: Basic Continuous Deployment (CD) to Google Cloud Run

**Goal:** Automate the deployment of the containerized application from Google Artifact Registry (AR) to Google Cloud Run whenever changes are pushed to the `main` branch and the CI/CD pipeline succeeds.

**Key Actions & Files:**
1.  **GCP Prerequisites & Configuration:**
    *   Ensured Cloud Run API and Artifact Registry API were enabled in the GCP project.
    *   Verified the deployment Service Account possessed `Cloud Run Admin`, `Service Account User`, and `Artifact Registry Writer` (or `Storage Admin` for GCR if used previously) IAM roles.
    *   (Recommended) Manually created the initial Cloud Run service (e.g., `sentiment-analysis-api` in `europe-west1`) with "Allow unauthenticated invocations" and Container port set to `5001`.

2.  **GitHub Secrets Configuration:**
    *   Verified/Created GitHub repository secrets: `GCP_SA_KEY`, `GCP_PROJECT_ID`, `CLOUD_RUN_SERVICE_NAME`, `CLOUD_RUN_REGION`, and `AR_REPO_NAME` (for the Artifact Registry repository name).

3.  **CI Workflow Update (`.github/workflows/ci.yml`):**
    *   The `build_docker_and_push_to_ar` job was configured to build the Docker image and push it to Google Artifact Registry, using regional AR paths (e.g., `europe-west1-docker.pkg.dev/YOUR_PROJECT_ID/YOUR_AR_REPO_NAME/sentiment-analysis-mlops:TAG`).
    *   Added a new job `deploy_to_cloud_run`.
    *   **Job Dependencies & Conditions:**
        *   `needs: build_docker_and_push_to_ar`: Runs only after the Docker image is successfully built and pushed to AR.
        *   `if: github.event_name == 'push' && github.ref == 'refs/heads/main'`: Deploys only on pushes to the `main` branch.
    *   **Steps within `deploy_to_cloud_run`:**
        *   `Authenticate to Google Cloud`: Uses `google-github-actions/auth@v2` with `credentials_json: '${{ secrets.GCP_SA_KEY }}'`.
        *   `Debug Deployment Parameters`: Added echo statements to verify the resolved values for service name, region, project ID, and image URL before deployment.
        *   `Deploy to Google Cloud Run`:
            *   Uses `google-github-actions/deploy-cloudrun@v2`.
            *   Configured with `service`, `region`, and `project_id` from GitHub & Files:**
1.  **GCP Prerequisites & Setup:**
    *   Enabled Cloud Run API, Container Registry API (or Artifact Registry API), and Cloud Build API in the GCP project.
    *   Created a dedicated Google Artifact Registry Docker repository (e.g., `mlops-images` in `europe-west1`).
    *   Ensured the deployment Service Account (from `GCP_SA_KEY` secret) has necessary IAM roles: `Artifact Registry Writer` (for pushing images), `Cloud Run Admin` (for deploying services), and `Service Account User`.
    *   (Recommended) Manually created the initial Cloud Run service via GCP Console (e.g., `sentiment-analysis-api` in `europe-west1`) with "Allow unauthenticated invocations" and Container port set to `5001`. This ensures the service construct exists before CI tries to update it.

2.  **GitHub Secrets Configuration:**
    *   Verified/Created GitHub repository secrets:
        *   `GCP_SA_KEY`: Service account JSON key content.
        *   `GCP_PROJECT_ID`: Google Cloud Project ID.
        *   `CLOUD_RUN_SERVICE_NAME`: Name of the Cloud Run service (e.g., `sentiment-analysis-api`).
        *   `CLOUD_RUN_REGION`: GCP region of the Cloud Run service (e.g., `europe-west1`).
        *   `AR_REPO_NAME`: Name of the Google Artifact Registry repository (e.g., `mlops-images`).

3.  **Updated CI Workflow (`.github/workflows/ci.yml`):**
    *   The `build_docker_and_push_to_ghcr` job was refactored to `build_docker_and_push_to_ar` to push the Docker image to Google Artifact Registry (AR) instead of GitHub Container Registry (GHCR). This involved:
        *   Authenticating Docker secrets/auth outputs.
            *   Specifies the `image` from Artifact Registry using the commit SHA for traceability.
            *   Uses `flags: '--allow-unauthenticated --port=5001'` to make the service public and specify the container port.
        *   `Print Cloud Run Service URL`: Outputs the URL of the deployed service.
    *   Resolved various YAML syntax, Docker tagging, and GCloud/DVC command issues through iterative debugging.

4.  **Git Commits:**
    *   Committed all changes to `.github/workflows/ci.yml`.

5.  **Workflow Execution and Deployment Verification:**
    *   Pushed changes to  to AR using `gcloud auth configure-docker`.
        *   Tagging the Docker image for the AR path`main`, triggering the full CI/CD pipeline.
    *   All jobs (`build_and_test`, `build_docker_and_push_to_ar`, `deploy_to_cloud_run`) completed successfully.
    *   Tested the root endpoint (`/`) of the deployed application via its Cloud Run URL, confirming it was live.
    *   Tested the `/predict` endpoint using `curl` (or Postman) with a `POST` request and JSON payload, verifying correct API functionality.

**Outcome:** A complete Continuous Integration and Continuous Deployment (CI/CD) pipeline is established. Changes merged to the `main` branch that pass all CI checks are automatically built into a Docker image, pushed to (e.g., `REGION-docker.pkg.dev/PROJECT_ID/AR_REPO_NAME/IMAGE_NAME Google Artifact Registry, and then deployed to Google Cloud Run, making the latest version of the application available at a public URL:TAG`).
    *   Added a new job `deploy_to_cloud_run`.
    *   **Job Dependencies & Conditions for `deploy_to_cloud_run`:**
        *   `needs: build_docker.

--------------

## Step 10: Basic Continuous Deployment (CD) to Google Cloud Run

**Goal:** Automate the deployment of the containerized application to Google Cloud Run whenever changes are pushed to the `main` branch and CI/CD pipeline succeeds.

**Key Actions & Files:**
1.  **GCP Prerequisites:**
    *   Enabled Cloud Run API, Artifact Registry API, and Cloud Build API in the GCP project.
    *   Ensured the deployment Service Account has `Cloud Run Admin` and `Service Account User` IAM roles.
    *   (Recommended) Manually created the initial Cloud Run service (e.g., `sentiment-analysis-api` in `europe-west1`) with "Allow unauthenticated invocations" and Container port set to `5001` (matching the Flask app).

2.  **GitHub Secrets Configuration:**
    *   Created/Verified GitHub repository secrets:
        *   `GCP_SA_KEY`: Service account JSON key content.
        *   `GCP_PROJECT_ID`: Google Cloud Project ID.
        *   `CLOUD_RUN_SERVICE_NAME`: Name of the Cloud Run service.
        *   `CLOUD_RUN_REGION`: GCP region of the Cloud Run service.

3.  **Updated Application (`src/app.py`) (Optional but Recommended):**
    *   Modified `app.run()` to use `port=int(os.environ.get("PORT", 5001))` to respect the `PORT` environment variable injected by Cloud Run, and set `debug=False`.

4.  **Updated CI Workflow (`.github/workflows/ci.yml`):**
    *   Added a new job `deploy_to_cloud_run`.
    *   **Job Dependencies & Conditions:**
        *   `needs: build_docker`: Runs only after the `build_docker` job is successful.
        *   `if: github.event_name == 'push' && github.ref == 'refs/heads/main'`: Deploys only on pushes to the `main` branch.
    *   **Steps within `deploy_to_cloud_run`:**
        *   `Checkout repository`.
        *   `Authenticate to Google Cloud`: Uses `google-github-actions/auth@v2` with `credentials_json: '${{ secrets.GCP_SA_KEY }}'`.
        *   `Deploy to Google Cloud Run`:
            *   Uses `google-github-actions/deploy-cloudrun@v2`.
            *   Configured with `service`, `region`, and `project_id` from GitHub secrets.
            *   Specifies the `image` from GHCR using the commit SHA for traceability (e.g., `ghcr.io/OWNER/IMAGE_NAME:${{ github.sha }}`).
            *   Set `allow_unauthenticated: true`.
        *   `Print Cloud Run Service URL`: Outputs the URL of the deployed service.

5.  **Git Commits:**
    *   Committed changes to `.github/workflows/ci.yml` and (if applicable) `src/app.py`.

6.  **Workflow Execution and Deployment Verification:**
    *   Pushed changes to `main`, triggering the full CI/CD pipeline.
    *   All jobs (`build_and_test`, `build_docker`, `deploy_to_cloud_run`) completed successfully.
    *   Tested the deployed application by accessing the Cloud Run service URL and sending requests to the `/predict` endpoint, verifying correct functionality.

**Outcome:** A basic Continuous Deployment pipeline is established. Changes merged to the `main` branch that pass all CI checks (linting, testing, DVC pipeline, Docker build & push) are automatically deployed to Google Cloud Run, making the latest version of the application available at a public URL.

------

## Step 11: Enhance Model Management with MLflow Model Registry

**Goal:** Centralize trained models using the MLflow Model Registry for versioning and stage management, and update the API to load models from the registry.

**Key Actions & Files:**
1.  **Local MLflow Tracking Server Setup:**
    *   Created directories `mlflow_server_data/backend_store` and `mlflow_server_data/artifact_store` for local server metadata and artifacts.
    *   Added `mlflow_server_data/` to `.gitignore`.
    *   Started a local MLflow server (e.g., on port 5002) using `mlflow server --backend-store-uri ./mlflow_server_data/backend_store --default-artifact-root ./mlflow_server_data/artifact_store`.

2.  **Modified Training Script (`src/train.py`):**
    *   Set a `REGISTERED_MODEL_NAME` (e.g., "SentimentAnalysisModelIMDB").
    *   Ensured MLflow logs to the server by setting the `MLFLOW_TRACKING_URI` environment variable (e.g., `http://localhost:5002`) before running training.
    *   Added logic to end any pre-existing active MLflow run before starting a new one to prevent conflicts.
    *   Used `mlflow.sklearn.log_model()` with the `registered_model_name` parameter to log the trained sentiment model and register its new version in the MLflow Model Registry.
    *   The TF-IDF vectorizer was also logged as an MLflow artifact/model (though the API currently still loads it from a DVC-tracked file for simplicity in this step).
    *   DVC outputs (`models/*.joblib` for both model and vectorizer) are still saved to the `models/` directory for DVC pipeline integrity and local API loading of the vectorizer.

3.  **Local Training and Model Registration:**
    *   Ran `dvc repro -v train_model` (with `MLFLOW_TRACKING_URI` set) to train, log to the MLflow server, and register the model.
    *   Updated `dvc.lock` as `src/train.py` changed.
    *   Accessed the MLflow UI (e.g., `http://localhost:5002`), navigated to the "Models" tab, and transitioned the newly registered model version (e.g., Version 1 of "SentimentAnalysisModelIMDB") to the "Staging" stage.

4.  **Modified API Application (`src/app.py`):**
    *   Configured to load the main sentiment model from the MLflow Model Registry at startup.
    *   Specified `REGISTERED_MODEL_NAME` and `MODEL_STAGE` (e.g., "Staging") to load.
    *   Uses `mlflow.pyfunc.load_model(f"models:/{REGISTERED_MODEL_NAME}/{MODEL_STAGE}")`.
    *   Requires `MLFLOW_TRACKING_URI` to be set in its environment to connect to the MLflow server.
    *   The TF-IDF vectorizer continues to be loaded from its DVC-tracked `.joblib` file located in the `models/` directory.
    *   Adjusted the prediction logic to work with the `mlflow.pyfunc` model wrapper (which typically expects a Pandas DataFrame input).

5.  **Local API Testing:**
    *   Ran the Flask API (`python src/app.py`) with `MLFLOW_TRACKING_URI` set.
    *   Tested the `/predict` endpoint using `curl`, confirming it loaded the main model from the registry and the vectorizer from the local DVC path, and served predictions successfully.

6.  **Git Commits:**
    *   Committed changes to `src/train.py`, `src/app.py`, `.gitignore`, and `dvc.lock`.

**Outcome:** The project now utilizes the MLflow Model Registry (via a local server setup) for managing the lifecycle of the main sentiment analysis model. The API is updated to load this model from the registry, while still using DVC for managing the vectorizer's artifact. This setup provides better model version control and governance for the main predictive model.

----------

## Step 12: Parameterizing DVC Pipeline with `params.yaml` & Basic Experimentation

**Goal:** Make the training pipeline configurable by externalizing hyperparameters into `params.yaml`, enabling systematic experimentation tracked by DVC and MLflow.

**Key Actions & Files:**
1.  **Created `params.yaml`:**
    *   A `params.yaml` file was created in the project root.
    *   Defined key hyperparameters and configuration values within this file, such as `data_split.test_split_ratio`, `featurization.tfidf_max_features`, and `training.logreg_C`.

2.  **Updated Python Dependencies:**
    *   Added `PyYAML` to `requirements.txt` to enable loading YAML files.
    *   Installed dependencies using `pip install -r requirements.txt`.

3.  **Modified Training Script (`src/train.py`):**
    *   The script now loads parameters from `params.yaml` at the beginning of the `train_model` function using `yaml.safe_load()`.
    *   Hardcoded values for parameters like TF-IDF max features, logistic regression solver, C value, and data split configurations were replaced with values read from the loaded `params` dictionary.
    *   The entire `params` dictionary is logged to MLflow using `mlflow.log_params(params)` for each run, ensuring all configurations are tracked.

4.  **Updated DVC Pipeline Definition (`dvc.yaml`):**
    *   The `train_model` stage in `dvc.yaml` was updated:
        *   `params.yaml` was added to the `deps:` list (to track any change in the file).
        *   A `params:` section was added, listing the specific dot-notation paths to the parameters within `params.yaml` that this stage depends on (e.g., `training.logreg_C`). This allows DVC to identify when a relevant parameter has changed, triggering a re-run of the stage.

5.  **Local Experimentation Workflow:**
    *   Ran `dvc repro -v train_model` locally. This executed the training with the initial parameters, logged them to MLflow, and updated `dvc.lock`.
    *   Modified a hyperparameter in `params.yaml` (e.g., changed `training.logreg_C`).
    *   Re-ran `dvc repro -v train_model`. DVC detected the change in the tracked parameter and re-executed the `train_model` stage.
    *   A new run appeared in the MLflow UI, with the updated parameter value logged, allowing for comparison between experiments.
    *   `dvc.lock` was updated again to reflect the outputs from this new run.

6.  **Git Commits:**
    *   Committed `params.yaml`, the updated `src/train.py`, `dvc.yaml`, `requirements.txt`, and the modified `dvc.lock`.

**Outcome:** The DVC pipeline is now parameterized, allowing for experiments to be driven by changes in `params.yaml`. DVC tracks these parameter dependencies, and MLflow logs the parameters used for each experiment run, facilitating systematic exploration and comparison of different model configurations.

--------

## Step 13: Basic Monitoring for Deployed API on Google Cloud Run

**Goal:** Understand and explore the built-in monitoring capabilities for the deployed Google Cloud Run service to gain insights into its operational health and performance.

**Key Actions & Activities:**
1.  **Generated Test Traffic (if necessary):**
    *   Sent sample `POST` requests to the `/predict` endpoint of the deployed Cloud Run API (and `GET` requests to the root `/`) to ensure metrics and logs would be populated for observation.

2.  **Explored Cloud Run Metrics in GCP Console:**
    *   Navigated to the specific Cloud Run service (e.g., `sentiment-analysis-api` in project `crested-grove-240711`, region `europe-west1`) within the Google Cloud Console.
    *   Viewed the **"METRICS"** tab for the service.
    *   Observed key built-in metrics provided by Google Cloud Monitoring, such as:
        *   Request Count
        *   Request Latency (e.g., 50th, 95th, 99th percentile)
        *   Request Count by Response Code (e.g., 2xx for success, 4xx for client errors, 5xx for server errors).
        *   Container Instance Count.
        *   Container CPU Utilization.
        *   Container Memory Utilization.
    *   Utilized different time range filters (e.g., last hour, last 6 hours) to analyze metric trends.

3.  **Explored Cloud Run Logs in GCP Console:**
    *   Viewed the **"LOGS"** tab for the Cloud Run service.
    *   Examined **Request Logs** generated by Cloud Run's frontend, showing HTTP method, path, status code, latency, etc., for each incoming request.
    *   Examined **Container Logs**, which include `stdout` and `stderr` from the Flask application running inside the Docker container. This includes custom log messages from `app.logger` (e.g., "Received text for prediction...") and any Python tracebacks if unhandled exceptions occurred within the application.

4.  **(Optional) Created a Basic Alerting Policy in Cloud Monitoring:**
    *   Navigated to Google Cloud Monitoring -> Alerting.
    *   Created a new alerting policy (e.g., to monitor the `Request Count` for the Cloud Run service, filtered by `response_code_class = 5xx`).
    *   Set a condition to trigger an alert if the count of server errors (5xx) exceeded a defined threshold within a specific time window (e.g., 5 errors in 1 minute).
    *   (Optionally) Configured a notification channel (e.g., email) for the alert.

**Outcome:**
Familiarity was gained with accessing and interpreting the standard operational metrics and logs automatically provided for services deployed on Google Cloud Run, via its integration with Cloud Monitoring and Cloud Logging. This provides essential visibility into the API's health, request patterns, error rates, and resource utilization. The process for setting up basic alerts for critical conditions (like high server error rates) was also understood. This forms the foundation for more advanced application performance monitoring (APM) and model-specific monitoring in the future.

----------

## Step 14: API Integration Tests with Pytest and Flask Test Client

**Goal:** Implement integration tests for the Flask API to ensure endpoints function correctly, handle valid and invalid inputs appropriately, and verify response integrity. These tests are integrated into the CI pipeline.

**Key Actions & Files:**
1.  **Updated Python Dependencies (`requirements.txt`):**
    *   Added `pytest-flask` to facilitate testing the Flask application.
    *   Added `requests` (though `pytest-flask`'s test client is primarily used for these tests).
    *   (Recommended) Standardized the `scikit-learn` version to avoid `InconsistentVersionWarning` and ensure model compatibility between training and serving/testing environments. This might have involved re-running `dvc repro -f train_model`, committing the updated `dvc.lock`, and running `dvc push`.

2.  **Created API Integration Test File (`tests/test_api.py`):**
    *   Imported the Flask application instance (`flask_app`) and relevant module-level variables (`app_model`, `app_vectorizer`) from `src.app`.
    *   Created Pytest fixtures:
        *   `app()`: Configures the `flask_app` for testing (`TESTING=True`). It also temporarily unsets the `MLFLOW_TRACKING_URI` environment variable during these API tests to ensure `app.py` uses its DVC fallback logic for loading the main model, making the API tests self-contained and independent of a live MLflow server.
        *   `client(app)`: Provides a Flask test client for making HTTP requests to the app without needing to run a separate server.
    *   Implemented several test functions for the API endpoints:
        *   `test_home_endpoint()`: Sends a `GET` request to the root (`/`) endpoint and verifies the status code and expected welcome message, also checking if the model and vectorizer (imported from `src.app`) were loaded.
        *   `test_predict_endpoint_positive_sentiment()`: Sends a `POST` request to `/predict` with positive review text, asserts a 200 status code, and checks for `"sentiment_label": "positive"` and correct numeric prediction (`1`) in the JSON response.
        *   `test_predict_endpoint_negative_sentiment()`: Similar test for negative sentiment, expecting `"sentiment_label": "negative"` and numeric prediction `0`.
        *   `test_predict_endpoint_missing_text_key()`: Tests error handling for a JSON payload missing the required `text` key; expects a 400 status code.
        *   `test_predict_endpoint_empty_text_value()`: Tests error handling for an empty or whitespace-only `text` value; expects a 400 status code.
        *   `test_predict_endpoint_malformed_json()`: Tests error handling for a malformed JSON payload; expects a 400 status code.

3.  **Modified API Application (`src/app.py` for Testability):**
    *   Updated the `predict` function to use `request.get_json(silent=True)`.
    *   Refined error handling to correctly return a 400 Bad Request status for malformed JSON or missing/invalid input keys, preventing these from being caught by the generic `except Exception` block and turned into 500 Internal Server Errors.

4.  **DVC Model Availability for Tests:**
    *   Ensured that DVC-tracked model files (`sentiment_model.joblib` and `tfidf_vectorizer.joblib`) are available in the `models/` directory for local testing (e.g., via `dvc pull` or having been generated by `dvc repro`).
    *   In the CI pipeline's `build_and_test` job, the `dvc repro` step generates these models, making them available when `pytest` runs. The `MLFLOW_TRACKING_URI` is unset by the test fixture, so `app.py`'s DVC fallback for model loading is used during these API tests.

5.  **Local and CI Test Execution:**
    *   Ran `pytest` locally to confirm all unit tests (from `test_preprocess.py`) and new API integration tests (from `test_api.py`) pass.
    *   The "Run Tests with Pytest" step in `.github/workflows/ci.yml` (which executes `pytest tests/ -v`) automatically discovers and runs these new API integration tests. The CI job now fails if any of these API tests fail.

6.  **Git Commits:**
    *   Committed all changes: updated `requirements.txt`, new `tests/test_api.py`, modified `src/app.py`, and (if applicable) updated `dvc.lock` and pushed new model versions via `dvc push`.

**Outcome:**
The project now includes a suite of automated API integration tests. These tests are executed as part of the CI pipeline, providing crucial validation that the Flask API endpoints are functioning as expected, including input validation, error handling, and correct prediction responses. This significantly increases confidence in the application's reliability before deployment.