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