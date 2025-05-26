name: MLOps CI Pipeline

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ] # Deploy job will be skipped for PRs
  workflow_dispatch:

jobs:
  build_and_test:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Set up DVC and Pull Data from GCS
        # For this job, train.py will run. If MLFLOW_TRACKING_URI is not set here,
        # train.py will use local mlruns or error if it expects a server and can't connect.
        # The main goal here is that dvc repro completes and DVC outputs are produced.
        env:
          MLFLOW_TRACKING_URI: "" # Explicitly make it use local mlruns for this CI training run
                                  # or a dummy value if train.py handles connection errors gracefully.
                                  # This prevents CI from trying to connect to a non-existent prod server.
        run: |
          echo "Authenticating to Google Cloud (for DVC raw data pull)..."
          echo '${{ secrets.GCP_SA_KEY }}' > gcp_creds_ci.json
          export GOOGLE_APPLICATION_CREDENTIALS=./gcp_creds_ci.json
          echo "DVC setup: Using GCS remote."
          echo "Pulling DVC tracked data (reviews.csv) from GCS..."
          dvc pull data/raw/reviews.csv -v
          echo "DVC data pull attempt finished."
          ls -lh data/raw/
          rm -f gcp_creds_ci.json

      - name: Reproduce DVC pipeline (generates models for DVC)
        env:
          MLFLOW_TRACKING_URI: "" # As above, ensure train.py handles this gracefully
                                  # or logs to a temporary local mlruns for this CI step.
        run: |
          echo "Attempting to reproduce DVC pipeline..."
          dvc repro -v
          echo "DVC pipeline reproduction attempt finished."
          echo "Listing models directory after DVC repro (for DVC tracking):"
          ls -lh models/

      - name: Lint with Flake8
        run: |
          echo "Running Flake8 linter on src/ directory..."
          flake8 src/

      - name: Run Tests with Pytest
        run: |
          echo "Running Pytest..."
          pytest tests/ -v

  build_docker_and_push_to_ar:
    name: Build and Push Docker Image to Artifact Registry
    needs: build_and_test
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Authenticate to Google Cloud (for DVC model pull & AR Push)
        id: auth 
        uses: 'google-github-actions/auth@v2'
        with:
          credentials_json: '${{ secrets.GCP_SA_KEY }}'

      - name: Set up Python for Docker Build Job
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'

      - name: Set up DVC and Pull Models for Docker Build (for fallback and vectorizer)
        env:
          # DVC needs GOOGLE_APPLICATION_CREDENTIALS to pull from GCS remote
          GOOGLE_APPLICATION_CREDENTIALS: ./gcp_creds_for_dvc.json
        run: |
          echo '${{ secrets.GCP_SA_KEY }}' > gcp_creds_for_dvc.json
          
          python -m pip install --upgrade pip
          pip install -r requirements.txt 
          echo "Finished installing dependencies from requirements.txt."
          
          python -c "import dvc_gs; print('dvc_gs imported successfully!')" || (pip list && exit 1)

          echo "Pulling DVC-tracked models (sentiment_model.joblib for fallback, tfidf_vectorizer.joblib) for Docker image..."
          mkdir -p models 
          # Pull both model (for fallback) and vectorizer
          dvc pull models/sentiment_model.joblib models/tfidf_vectorizer.joblib -v
          echo "Models pulled for Docker build."
          ls -lh models/
          if [ ! -s models/sentiment_model.joblib ] || [ ! -s models/tfidf_vectorizer.joblib ]; then
            echo "ERROR: Model/vectorizer files not pulled correctly by DVC for Docker build."
            exit 1
          fi
          rm -f gcp_creds_for_dvc.json

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Configure Docker to use Google Cloud CLI credential helper for Artifact Registry
        run: gcloud auth configure-docker ${{ secrets.CLOUD_RUN_REGION }}-docker.pkg.dev --quiet

      - name: Get current date
        id: date
        run: echo "DATE=$(date --iso-8601=seconds)" >> $GITHUB_OUTPUT

      - name: Build and push Docker image to Artifact Registry
        id: docker_build
        uses: docker/build-push-action@v5
        with:
          context: .
          file: ./Dockerfile
          push: true
          tags: | 
            ${{ secrets.CLOUD_RUN_REGION }}-docker.pkg.dev/${{ steps.auth.outputs.project_id }}/${{ secrets.AR_REPO_NAME }}/sentiment-analysis-mlops:latest
            ${{ secrets.CLOUD_RUN_REGION }}-docker.pkg.dev/${{ steps.auth.outputs.project_id }}/${{ secrets.AR_REPO_NAME }}/sentiment-analysis-mlops:${{ github.sha }}
          labels: |
            org.opencontainers.image.source=${{ github.server_url }}/${{ github.repository }}
            org.opencontainers.image.revision=${{ github.sha }}
            org.opencontainers.image.created=${{ steps.date.outputs.DATE }}

      - name: Image Digest
        run: |
          echo "Pushed image with digest: ${{ steps.docker_build.outputs.digest }}"

  deploy_to_cloud_run:
    name: Deploy to Google Cloud Run
    needs: build_docker_and_push_to_ar
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'

    steps:
      - name: Authenticate to Google Cloud
        id: auth 
        uses: 'google-github-actions/auth@v2'
        with:
          credentials_json: '${{ secrets.GCP_SA_KEY }}'

      - name: Debug Deployment Parameters
        run: |
          echo "Service Name: ${{ secrets.CLOUD_RUN_SERVICE_NAME || 'sentiment-analysis-api' }}"
          echo "Region: ${{ secrets.CLOUD_RUN_REGION || 'europe-west1' }}"
          echo "Project ID (from auth): ${{ steps.auth.outputs.project_id }}"
          echo "AR Repo Name: ${{ secrets.AR_REPO_NAME }}"
          echo "Image to deploy: ${{ secrets.CLOUD_RUN_REGION }}-docker.pkg.dev/${{ steps.auth.outputs.project_id }}/${{ secrets.AR_REPO_NAME }}/sentiment-analysis-mlops:${{ github.sha }}"

      - name: Deploy to Google Cloud Run
        id: deploy_cloud_run
        uses: 'google-github-actions/deploy-cloudrun@v2'
        with:
          service: ${{ secrets.CLOUD_RUN_SERVICE_NAME || 'sentiment-analysis-api' }} 
          region: ${{ secrets.CLOUD_RUN_REGION || 'europe-west1' }}     
          project_id: ${{ steps.auth.outputs.project_id }} 
          image: ${{ secrets.CLOUD_RUN_REGION }}-docker.pkg.dev/${{ steps.auth.outputs.project_id }}/${{ secrets.AR_REPO_NAME }}/sentiment-analysis-mlops:${{ github.sha }}
          flags: '--allow-unauthenticated --port=5001'
          # Pass MLFLOW_TRACKING_URI to Cloud Run service if you have a PROD MLflow server
          # If MLFLOW_SERVER_URI_PROD secret is not set or empty, app.py will use DVC fallback.
          env_vars: |
            MLFLOW_TRACKING_URI=${{ secrets.MLFLOW_SERVER_URI_PROD || '' }}

      - name: Print Cloud Run Service URL
        if: steps.deploy_cloud_run.outputs.url 
        run: echo "Service deployed to: ${{ steps.deploy_cloud_run.outputs.url }}"

# Ensure a final newline character below this line