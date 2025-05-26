"train the model"
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score,
                             classification_report,
                             confusion_matrix)
import joblib
import os
from .preprocess import preprocess_text
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import yaml  # For loading params.yaml

# Define DVC output paths
RAW_DATA_PATH = "data/raw/reviews.csv"
MODEL_DIR = "models"
PARAMS_FILE = "params.yaml"  # Define path to params file

# MLflow Model Registry Configuration
REGISTERED_MODEL_NAME = "SentimentAnalysisModelIMDB"

# MLflow Experiment Setup
EXPERIMENT_NAME = "SentimentAnalysisIMDB"
try:
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        mlflow.create_experiment(EXPERIMENT_NAME)
    mlflow.set_experiment(EXPERIMENT_NAME)
except Exception as e:
    print(
        f"MLflow experiment setup error (continuing with default behavior): {e}"
    )


def plot_confusion_matrix_mlflow(
        cm, class_names, filename="confusion_matrix.png"):
    # ... (this function remains the same as before) ...
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plot_path = filename
    plt.savefig(plot_path)
    plt.close()
    mlflow.log_artifact(plot_path, "plots")
    print(
        f"Confusion matrix plot saved to {plot_path} and logged to MLflow."
    )
    os.remove(plot_path)


def train_model():
    print(
        "Starting model training with MLflow server logging, "
        "model registration, and params from params.yaml..."
    )
    print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"MLflow Artifact URI: {mlflow.get_artifact_uri()}")

    # --- Load parameters from params.yaml ---
    try:
        with open(PARAMS_FILE, 'r') as f:
            params = yaml.safe_load(f)
        print(f"Loaded parameters from {PARAMS_FILE}: {params}")
    except FileNotFoundError:
        print(f"ERROR: {PARAMS_FILE} not found. Using default script parameters.")
        # Define default params here if file not found, or exit
        params = {  # Fallback default parameters
            'data_split': {'test_split_ratio': 0.2, 'random_seed_split': 42},
            'featurization': {'tfidf_max_features': 5000},
            'training': {'logreg_solver': 'liblinear', 'logreg_C': 1.0, 'random_seed_model': 42}
        }
    except Exception as e:
        print(f"Error loading {PARAMS_FILE}: {e}. Using default script parameters.")
        # Define default params here or exit
        params = {  # Fallback default parameters
            'data_split': {'test_split_ratio': 0.2, 'random_seed_split': 42},
            'featurization': {'tfidf_max_features': 5000},
            'training': {'logreg_solver': 'liblinear', 'logreg_C': 1.0, 'random_seed_model': 42}
        }
    # -----------------------------------------

    if mlflow.active_run():
        active_run_id = mlflow.active_run().info.run_id
        print(f"Warning: Found active MLflow run: {active_run_id}. Ending it.")
        mlflow.end_run()

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")
        mlflow.set_tag(
            "mlflow.runName",
            f"training_run_{pd.Timestamp.now():%Y%m%d_%H%M%S}"
        )

        # --- Use parameters from loaded params ---
        test_split_ratio = params['data_split']['test_split_ratio']
        random_seed_split = params['data_split']['random_seed_split']
        tfidf_max_features = params['featurization']['tfidf_max_features']
        logreg_solver = params['training']['logreg_solver']
        logreg_C = params['training']['logreg_C']  # New parameter
        random_seed_model = params['training']['random_seed_model']

        # Log all parameters (params is a nested dict, MLflow can log it)
        mlflow.log_params(params)
        # Or log them individually if you prefer more control over naming in UI
        # mlflow.log_param("test_split_ratio", test_split_ratio)
        # ... (etc. for all params) ...
        print(f"Using parameters: C={logreg_C}, max_features={tfidf_max_features}")
        # -----------------------------------------

        os.makedirs(MODEL_DIR, exist_ok=True)
        df = pd.read_csv(
            RAW_DATA_PATH, sep='\t', header=None,
            names=['review', 'sentiment']
        )
        df['review'] = df['review'].astype(str)
        df['processed_review'] = df['review'].apply(preprocess_text)

        X = df['processed_review']
        y = df['sentiment']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_split_ratio,
            random_state=random_seed_split, stratify=y
        )
        mlflow.log_metric("train_set_size", len(X_train))
        mlflow.log_metric("test_set_size", len(X_test))

        vectorizer = TfidfVectorizer(max_features=tfidf_max_features)
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        # Use logreg_C in model initialization
        model = LogisticRegression(
            C=logreg_C,  # Using the parameter
            solver=logreg_solver,
            random_state=random_seed_model
        )
        model.fit(X_train_tfidf, y_train)
        print("Model training complete.")

        y_pred = model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)
        print(f"Model Accuracy: {accuracy:.4f}")

        report = classification_report(y_test, y_pred, output_dict=True)
        # ... (metric logging for report remains the same) ...
        for class_or_avg, metrics_dict in report.items():
            if isinstance(metrics_dict, dict):
                for metric_name, metric_value in metrics_dict.items():
                    safe_metric_name = metric_name.replace("-", "_")
                    mlflow.log_metric(
                        f"{class_or_avg}_{safe_metric_name}", metric_value
                    )
            elif class_or_avg == "accuracy":
                pass
            else:
                mlflow.log_metric(f"{class_or_avg}", metrics_dict)
        print(
            "\nClassification Report (formatted):\n",
            classification_report(y_test, y_pred)
        )

        cm = confusion_matrix(y_test, y_pred)
        class_names = ['negative', 'positive']
        plot_confusion_matrix_mlflow(
            cm, class_names, filename="confusion_matrix.png"
        )

        dvc_model_path = os.path.join(MODEL_DIR, "sentiment_model.joblib")
        dvc_vectorizer_path = os.path.join(
            MODEL_DIR, "tfidf_vectorizer.joblib"
        )
        joblib.dump(model, dvc_model_path)
        joblib.dump(vectorizer, dvc_vectorizer_path)
        print("Model and vectorizer saved for DVC.")

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="sentiment-model-sklearn",
            registered_model_name=REGISTERED_MODEL_NAME
        )
        print(
          f"Model registered in MLflow Model Registry as '{REGISTERED_MODEL_NAME}'."
        )
        mlflow.sklearn.log_model(
            sk_model=vectorizer,
            artifact_path="tfidf-vectorizer-sklearn",
        )
        print("Vectorizer logged as an artifact/model to MLflow.")
        print("Training process finished. MLflow run logged to server.")
        mlflow.set_tag("status", "completed")


if __name__ == '__main__':
    if not os.getenv("MLFLOW_TRACKING_URI"):
        print(
            "Warning: MLFLOW_TRACKING_URI is not set. "
            "MLflow will use local 'mlruns' directory."
        )
    train_model()
