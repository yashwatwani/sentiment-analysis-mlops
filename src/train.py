import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score,
                             classification_report,
                             confusion_matrix)
import joblib
import os
from preprocess import preprocess_text
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns

# Define DVC output paths (still needed for DVC pipeline)
RAW_DATA_PATH = "data/raw/reviews.csv"
MODEL_DIR = "models"

# --- MLflow Model Registry Configuration ---
# This is the name your model will be registered under in the MLflow Model Registry
REGISTERED_MODEL_NAME = "SentimentAnalysisModelIMDB"
# ------------------------------------------

# MLflow Experiment Setup (this will now log to your server if MLFLOW_TRACKING_URI is set)
EXPERIMENT_NAME = "SentimentAnalysisIMDB" 
# If MLFLOW_TRACKING_URI is set, mlflow.set_experiment will use it.
# Otherwise, it defaults to local 'mlruns'.
try:
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        mlflow.create_experiment(EXPERIMENT_NAME)
    mlflow.set_experiment(EXPERIMENT_NAME)
except Exception as e:
    print(f"MLflow experiment setup error (continuing with default behavior): {e}")


def plot_confusion_matrix_mlflow(
        cm, class_names, filename="confusion_matrix.png"):
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
    print("Starting model training with MLflow server logging and model registration...")
    # Ensure MLFLOW_TRACKING_URI is set in the environment running this script
    # e.g., export MLFLOW_TRACKING_URI=http://localhost:5002
    print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"MLflow Artifact URI: {mlflow.get_artifact_uri()}")

    # --- Ensure no previous run is active ---
    if mlflow.active_run():
        print(f"Warning: Found active MLflow run: {mlflow.active_run().info.run_id}. Ending it.")
        mlflow.end_run()
    # ---------------------------------------

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")
        mlflow.set_tag(
            "mlflow.runName",
            f"training_run_{pd.Timestamp.now():%Y%m%d_%H%M%S}"
        )

        test_split_ratio = 0.2
        random_seed_split = 42
        tfidf_max_features = 5000
        logreg_solver = 'liblinear'
        random_seed_model = 42

        mlflow.log_param("test_split_ratio", test_split_ratio)
        mlflow.log_param("random_seed_split", random_seed_split)
        mlflow.log_param("tfidf_max_features", tfidf_max_features)
        mlflow.log_param("logreg_solver", logreg_solver)
        mlflow.log_param("random_seed_model", random_seed_model)

        os.makedirs(MODEL_DIR, exist_ok=True)

        print(f"Loading data from {RAW_DATA_PATH}...")
        df = pd.read_csv(
            RAW_DATA_PATH, sep='\t', header=None,
            names=['review', 'sentiment']
        )
        print(f"Data loaded successfully. Shape: {df.shape}")

        print("Preprocessing text data...")
        df['review'] = df['review'].astype(str)
        df['processed_review'] = df['review'].apply(preprocess_text)
        print("Text preprocessing complete.")

        X = df['processed_review']
        y = df['sentiment']

        print("Splitting data into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_split_ratio,
            random_state=random_seed_split, stratify=y
        )
        mlflow.log_metric("train_set_size", len(X_train))
        mlflow.log_metric("test_set_size", len(X_test))

        print("Vectorizing text using TF-IDF...")
        vectorizer = TfidfVectorizer(max_features=tfidf_max_features)
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        print("Text vectorization complete.")

        print("Training Logistic Regression model...")
        model = LogisticRegression(
            solver=logreg_solver, random_state=random_seed_model
        )
        model.fit(X_train_tfidf, y_train)
        print("Model training complete.")

        print("Evaluating model...")
        y_pred = model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)
        print(f"Model Accuracy: {accuracy:.4f}")

        report = classification_report(y_test, y_pred, output_dict=True)
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

        # DVC output paths (still save for DVC pipeline)
        dvc_model_path = os.path.join(MODEL_DIR, "sentiment_model.joblib")
        dvc_vectorizer_path = os.path.join(
            MODEL_DIR, "tfidf_vectorizer.joblib"
        )
        joblib.dump(model, dvc_model_path)
        joblib.dump(vectorizer, dvc_vectorizer_path)
        print("Model and vectorizer saved for DVC.")

        # --- Log and Register Model with MLflow Model Registry ---
        print("Logging model to MLflow Tracking Server...")
        # Log the scikit-learn model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="sentiment-model-sklearn", # Subdirectory in MLflow run's artifact store
            registered_model_name=REGISTERED_MODEL_NAME # Register under this name
        )
        print(f"Model registered in MLflow Model Registry as '{REGISTERED_MODEL_NAME}'.")

        # Log the vectorizer as a separate artifact (or model if complex)
        # For simplicity, we can log it as a generic artifact, or another sklearn model
        mlflow.sklearn.log_model(
            sk_model=vectorizer,
            artifact_path="tfidf-vectorizer-sklearn", # Subdirectory for vectorizer
            # Optionally register vectorizer if you want to version it via registry too
            # registered_model_name="SentimentVectorizerIMDB" 
        )
        print("Vectorizer logged as an artifact/model to MLflow.")
        # ---------------------------------------------------------

        print("Training process finished. MLflow run logged to server.")
        mlflow.set_tag("status", "completed")

if __name__ == '__main__':
    # For direct execution, ensure MLFLOW_TRACKING_URI is set, e.g.:
    # export MLFLOW_TRACKING_URI='http://localhost:5002'
    # python src/train.py
    if not os.getenv("MLFLOW_TRACKING_URI"):
        print("Warning: MLFLOW_TRACKING_URI is not set. MLflow will use local 'mlruns' directory.")
    train_model()