import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
from preprocess import preprocess_text
import mlflow
import mlflow.sklearn # For scikit-learn specific logging
import matplotlib.pyplot as plt # For plotting
import seaborn as sns # For plotting
import numpy as np # For metrics extraction

# Define paths (MLflow will also store artifacts, but good to have defined paths for DVC outputs)
RAW_DATA_PATH = "data/raw/reviews.csv"
MODEL_DIR = "models" # DVC output directory
# MODEL_PATH = os.path.join(MODEL_DIR, "sentiment_model.joblib") # DVC output
# VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib") # DVC output
# Let MLflow manage artifact paths within its run structure for now,
# DVC will still track the 'models' directory as an output of the dvc stage.
# We can also explicitly log DVC tracked files as MLflow artifacts if needed.

# --- MLflow Experiment Setup ---
# Best practice: set experiment name. If it doesn't exist, MLflow creates it.
# You can also set this via an environment variable MLFLOW_EXPERIMENT_NAME
EXPERIMENT_NAME = "SentimentAnalysisIMDB"
mlflow.set_experiment(EXPERIMENT_NAME)
# -------------------------------

def plot_confusion_matrix_mlflow(cm, class_names, filename="confusion_matrix.png"):
    """Helper function to plot and log confusion matrix to MLflow"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    # Save plot locally before logging
    plot_path = filename 
    plt.savefig(plot_path)
    plt.close() # Close the plot to free memory
    
    # Log plot to MLflow
    mlflow.log_artifact(plot_path, "plots")
    print(f"Confusion matrix plot saved to {plot_path} and logged to MLflow.")
    os.remove(plot_path) # Clean up local plot file after logging

def train_model():
    """
    Trains a sentiment analysis model and logs with MLflow.
    """
    print("Starting model training with MLflow logging...")

    # Use mlflow.start_run() to create a new MLflow run
    # This is a context manager, so it handles start and end of run
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")
        mlflow.set_tag("mlflow.runName", f"training_run_{pd.Timestamp.now():%Y%m%d_%H%M%S}")


        # --- Parameters for logging ---
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
        # -----------------------------

        os.makedirs(MODEL_DIR, exist_ok=True) # Ensure DVC output dir exists

        print(f"Loading data from {RAW_DATA_PATH}...")
        try:
            df = pd.read_csv(RAW_DATA_PATH, sep='\t', header=None, names=['review', 'sentiment'])
        except FileNotFoundError:
            print(f"Error: Raw data file not found at {RAW_DATA_PATH}.")
            return
        print(f"Data loaded successfully. Shape: {df.shape}")

        print("Preprocessing text data...")
        df['review'] = df['review'].astype(str)
        df['processed_review'] = df['review'].apply(preprocess_text)
        print("Text preprocessing complete.")

        X = df['processed_review']
        y = df['sentiment']

        print("Splitting data into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_split_ratio, random_state=random_seed_split, stratify=y
        )
        print(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")
        mlflow.log_metric("train_set_size", len(X_train))
        mlflow.log_metric("test_set_size", len(X_test))

        print("Vectorizing text using TF-IDF...")
        vectorizer = TfidfVectorizer(max_features=tfidf_max_features)
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        print("Text vectorization complete.")

        print("Training Logistic Regression model...")
        model = LogisticRegression(solver=logreg_solver, random_state=random_seed_model)
        model.fit(X_train_tfidf, y_train)
        print("Model training complete.")

        print("Evaluating model...")
        y_pred = model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log main metrics
        mlflow.log_metric("accuracy", accuracy)
        print(f"Model Accuracy: {accuracy:.4f}")

        # Detailed classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        print("\nClassification Report (raw dict):\n", report)

        # Log precision, recall, f1-score for each class and macro/weighted averages
        for class_or_avg, metrics_dict in report.items():
            if isinstance(metrics_dict, dict): # Handles individual classes and averages
                for metric_name, metric_value in metrics_dict.items():
                    # Sanitize metric name for MLflow (e.g. "f1-score" -> "f1_score")
                    safe_metric_name = metric_name.replace("-", "_")
                    mlflow.log_metric(f"{class_or_avg}_{safe_metric_name}", metric_value)
            elif class_or_avg == "accuracy": # Already logged
                pass 
            else: # Handles single values like support if not dict
                 mlflow.log_metric(f"{class_or_avg}", metrics_dict)


        print("\nClassification Report (formatted):\n", classification_report(y_test, y_pred))

        # Log confusion matrix plot
        cm = confusion_matrix(y_test, y_pred)
        class_names = ['negative', 'positive'] # Assuming 0 is negative, 1 is positive
        plot_confusion_matrix_mlflow(cm, class_names, filename="confusion_matrix.png")

        # --- Saving and Logging Model & Vectorizer ---
        # DVC output paths (still needed for DVC to track stage outputs)
        dvc_model_path = os.path.join(MODEL_DIR, "sentiment_model.joblib")
        dvc_vectorizer_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")

        print(f"Saving model for DVC to {dvc_model_path}...")
        joblib.dump(model, dvc_model_path)
        print(f"Saving TF-IDF vectorizer for DVC to {dvc_vectorizer_path}...")
        joblib.dump(vectorizer, dvc_vectorizer_path)
        print("Model and vectorizer saved for DVC.")

        # Log model and vectorizer with MLflow
        # Option 1: Log as generic artifacts (what we saved for DVC)
        # mlflow.log_artifact(dvc_model_path, artifact_path="dvc_outputs")
        # mlflow.log_artifact(dvc_vectorizer_path, artifact_path="dvc_outputs")
        
        # Option 2: Use MLflow's scikit-learn auto-logging or specific model flavor
        # This is often preferred as it stores models in MLflow's format,
        # making them easier to load later with mlflow.pyfunc.load_model(), etc.
        # It can also automatically log parameters and metrics if enabled.
        # For this example, we'll explicitly log the scikit-learn model.
        # The 'artifact_path' is a sub-directory within the MLflow run's artifact store.
        print("Logging model to MLflow...")
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="sklearn-model", # This will create a folder 'sklearn-model' in MLflow artifacts
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE # or 'pickle'
        )
        print("Logging vectorizer to MLflow...")
        mlflow.sklearn.log_model( # Can use log_model for any scikit-learn object
            sk_model=vectorizer,
            artifact_path="tfidf-vectorizer",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
        )
        # ---------------------------------------------

        print("Training process finished. MLflow run logged.")
        mlflow.set_tag("status", "completed")

if __name__ == '__main__':
    train_model()