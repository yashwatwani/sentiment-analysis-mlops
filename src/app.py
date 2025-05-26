from flask import Flask, request, jsonify
import os
from preprocess import preprocess_text
import traceback
import mlflow
import joblib # Ensure joblib is imported for fallback
import pandas as pd # Ensure pandas is imported for pyfunc input

# --- Configuration ---
MLFLOW_TRACKING_URI_ENV = os.getenv("MLFLOW_TRACKING_URI") # Get from environment
REGISTERED_MODEL_NAME = "SentimentAnalysisModelIMDB"
MODEL_STAGE = "Staging"

MODEL_DIR_DVC = "models" 
SENTIMENT_MODEL_PATH_DVC = os.path.join(MODEL_DIR_DVC, "sentiment_model.joblib")
VECTORIZER_PATH_DVC = os.path.join(MODEL_DIR_DVC, "tfidf_vectorizer.joblib")

# --- Initialize Flask App ---
app = Flask(__name__)

# --- Load Model and Vectorizer ---
model = None
vectorizer = None

def get_absolute_path(relative_path):
    base_dir = os.path.dirname(os.path.abspath(__file__)) # src/
    project_root = os.path.dirname(base_dir) # project root
    return os.path.join(project_root, relative_path)

def load_models_at_startup():
    global model, vectorizer

    # 1. Attempt to load main model from MLflow Model Registry
    if MLFLOW_TRACKING_URI_ENV:
        app.logger.info(f"Using MLflow Tracking URI: {MLFLOW_TRACKING_URI_ENV}")
        model_uri = f"models:/{REGISTERED_MODEL_NAME}/{MODEL_STAGE}"
        app.logger.info(f"Attempting to load model from MLflow Registry: {model_uri}")
        try:
            # Set tracking URI for this loading operation if not already globally set by env var
            # for the current Python process (though env var is better)
            # mlflow.set_tracking_uri(MLFLOW_TRACKING_URI_ENV) # Usually not needed if env var is picked up
            model = mlflow.pyfunc.load_model(model_uri)
            app.logger.info(
                f"Model '{REGISTERED_MODEL_NAME}' version for stage '{MODEL_STAGE}' "
                "loaded successfully from MLflow Registry."
            )
        except Exception as e:
            app.logger.warning(
                f"Failed to load model from MLflow Registry (URI: {model_uri}). Error: {e}"
            )
            model = None # Ensure model is None if registry load fails
    else:
        app.logger.warning(
            "MLFLOW_TRACKING_URI not set. Skipping load from MLflow Registry."
        )

    # 2. Fallback: If model not loaded from registry, load from DVC path
    if model is None:
        actual_model_path_dvc = get_absolute_path(SENTIMENT_MODEL_PATH_DVC)
        app.logger.info(
            f"Attempting to load model from DVC path (fallback): {actual_model_path_dvc}"
        )
        try:
            if not os.path.exists(actual_model_path_dvc):
                app.logger.error(
                    f"DVC model file not found at {actual_model_path_dvc}."
                )
            else:
                model = joblib.load(actual_model_path_dvc)
                app.logger.info(
                    "Model loaded successfully from DVC path (fallback)."
                )
        except Exception as e:
            app.logger.error(
                f"Failed to load model from DVC path. Error: {e}"
            )
            app.logger.error(traceback.format_exc())
            model = None # Ensure model is None if DVC load also fails

    # 3. Load vectorizer from DVC path (as before)
    actual_vectorizer_path_dvc = get_absolute_path(VECTORIZER_PATH_DVC)
    app.logger.info(
        f"Loading vectorizer from DVC path: {actual_vectorizer_path_dvc}"
    )
    try:
        if not os.path.exists(actual_vectorizer_path_dvc):
            app.logger.error(
                f"Vectorizer file not found at {actual_vectorizer_path_dvc}."
            )
        else:
            vectorizer = joblib.load(actual_vectorizer_path_dvc)
            app.logger.info("Vectorizer loaded successfully from DVC path.")
    except Exception as e:
        app.logger.error(
            f"Failed to load vectorizer from DVC path. Error: {e}"
        )
        app.logger.error(traceback.format_exc())
        vectorizer = None

# Load models when app starts
load_models_at_startup()


@app.route('/')
def home():
    if model and vectorizer:
        return "Sentiment Analysis API is running! Model and Vectorizer loaded. Use /predict."
    elif model:
        return "Sentiment Analysis API is running! Model loaded, Vectorizer FAILED. API may not work."
    elif vectorizer:
        return "Sentiment Analysis API is running! Vectorizer loaded, Model FAILED. API may not work."
    else:
        return "Sentiment Analysis API is running! CRITICAL: Model and Vectorizer FAILED to load."


@app.route('/predict', methods=['POST'])
def predict():
    if model is None or vectorizer is None:
        return jsonify({
            "error": "Model or vectorizer not loaded correctly. Check server logs."
        }), 500
    
    try:
        data = request.get_json()
        # ... (input validation for data and data['text'] remains the same)
        if not data or 'text' not in data:
            return jsonify({"error": "Invalid input. JSON with 'text' key required."}), 400

        review_text = data['text']
        if not isinstance(review_text, str) or not review_text.strip():
            return jsonify({"error": "'text' must be a non-empty string."}), 400

        app.logger.info(f"Received text for prediction: '{review_text}'")
        processed_text = preprocess_text(review_text)
        app.logger.info(f"Processed text: '{processed_text}'")
        vectorized_text = vectorizer.transform([processed_text])
        
        prediction_numeric = None
        # Check if the loaded model is an MLflow pyfunc model or a raw scikit-learn model
        if hasattr(model, 'predict') and callable(getattr(model, 'predict')) and hasattr(model, '_model_impl'): # Heuristic for MLflow pyfunc
            app.logger.info("Predicting using MLflow pyfunc model.")
            input_df = pd.DataFrame([processed_text], columns=['text']) # Assuming 'text' is the expected input column
            prediction_result = model.predict(input_df)
            
            if isinstance(prediction_result, pd.DataFrame) and not prediction_result.empty:
                prediction_numeric = prediction_result.iloc[0,0]
            elif isinstance(prediction_result, (list, pd.Series)) and len(prediction_result) > 0:
                prediction_numeric = prediction_result[0]
            else:
                prediction_numeric = int(prediction_result) if not isinstance(prediction_result, int) else prediction_result
            prediction_numeric = int(prediction_numeric)

        elif hasattr(model, 'predict'): # Assume it's a scikit-learn like model (from DVC fallback)
            app.logger.info("Predicting using DVC/joblib loaded scikit-learn model.")
            prediction_numeric = model.predict(vectorized_text)[0]
            # prediction_proba = model.predict_proba(vectorized_text) # If you want probabilities for this path
        else:
            app.logger.error("Loaded model does not have a recognized predict method.")
            return jsonify({"error": "Model loaded but cannot perform prediction."}), 500

        sentiment_label = "positive" if prediction_numeric == 1 else "negative"
        app.logger.info(f"Prediction: {sentiment_label}")

        response = {
            "input_text": review_text,
            "processed_text": processed_text,
            "sentiment_label": sentiment_label,
            "prediction_numeric": int(prediction_numeric)
        }
        return jsonify(response), 200

    except Exception as e:
        app.logger.error(f"Error during prediction: {e}")
        app.logger.error(traceback.format_exc())
        return jsonify({"error": "An error occurred during prediction.", "details": str(e)}), 500


if __name__ == '__main__':
    if model is None or vectorizer is None:
        print("CRITICAL: Model or Vectorizer failed to load. API will not function correctly.")
    else:
        print("Model and Vectorizer loaded. Starting Flask server...")
    port = int(os.environ.get("PORT", 5001))
    app.run(host='0.0.0.0', port=port, debug=False)