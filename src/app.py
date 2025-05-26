from flask import Flask, request, jsonify
# import joblib # No longer loading directly with joblib from DVC path for primary model
import os
from preprocess import preprocess_text
import traceback
import mlflow

# --- Configuration ---
# MLflow Model Registry configuration
# Ensure MLFLOW_TRACKING_URI is set in the environment where this app runs
# e.g., export MLFLOW_TRACKING_URI='http://localhost:5002'
REGISTERED_MODEL_NAME = "SentimentAnalysisModelIMDB"
MODEL_STAGE = "Staging"  # Or "Production" - the stage to load

# DVC path for vectorizer (we'll still load this from DVC for now for simplicity)
# Alternatively, you can log/register and load vectorizer from MLflow too.
MODEL_DIR_DVC = "models" # DVC tracked models directory
VECTORIZER_PATH_DVC = os.path.join(MODEL_DIR_DVC, "tfidf_vectorizer.joblib")


# --- Initialize Flask App ---
app = Flask(__name__)

# --- Load Model and Vectorizer ---
model = None
vectorizer = None # Will be loaded from DVC path for this iteration

def load_model_from_registry():
    global model
    model_uri = f"models:/{REGISTERED_MODEL_NAME}/{MODEL_STAGE}"
    app.logger.info(f"Attempting to load model from MLflow Registry: {model_uri}")
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        app.logger.info(f"Model '{REGISTERED_MODEL_NAME}' version for stage '{MODEL_STAGE}' loaded successfully from MLflow Registry.")
    except Exception as e:
        app.logger.error(f"Failed to load model from MLflow Registry: {model_uri}")
        app.logger.error(f"Error: {e}")
        app.logger.error(traceback.format_exc())
        model = None # Ensure model is None if loading fails

def load_vectorizer_from_dvc():
    global vectorizer
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(base_dir)
        actual_vectorizer_path = os.path.join(project_root, VECTORIZER_PATH_DVC)

        if not os.path.exists(actual_vectorizer_path):
            app.logger.error(
                f"Vectorizer file not found at {actual_vectorizer_path}. "
                f"Ensure 'dvc repro' or 'dvc pull {VECTORIZER_PATH_DVC}' has been run."
            )
            return # vectorizer remains None

        app.logger.info(f"Loading vectorizer from DVC path: {actual_vectorizer_path}")
        # Need to import joblib here as it's locally used
        import joblib 
        vectorizer = joblib.load(actual_vectorizer_path)
        app.logger.info("Vectorizer loaded successfully from DVC path.")

    except Exception as e:
        app.logger.error(f"An unexpected error occurred during vectorizer loading: {e}")
        app.logger.error(traceback.format_exc())
        vectorizer = None # Ensure vectorizer is None

# Load models at startup
if not os.getenv("MLFLOW_TRACKING_URI"):
    app.logger.warning(
        "MLFLOW_TRACKING_URI is not set. MLflow will use local 'mlruns' "
        "and may not find the registered model from the server."
    )
else:
    app.logger.info(f"Using MLflow Tracking URI: {os.getenv('MLFLOW_TRACKING_URI')}")
    
load_model_from_registry()
load_vectorizer_from_dvc() # Continue loading vectorizer from DVC path


@app.route('/')
def home():
    return "Sentiment Analysis API is running! Use the /predict endpoint. Model loaded from Registry."


@app.route('/predict', methods=['POST'])
def predict():
    # No 'global model, vectorizer' needed as we access module-level variables

    if model is None: # Check only model, vectorizer check remains as is or can be combined
        return jsonify({"error": "Sentiment model not loaded from MLflow Registry. Check server logs."}), 500
    if vectorizer is None:
        return jsonify({"error": "Vectorizer not loaded from DVC path. Check server logs."}), 500
    
    # ... (rest of the predict function remains the same as before) ...
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "Invalid input. JSON with 'text' key required."}), 400

        review_text = data['text']
        if not isinstance(review_text, str) or not review_text.strip():
            return jsonify({"error": "'text' must be a non-empty string."}), 400

        app.logger.info(f"Received text for prediction: '{review_text}'")
        processed_text = preprocess_text(review_text)
        app.logger.info(f"Processed text: '{processed_text}'")
        vectorized_text = vectorizer.transform([processed_text])
        
        # If using mlflow.pyfunc.load_model, the 'model' is a PyFuncModel wrapper.
        # It has a predict method that takes a pandas DataFrame.
        input_df = pd.DataFrame([processed_text], columns=['text']) # Or appropriate column name
        prediction_result = model.predict(input_df)

        # Assuming the pyfunc model directly returns the class (0 or 1) or a structure
        # If it returns a DataFrame with a prediction column:
        # prediction_numeric = prediction_result['prediction_column_name'].iloc[0]
        # If it directly returns the prediction (e.g., for scikit-learn models loaded via pyfunc):
        if isinstance(prediction_result, pd.DataFrame) and not prediction_result.empty:
             # Assuming the prediction is in the first column if it's a DataFrame
            prediction_numeric = prediction_result.iloc[0,0]
        elif isinstance(prediction_result, (list, pd.Series)) and len(prediction_result) > 0:
            prediction_numeric = prediction_result[0]
        else: # Fallback if unsure about pyfunc output structure for this model type
            app.logger.warning(f"Unexpected prediction result type or empty: {type(prediction_result)}")
            # Try to get probabilities if available from the underlying model if pyfunc doesn't give them directly
            # This part might need adjustment based on how your sklearn model is wrapped by pyfunc
            # For direct sklearn, we used model.predict_proba and model.predict
            # For now, we'll assume prediction_numeric is obtained correctly.
            # We might lose direct access to predict_proba unless the pyfunc model exposes it
            # or if we load the raw sklearn model from the logged artifact instead of pyfunc.
            # For simplicity, let's just use the prediction from pyfunc.
            # We would need to re-think probability access if strictly needed from pyfunc model.
            # For now, let's create dummy probabilities if we can't get them directly.
            prediction_numeric = int(prediction_numeric) if not isinstance(prediction_numeric, int) else prediction_numeric
            dummy_proba_positive = 0.9 if prediction_numeric == 1 else 0.1
            prediction_proba = [[1-dummy_proba_positive, dummy_proba_positive]]


        sentiment_label = "positive" if prediction_numeric == 1 else "negative"
        
        app.logger.info(
            f"Prediction from Registry Model: {sentiment_label}"
            # f", Probabilities: {prediction_proba[0]}" # May not have direct proba from pyfunc
        )

        response = {
            "input_text": review_text,
            "processed_text": processed_text,
            "sentiment_label": sentiment_label,
            "prediction_numeric": int(prediction_numeric),
            # "probabilities": { # Commenting out direct probabilities if pyfunc doesn't provide them easily
            #     "negative": float(prediction_proba[0][0]),
            #     "positive": float(prediction_proba[0][1])
            # }
        }
        return jsonify(response), 200

    except Exception as e:
        app.logger.error(f"Error during prediction: {e}")
        app.logger.error(traceback.format_exc())
        return jsonify({"error": "An error occurred during prediction.", "details": str(e)}), 500


if __name__ == '__main__':
    # For direct execution, ensure MLFLOW_TRACKING_URI is set, e.g.:
    # export MLFLOW_TRACKING_URI='http://localhost:5002'
    # And vectorizer from DVC is available
    # python src/app.py
    if model is None or vectorizer is None:
        print(
            "CRITICAL: Model from Registry or Vectorizer from DVC failed to load. "
            "The API will not work correctly."
        )
    else:
        print("Model from Registry and Vectorizer from DVC loaded. Starting Flask server...")
        port = int(os.environ.get("PORT", 5001))
        app.run(host='0.0.0.0', port=port, debug=False) # Set debug=False