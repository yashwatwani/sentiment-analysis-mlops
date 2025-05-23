from flask import Flask, request, jsonify
import joblib
import os
# Assuming preprocess.py is in the same src directory
from preprocess import preprocess_text
import traceback  # For detailed error logging

# --- Configuration ---
# Path to the DVC-tracked model and vectorizer
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "sentiment_model.joblib")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")

# --- Initialize Flask App ---
app = Flask(__name__)

# --- Load Model and Vectorizer ---
# Load them once when the application starts
model = None
vectorizer = None

try:
    # Gets directory of app.py (src/)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)  # Gets project root

    actual_model_path = os.path.join(project_root, MODEL_PATH)
    actual_vectorizer_path = os.path.join(project_root, VECTORIZER_PATH)

    app.logger.info(f"Loading model from: {actual_model_path}")
    app.logger.info(f"Loading vectorizer from: {actual_vectorizer_path}")

    if not os.path.exists(actual_model_path):
        app.logger.error(
            f"Model file not found at {actual_model_path}. "
            f"Ensure 'dvc pull' or 'dvc repro' has been run."
        )
    if not os.path.exists(actual_vectorizer_path):
        app.logger.error(
            f"Vectorizer file not found at {actual_vectorizer_path}. "
            f"Ensure 'dvc pull' or 'dvc repro' has been run."
        )

    model = joblib.load(actual_model_path)
    vectorizer = joblib.load(actual_vectorizer_path)
    app.logger.info("Model and vectorizer loaded successfully.")

except FileNotFoundError as e:
    app.logger.error(f"Error loading model/vectorizer: {e}")
    app.logger.error(
        "Please ensure the model and vectorizer files exist. "
        "You might need to run 'dvc pull' or 'dvc repro'."
    )
except Exception as e:
    app.logger.error(
        f"An unexpected error occurred during model loading: {e}"
    )
    app.logger.error(traceback.format_exc())


# --- API Endpoints ---
@app.route('/')
def home():
    return "Sentiment Analysis API is running! Use the /predict endpoint."


@app.route('/predict', methods=['POST'])
def predict():
    # global model, vectorizer  # Access the globally loaded model/vectorizer

    if model is None or vectorizer is None:
        return jsonify({
            "error": "Model or vectorizer not loaded. Check server logs."
        }), 500

    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({
                "error": "Invalid input. JSON with 'text' key required."
            }), 400

        review_text = data['text']
        if not isinstance(review_text, str) or not review_text.strip():
            return jsonify({
                "error": "'text' must be a non-empty string."
            }), 400

        app.logger.info(f"Received text for prediction: '{review_text}'")

        # 1. Preprocess the input text
        processed_text = preprocess_text(review_text)
        app.logger.info(f"Processed text: '{processed_text}'")

        # 2. Vectorize the processed text
        vectorized_text = vectorizer.transform([processed_text])

        # 3. Make a prediction
        # Get probabilities
        prediction_proba = model.predict_proba(vectorized_text)
        prediction_numeric = model.predict(vectorized_text)[0]  # Get 0 or 1

        # Assuming 1 is 'positive' and 0 is 'negative'
        sentiment_label = "positive" if prediction_numeric == 1 else "negative"

        app.logger.info(
            f"Prediction: {sentiment_label}, Probabilities: {prediction_proba.tolist()}"
        )

        response = {
            "input_text": review_text,
            "processed_text": processed_text,
            "sentiment_label": sentiment_label,
            "prediction_numeric": int(prediction_numeric),
            "probabilities": {
                "negative": float(prediction_proba[0][0]),
                "positive": float(prediction_proba[0][1])
            }
        }
        return jsonify(response), 200

    except Exception as e:
        app.logger.error(f"Error during prediction: {e}")
        app.logger.error(traceback.format_exc())
        return jsonify({
            "error": "An error occurred during prediction.",
            "details": str(e)
        }), 500


# To run the app directly (for development)
if __name__ == '__main__':
    if model is None or vectorizer is None:
        print(
            "CRITICAL: Model or vectorizer failed to load. "
            "The API will not work correctly."
        )
        print(
            "Ensure model files are present at the specified paths "
            "(e.g., run 'dvc pull' or 'dvc repro')."
        )
    else:
        print("Model and vectorizer loaded. Starting Flask development server...")
        # For Cloud Run, it's better to respect the PORT environment variable
        port = int(os.environ.get("PORT", 5001))
        app.run(host='0.0.0.0', port=port, debug=False)  # Set debug=False for production/Cloud Run
