import pytest
from src.app import app as flask_app # Import your Flask app instance
from src.app import model as app_model # Import the module-level 'model'
from src.app import vectorizer as app_vectorizer # Import the module-level 'vectorizer'
import json
import os

# Configure the Flask app for testing
# This ensures that exceptions are propagated rather than handled by the app's error handlers
# and that the app is in testing mode.
@pytest.fixture(scope='module')
def app():
    """Instance of Flask app for testing"""
    # Set the testing config. This is important for Flask internals.
    flask_app.config.update({
        "TESTING": True,
    })

    # Critical: Ensure models are loaded for tests.
    # The app loads models at startup. If this fixture runs before app startup,
    # we might need to ensure the loading logic in app.py is robust or
    # explicitly trigger it here if it's deferred.
    # For now, assuming app.py's global model/vectorizer loading works
    # when this test module is imported and flask_app is initialized.
    # We also need MLFLOW_TRACKING_URI to be unset or pointing to a test/dummy server
    # if we don't want tests to depend on a live MLflow server for model loading.
    # For simplicity, let's assume fallback to DVC models for testing.
    
    # If MLFLOW_TRACKING_URI is set in the test environment, app will try to use it.
    # To force DVC fallback for API tests (making them more self-contained):
    if "MLFLOW_TRACKING_URI" in os.environ:
        original_uri = os.environ.pop("MLFLOW_TRACKING_URI")
        print(f"Temporarily unsetting MLFLOW_TRACKING_URI for API tests to use DVC fallback.")
        yield flask_app
        os.environ["MLFLOW_TRACKING_URI"] = original_uri # Restore it
    else:
        yield flask_app


@pytest.fixture(scope='module')
def client(app):
    """Test client for the Flask app."""
    return app.test_client()


def test_home_endpoint(client):
    """Test the '/' home endpoint."""
    response = client.get('/')
    assert response.status_code == 200
    assert b"Sentiment Analysis API is running!" in response.data
    # Now check the imported module-level variables
    if app_model and app_vectorizer: 
         assert b"Model and Vectorizer loaded." in response.data
    elif app_model:
         assert b"Model loaded, Vectorizer FAILED." in response.data
    elif app_vectorizer:
         assert b"Vectorizer loaded, Model FAILED." in response.data
    else:
         assert b"CRITICAL: Model and Vectorizer FAILED to load." in response.data


def test_predict_endpoint_positive_sentiment(client):
    """Test the '/predict' endpoint with a positive sentiment review."""
    payload = {"text": "This movie was absolutely fantastic and a joy to watch!"}
    response = client.post('/predict', 
                           data=json.dumps(payload), 
                           content_type='application/json')
    
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data is not None
    assert "sentiment_label" in json_data
    assert json_data["sentiment_label"] == "positive"
    assert "prediction_numeric" in json_data
    assert json_data["prediction_numeric"] == 1
    assert "input_text" in json_data
    assert json_data["input_text"] == payload["text"]
    assert "processed_text" in json_data


def test_predict_endpoint_negative_sentiment(client):
    """Test the '/predict' endpoint with a negative sentiment review."""
    payload = {"text": "A truly awful and boring film, I hated every minute."}
    response = client.post('/predict',
                           data=json.dumps(payload),
                           content_type='application/json')
    
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data is not None
    assert json_data["sentiment_label"] == "negative"
    assert json_data["prediction_numeric"] == 0


def test_predict_endpoint_missing_text_key(client):
    """Test '/predict' with missing 'text' key in JSON payload."""
    payload = {"wrong_key": "some value"}
    response = client.post('/predict',
                           data=json.dumps(payload),
                           content_type='application/json')
    assert response.status_code == 400
    json_data = response.get_json()
    assert "error" in json_data
    assert "JSON with 'text' key required" in json_data["error"]


def test_predict_endpoint_empty_text_value(client):
    """Test '/predict' with an empty string for 'text'."""
    payload = {"text": "  "} # Empty or whitespace only
    response = client.post('/predict',
                           data=json.dumps(payload),
                           content_type='application/json')
    assert response.status_code == 400
    json_data = response.get_json()
    assert "error" in json_data
    assert "'text' must be a non-empty string" in json_data["error"]

def test_predict_endpoint_malformed_json(client):
    malformed_payload = "{'text': 'this is not proper JSON}" 
    response = client.post('/predict',
                           data=malformed_payload,
                           content_type='application/json')
    assert response.status_code == 400 # Expecting 400
