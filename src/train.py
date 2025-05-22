import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib # For saving the model
import os
from preprocess import preprocess_text # Import from our preprocess.py

# Define paths
RAW_DATA_PATH = "data/raw/reviews.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "sentiment_model.joblib")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")

def train_model():
    """
    Trains a sentiment analysis model.
    - Loads data
    - Preprocesses text
    - Splits data
    - Vectorizes text using TF-IDF
    - Trains a Logistic Regression model
    - Evaluates the model
    - Saves the model and vectorizer
    """
    print("Starting model training...")

    # Create models directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. Load Data
    print(f"Loading data from {RAW_DATA_PATH}...")
    try:
        # The data is tab-separated, with no header
        df = pd.read_csv(RAW_DATA_PATH, sep='\t', header=None, names=['review', 'sentiment'])
    except FileNotFoundError:
        print(f"Error: Raw data file not found at {RAW_DATA_PATH}.")
        print("Please ensure you have downloaded the data and placed it in data/raw/reviews.csv")
        return

    print(f"Data loaded successfully. Shape: {df.shape}")
    # print(df.head()) # Optional: inspect data

    # 2. Preprocess Text
    print("Preprocessing text data...")
    # Ensure 'review' column is string type
    df['review'] = df['review'].astype(str)
    df['processed_review'] = df['review'].apply(preprocess_text)
    print("Text preprocessing complete.")
    # print(df[['review', 'processed_review']].head()) # Optional: inspect processed text

    # 3. Prepare features (X) and target (y)
    X = df['processed_review']
    y = df['sentiment']

    # 4. Split Data
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")

    # 5. Vectorize Text (TF-IDF)
    print("Vectorizing text using TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000) # Limit features to 5000
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print("Text vectorization complete.")

    # 6. Train Model
    print("Training Logistic Regression model...")
    model = LogisticRegression(solver='liblinear', random_state=42) # liblinear is good for smaller datasets
    model.fit(X_train_tfidf, y_train)
    print("Model training complete.")

    # 7. Evaluate Model
    print("Evaluating model...")
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # 8. Save Model and Vectorizer
    print(f"Saving model to {MODEL_PATH}...")
    joblib.dump(model, MODEL_PATH)
    print(f"Saving TF-IDF vectorizer to {VECTORIZER_PATH}...")
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print("Model and vectorizer saved successfully.")

    print("Training process finished.")

if __name__ == '__main__':
    train_model()