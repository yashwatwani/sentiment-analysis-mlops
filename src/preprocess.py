import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords if not already present
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')


def preprocess_text(text):
    """
    Cleans and preprocesses a single text string.
    - Lowercases
    - Removes punctuation and numbers
    - Removes stopwords
    - Applies stemming
    """
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.strip()

    stop_words_set = set(stopwords.words('english'))
    words = text.split()
    words = [w for w in words if w not in stop_words_set]

    stemmer = PorterStemmer()
    words = [stemmer.stem(w) for w in words]

    return " ".join(words)


if __name__ == '__main__':
    # Example usage
    sample_text_1 = (
        "This is a sample sentence, with 123 numbers and punctuation!"
    )
    print(f"Original: {sample_text_1}")
    print(f"Processed: {preprocess_text(sample_text_1)}")

    sample_text_2 = "I loved the movie, it was fantastic and great fun!!!"
    print(f"Original: {sample_text_2}")
    print(f"Processed: {preprocess_text(sample_text_2)}")
