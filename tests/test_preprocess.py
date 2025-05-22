# tests/test_preprocess.py
import pytest
from src.preprocess import preprocess_text # Adjust if your src structure is different

# Test cases: (input_string, expected_output_string)
preprocess_test_cases = [
    ("This is a Sample Sentence, with 123 numbers and punctuation!", "sampl sentenc number punctuat"),
    ("I LOVED the movie!! It was great fun.", "love movi great fun"),
    ("  leading and trailing spaces   ", "lead trail space"),
    ("NothING To cHaNGE", "noth chang"),
    ("", ""), 
    ("12345", "")
]

@pytest.mark.parametrize("input_text, expected_output", preprocess_test_cases)
def test_preprocess_text_parametrized(input_text, expected_output):
    """Test the preprocess_text function with various inputs."""
    assert preprocess_text(input_text) == expected_output

def test_preprocess_text_stopwords():
    """Test specifically for stopword removal."""
    text = "this is a test of the stopwords function"
    expected = "test stopword function" 
    assert preprocess_text(text) == expected

def test_preprocess_text_punctuation_and_numbers():
    """Test removal of punctuation and numbers."""
    text = "Hello World! 123... This is it. #tag"
    expected = "hello world tag" 
    assert preprocess_text(text) == expected

# Ensure there's a newline at the end of this file