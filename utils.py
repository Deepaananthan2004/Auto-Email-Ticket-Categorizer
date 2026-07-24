# utils.py

import re
import nltk
from nltk.corpus import stopwords

# Download stopwords (only runs the first time)
nltk.download("stopwords", quiet=True)

# Load English stopwords
stop_words = set(stopwords.words("english"))


def clean_text(text):
    """
    Clean and preprocess text for NLP classification.

    Steps:
    1. Convert to lowercase
    2. Remove URLs
    3. Remove email addresses
    4. Remove numbers
    5. Remove punctuation
    6. Remove extra spaces
    7. Remove stopwords
    """

    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", "", text)

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove punctuation
    text = re.sub(r"[^a-z\s]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Remove stopwords
    words = text.split()
    words = [word for word in words if word not in stop_words]

    return " ".join(words)


# Priority keywords
URGENT_KEYWORDS = [
    "urgent",
    "critical",
    "asap",
    "immediately",
    "down",
    "failed",
    "failure",
    "not working",
    "crash",
    "crashed",
    "error",
    "issue",
    "blocked",
    "cannot",
    "can't",
    "unable",
    "broken",
    "offline"
]


def get_priority(text):
    """
    Returns:
        Urgent
        Normal
    """

    text = text.lower()

    for keyword in URGENT_KEYWORDS:
        if keyword in text:
            return "Urgent"

    return "Normal"


def combine_text(subject, body):
    """
    Combine subject and body into one string.
    """

    subject = "" if subject is None else str(subject)
    body = "" if body is None else str(body)

    return subject + " " + body