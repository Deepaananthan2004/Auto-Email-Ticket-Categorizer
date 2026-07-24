# train.py

import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from utils import clean_text, combine_text


# ===========================
# Load Dataset
# ===========================

DATA_PATH = "data/support_tickets.csv"

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("Dataset Loaded Successfully")
print("=" * 60)
print(df.head())


# ===========================
# Data Preprocessing
# ===========================

# Combine Subject + Body
df["text"] = df.apply(
    lambda row: combine_text(row["subject"], row["body"]),
    axis=1,
)

# Clean Text
df["clean_text"] = df["text"].apply(clean_text)

print("\nSample Cleaned Text:\n")
print(df[["text", "clean_text"]].head())


# ===========================
# Features & Labels
# ===========================

X = df["clean_text"]
y = df["category"]


# ===========================
# Train-Test Split
# ===========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))


# ===========================
# Build Pipeline
# ===========================

model = Pipeline(
    [
        (
            "tfidf",
            TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
            ),
        ),
        (
            "classifier",
            MultinomialNB(),
        ),
    ]
)


# ===========================
# Train Model
# ===========================

print("\nTraining Model...\n")

model.fit(X_train, y_train)

print("Model Training Completed")


# ===========================
# Predictions
# ===========================

predictions = model.predict(X_test)


# ===========================
# Evaluation
# ===========================

accuracy = accuracy_score(y_test, predictions)

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print(f"\nAccuracy : {accuracy:.2%}")

print("\nClassification Report\n")

print(classification_report(y_test, predictions))

print("\nConfusion Matrix\n")

cm = confusion_matrix(y_test, predictions)

labels = model.classes_

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels,
)

print(cm_df)


# ===========================
# Save Model
# ===========================

os.makedirs("models", exist_ok=True)

MODEL_PATH = "models/ticket_classifier.pkl"

joblib.dump(model, MODEL_PATH)

print("\nModel Saved Successfully")

print(MODEL_PATH)


# ===========================
# Test Predictions
# ===========================

print("\n" + "=" * 60)
print("Sample Predictions")
print("=" * 60)

sample_tickets = [

    "Payment deducted twice from my bank account.",

    "My laptop crashes whenever I start the application.",

    "I need leave approval for next Monday.",

    "Can you share your office timings?",

    "URGENT! Production server is down.",

]

for ticket in sample_tickets:

    cleaned = clean_text(ticket)

    prediction = model.predict([cleaned])[0]

    confidence = model.predict_proba([cleaned]).max()

    print("\nTicket")
    print("-" * 50)
    print(ticket)

    print("Predicted Category :", prediction)
    print(f"Confidence         : {confidence:.2%}")


print("\nTraining Completed Successfully!")