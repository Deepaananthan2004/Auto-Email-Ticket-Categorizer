# =====================================================
# AI Ticket Categorizer
# Model Training Script
# Part 1
# =====================================================

import os
import joblib
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
)

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from utils import (
    clean_text,
    combine_text,
)

# =====================================================
# Dataset
# =====================================================

DATA_PATH = "data/support_tickets.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found : {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("DATASET LOADED")
print("=" * 70)

print(df.head())

# =====================================================
# Column Validation
# =====================================================

df.columns = df.columns.str.lower().str.strip()

required_columns = [
    "subject",
    "body",
    "category",
]

for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Missing column : {column}"
        )

print("\nColumns Verified")

# =====================================================
# Remove Missing Values
# =====================================================

before = len(df)

df.dropna(
    subset=[
        "subject",
        "body",
        "category",
    ],
    inplace=True,
)

after = len(df)

print(f"\nRemoved Missing Rows : {before-after}")

# =====================================================
# Remove Duplicate Emails
# =====================================================

before = len(df)

df.drop_duplicates(inplace=True)

after = len(df)

print(f"Removed Duplicate Rows : {before-after}")

# =====================================================
# Shuffle Dataset
# =====================================================

df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

print("\nDataset Shuffled")

# =====================================================
# Combine Subject + Body
# =====================================================

df["text"] = df.apply(

    lambda row: combine_text(

        row["subject"],
        row["body"]

    ),

    axis=1,

)

# =====================================================
# Clean Text
# =====================================================

df["clean_text"] = df["text"].apply(
    clean_text
)

print("\nSample Cleaned Data\n")

print(

    df[
        [
            "text",
            "clean_text",
            "category",
        ]
    ].head()

)

# =====================================================
# Features & Labels
# =====================================================

X = df["clean_text"]

y = df["category"]

print("\nCategory Distribution\n")

print(y.value_counts())

# =====================================================
# Train Test Split
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y,

)

print("\nTraining Samples :", len(X_train))

print("Testing Samples  :", len(X_test))

# =====================================================
# Model Pipeline
# =====================================================

model = Pipeline(

    [

        (

            "tfidf",

            TfidfVectorizer(

                max_features=10000,

                ngram_range=(1, 2),

                min_df=2,

                sublinear_tf=True,

            ),

        ),

        (

            ("classifier", LogisticRegression(
                max_iter=1000,
                random_state=42
            ))

        ),

    ]

)

print("\nPipeline Created Successfully")

# =====================================================
# Train Model
# =====================================================

print("\n" + "=" * 70)
print("TRAINING MODEL")
print("=" * 70)

model.fit(X_train, y_train)

print("\nModel Training Completed Successfully!")

# =====================================================
# Cross Validation
# =====================================================

print("\nRunning 5-Fold Cross Validation...")

cv_scores = cross_val_score(

    model,

    X,

    y,

    cv=5,

    scoring="accuracy",

)

print("\nCross Validation Scores")

for index, score in enumerate(cv_scores):

    print(f"Fold {index+1} : {score:.2%}")

print(f"\nAverage CV Accuracy : {cv_scores.mean():.2%}")

# =====================================================
# Prediction
# =====================================================

predictions = model.predict(X_test)

probabilities = model.predict_proba(X_test)

# =====================================================
# Evaluation
# =====================================================

accuracy = accuracy_score(

    y_test,

    predictions,

)

print("\n" + "=" * 70)
print("MODEL EVALUATION")
print("=" * 70)

print(f"\nAccuracy : {accuracy:.2%}")

print("\nClassification Report\n")

print(

    classification_report(

        y_test,

        predictions,

    )

)

print("\nConfusion Matrix\n")

cm = confusion_matrix(

    y_test,

    predictions,

)

labels = model.classes_

cm_df = pd.DataFrame(

    cm,

    index=labels,

    columns=labels,

)

print(cm_df)

# =====================================================
# Save Model
# =====================================================

os.makedirs(

    "models",

    exist_ok=True,

)

MODEL_PATH = "models/ticket_classifier.pkl"

joblib.dump(

    model,

    MODEL_PATH,

)

joblib.dump(

    model.classes_,

    "models/classes.pkl",

)

print("\nModel Saved Successfully")

print(MODEL_PATH)

# =====================================================
# Save Model Information
# =====================================================

with open(

    "models/model_info.txt",

    "w",

) as file:

    file.write(

        f"Accuracy : {accuracy:.2%}\n"

    )

    file.write(

        f"Cross Validation Accuracy : {cv_scores.mean():.2%}\n"

    )

    file.write(

        f"Training Samples : {len(X_train)}\n"

    )

    file.write(

        f"Testing Samples : {len(X_test)}\n"

    )

print("\nModel Information Saved")

# =====================================================
# Sample Predictions
# =====================================================

print("\n" + "=" * 70)
print("SAMPLE PREDICTIONS")
print("=" * 70)

sample_tickets = [

    "Payment deducted twice from my bank account. Please refund immediately.",

    "Unable to login after resetting my password.",

    "Need leave approval for next Friday.",

    "Can you share your office timings?",

    "URGENT! Production server is down and customers cannot access the website.",

    "Application crashes while uploading PDF files.",

    "Please send my salary slip for June.",

    "My order was cancelled but money has not been refunded.",

]

for ticket in sample_tickets:

    cleaned = clean_text(ticket)

    prediction = model.predict(

        [cleaned]

    )[0]

    probabilities = model.predict_proba(

        [cleaned]

    )[0]

    confidence = probabilities.max()

    print("\n" + "-" * 70)

    print("Ticket")

    print(ticket)

    print("\nPrediction")

    print(prediction)

    print(f"\nConfidence : {confidence:.2%}")

    print("\nProbability Distribution")

    for label, score in sorted(

        zip(

            model.classes_,

            probabilities,

        ),

        key=lambda x: x[1],

        reverse=True,

    ):

        print(

            f"{label:<15} {score:.2%}"

        )

    if confidence < 0.60:

        print("\nStatus : Needs Human Review")

    else:

        print("\nStatus : Auto Classified")

print("\n" + "=" * 70)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 70)

print(f"\nFinal Accuracy : {accuracy:.2%}")
print(f"Cross Validation Accuracy : {cv_scores.mean():.2%}")

print("\nSaved Files")

print("-----------------------------")

print("✓ models/ticket_classifier.pkl")
print("✓ models/classes.pkl")
print("✓ models/model_info.txt")

print("\nYour AI Ticket Classifier is Ready!")