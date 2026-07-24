# Auto Email / Ticket Categorizer

## Overview

The **Auto Email / Ticket Categorizer** is a lightweight Natural Language Processing (NLP) application that automatically classifies incoming support tickets into the appropriate department.

The project uses **TF-IDF Vectorization** and a **Machine Learning classifier (Naive Bayes)** to categorize tickets into:

* Billing
* Technical
* HR
* General

It also predicts a confidence score, assigns a priority level, and flags low-confidence tickets for manual review.

---

# Features

* NLP-based support ticket classification
* TF-IDF text vectorization
* Naive Bayes classifier
* Automatic ticket routing
* Confidence score
* Human review threshold
* Priority tagging (Urgent / Normal)
* Interactive web interface using Flask
* Clean and modular project structure

---

# Tech Stack

* Python 3.12
* Flask
* Scikit-learn
* Pandas
* NLTK
* Joblib

---

# Project Structure

```
Auto_Email_Categorizer/
│
├── app.py
├── train.py
├── utils.py
├── requirements.txt
├── README.md
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
├── data/
│   └── support_tickets.csv
│
└── models/
    └── ticket_classifier.pkl
```

---

# Dataset

This project uses a **self-created dummy dataset** consisting of support tickets.

Dataset Fields:

* Subject
* Body
* Category

Categories:

* Billing
* Technical
* HR
* General

---

# Workflow

1. Load Dataset
2. Combine Subject + Body
3. Clean Text
4. Remove Stopwords
5. Convert text into TF-IDF vectors
6. Train Naive Bayes classifier
7. Evaluate model
8. Save trained model
9. Predict new tickets through the web application

---

# Text Preprocessing

The preprocessing pipeline includes:

* Lowercase conversion
* URL removal
* Email removal
* Number removal
* Punctuation removal
* Stopword removal
* Extra whitespace removal

---

# Feature Engineering

The project converts text into numerical vectors using **TF-IDF (Term Frequency–Inverse Document Frequency)**.

Why TF-IDF?

* Gives higher importance to meaningful words.
* Reduces the impact of common words.
* Produces sparse vectors suitable for text classification.

---

# Machine Learning Model

Classifier Used:

**Multinomial Naive Bayes**

Reason:

* Fast training
* Efficient prediction
* Excellent baseline for text classification
* Commonly used for spam filtering and email categorization

---

# Evaluation Metrics

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

---

# Human Review Logic

If the prediction confidence is below **60%**, the ticket is routed to:

**Needs Human Review**

instead of being automatically assigned.

---

# Priority Detection

Keyword-based rules assign ticket priority.

Urgent keywords include:

* urgent
* critical
* server down
* failed
* error
* crash
* cannot
* not working

Priority Levels:

* Urgent
* Normal

---

# How to Run

## 1. Clone the project

```bash
git clone https://github.com/Deepaananthan2004/Auto-Email-Ticket-Categorizer.git
```

## 2. Create Virtual Environment

```bash
python -m venv venv
```

Windows

```bash
venv\Scripts\activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Train the model

```bash
python train.py
```

This creates:

```
models/ticket_classifier.pkl
```

## 5. Start the application

```bash
python app.py
```

Open your browser:

```
http://127.0.0.1:5000
```

---

# Sample Prediction

**Input**

```
My payment was deducted twice from my account.
```

**Prediction**

```
Category   : Billing

Confidence : 96%

Priority   : Normal

Status     : Auto Assigned
```

---

# Future Improvements

* Use Logistic Regression or transformer-based models (e.g., DistilBERT)
* Expand the dataset with more diverse real-world tickets
* Add lemmatization and stemming
* Deploy the application using Docker and cloud hosting
* Integrate with enterprise helpdesk platforms
* Add user authentication and ticket history

---

# Conclusion

This project demonstrates how Natural Language Processing and Machine Learning can automate support ticket routing. It showcases text preprocessing, feature extraction, model training, evaluation, and deployment through a simple web interface, providing a practical foundation for AI-powered helpdesk systems.
