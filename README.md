# Auto Email / Ticket Categorizer

## Overview

The **Auto Email / Ticket Categorizer** is an AI-powered support ticket classification system that automatically categorizes incoming customer emails into the appropriate department.

The application combines **Traditional Machine Learning** with a **Large Language Model (LLM)** to achieve accurate and reliable predictions.

* **Primary Model:** TF-IDF + Multinomial Naive Bayes
* **Fallback Model:** NVIDIA NIM API using **Llama 3.3 70B Instruct**

If the Machine Learning model predicts with high confidence, the ticket is automatically categorized. If the confidence is low, the ticket is intelligently analyzed using an LLM to improve prediction accuracy.

---

# Features

* AI-powered support ticket classification
* Hybrid ML + LLM architecture
* TF-IDF text vectorization
* Multinomial Naive Bayes classifier
* NVIDIA Llama 3.3 integration
* Automatic confidence scoring
* Human review detection
* Priority prediction (Urgent / Normal)
* Explanation (Reason) for every prediction
* Modern Flask web interface
* Clean and modular project structure

---

# Tech Stack

* Python 3.12
* Flask
* Scikit-learn
* Pandas
* NLTK
* Joblib
* OpenAI Python SDK (NVIDIA NIM compatible)
* NVIDIA NIM API (Llama 3.3 70B Instruct)
* HTML5
* CSS3

---

# Project Structure

```text
Auto-Email-Ticket-Categorizer/
│
├── app.py
├── train.py
├── utils.py
├── requirements.txt
├── README.md
├── .env
│
├── data/
│   └── support_tickets.csv
│
├── models/
│   └── ticket_classifier.pkl
│
├── templates/
│   └── index.html
│
└── static/
    └── style.css
```

---

# Dataset

The project uses a self-created support ticket dataset containing multiple business scenarios.

### Dataset Fields

* Subject
* Body
* Category

### Categories

* Billing
* Technical
* HR
* General

---

# System Architecture

```text
                 Support Ticket
                       │
                       ▼
           Text Cleaning & Preprocessing
                       │
                       ▼
          TF-IDF + Naive Bayes Classifier
                       │
             Confidence Score Generated
                       │
         ┌─────────────┴─────────────┐
         │                           │
 Confidence ≥ 60%           Confidence < 60%
         │                           │
         ▼                           ▼
 Machine Learning             NVIDIA Llama 3.3
 Prediction                  (LLM Prediction)
         │                           │
         └─────────────┬─────────────┘
                       ▼
               Final Prediction
```

---

# Workflow

1. Load support ticket dataset
2. Combine subject and body
3. Clean and preprocess text
4. Convert text into TF-IDF vectors
5. Train Multinomial Naive Bayes classifier
6. Save trained model
7. Predict category using Machine Learning
8. Calculate confidence score
9. If confidence is below 60%, use NVIDIA Llama 3.3
10. Display:

* Category
* Confidence
* Priority
* Prediction Reason
* Prediction Source

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

### Why TF-IDF?

* Gives higher importance to meaningful words
* Reduces the influence of common words
* Produces efficient sparse vectors
* Works well for traditional NLP classifiers

---

# Machine Learning Model

### Primary Model

**Multinomial Naive Bayes**

Advantages:

* Fast training
* Lightweight
* Efficient prediction
* Excellent baseline for text classification

---

# LLM Integration

When the Machine Learning model is uncertain, the application automatically calls **NVIDIA NIM API** using:

**Meta Llama 3.3 70B Instruct**

The LLM returns:

* Category
* Confidence
* Priority
* Explanation (Reason)

This improves prediction quality for complex or ambiguous support tickets.

---

# Evaluation Metrics

The Machine Learning model is evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

---

# Human Review Logic

If:

```text
Confidence < 60%
```

the application automatically switches to **NVIDIA Llama 3.3** for a more intelligent prediction.

This hybrid strategy improves accuracy while keeping inference efficient.

---

# Priority Detection

Priority is determined using keyword-based rules.

### Urgent Keywords

* urgent
* critical
* server down
* failed
* crash
* cannot
* error
* outage
* not working

### Priority Levels

* Urgent
* Normal

---

# How to Run

## 1. Clone the Repository

```bash
git clone https://github.com/Deepaananthan2004/Auto-Email-Ticket-Categorizer.git

cd Auto-Email-Ticket-Categorizer
```

---

## 2. Create Virtual Environment

```bash
python -m venv venv
```

Windows

```bash
venv\Scripts\activate
```

Linux / macOS

```bash
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Configure Environment Variables

Create a `.env` file.

```env
NVIDIA_API_KEY=your_nvidia_api_key
```

---

## 5. Train the Machine Learning Model

```bash
python train.py
```

This generates:

```text
models/ticket_classifier.pkl
```

---

## 6. Run the Flask Application

```bash
python app.py
```

Open your browser:

```text
http://127.0.0.1:5000
```

---

# Example Prediction

### Input

```text
Hi,

We have received your details.

After deducting the applicable registration cancellation charges,
a refund of Rs.2000 will be processed and credited within 2–5 working days.
```

### Output

```text
Category   : Billing

Confidence : 98%

Priority   : Normal

Status     : Predicted using NVIDIA Llama 3.3

Reason     : The ticket discusses refund processing and cancellation charges.
```

---

# Future Improvements

* Replace Naive Bayes with Logistic Regression
* Fine-tune a transformer model (BERT/DistilBERT)
* Expand dataset with real-world enterprise tickets
* Add OCR for email attachments
* Integrate Retrieval-Augmented Generation (RAG)
* Connect with Jira, ServiceNow, or Zendesk
* Add authentication and ticket history
* Deploy using Docker and cloud platforms

---

# Conclusion

This project demonstrates a practical hybrid AI workflow by combining a traditional NLP classifier with a Large Language Model. The Machine Learning model efficiently handles routine support tickets, while NVIDIA Llama 3.3 provides intelligent fallback predictions for low-confidence cases. The result is a scalable, explainable, and production-inspired ticket categorization system suitable for enterprise helpdesk automation and AI engineering portfolios.
