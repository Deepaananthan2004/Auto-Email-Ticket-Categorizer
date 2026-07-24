from flask import Flask, render_template, request
import joblib
import json
import os

from openai import OpenAI

from utils import clean_text, get_priority
from dotenv import load_dotenv
app = Flask(__name__)
load_dotenv()
# Load trained ML model
model = joblib.load("models/ticket_classifier.pkl")

# NVIDIA NIM Client
client = OpenAI(
    api_key=os.getenv("NVIDIA_API_KEY"),   # Set as an environment variable
    base_url="https://integrate.api.nvidia.com/v1"
)


def predict_with_llm(ticket):

    prompt = f"""
You are an AI Support Ticket Classifier.

Classify the following support ticket into ONLY ONE category.

Categories:
- Billing
- Technical
- HR
- General

Determine:
- category
- priority (Urgent or Normal)
- confidence (0-100)
- reason

Return ONLY valid JSON.

Ticket:
{ticket}
"""

    response = client.chat.completions.create(
        model="meta/llama-3.3-70b-instruct",
        temperature=0,
        max_tokens=200,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You are an expert support ticket classifier."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return json.loads(response.choices[0].message.content)


@app.route("/", methods=["GET", "POST"])
def home():

    result = None

    if request.method == "POST":

        ticket = request.form["ticket"]

        cleaned = clean_text(ticket)

        # -------------------------
        # Traditional ML Prediction
        # -------------------------
        prediction = model.predict([cleaned])[0]
        confidence = model.predict_proba([cleaned]).max()

        priority = get_priority(ticket)

        # -------------------------
        # Low confidence → LLM
        # -------------------------
        if confidence < 0.60:

            llm_result = predict_with_llm(ticket)

            result = {
                "ticket": ticket,
                "category": llm_result["category"],
                "confidence": llm_result["confidence"],
                "priority": llm_result["priority"],
                "review": "Predicted using NVIDIA Llama 3.3",
                "reason": llm_result["reason"]
            }

        else:

            result = {
                "ticket": ticket,
                "category": prediction,
                "confidence": round(confidence * 100, 2),
                "priority": priority,
                "review": "Predicted using ML Model",
                "reason": "High confidence prediction from TF-IDF + Naive Bayes."
            }

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)