from flask import Flask, render_template, request
import joblib

from utils import clean_text, get_priority

app = Flask(__name__)

model = joblib.load("models/ticket_classifier.pkl")


@app.route("/", methods=["GET", "POST"])
def home():

    result = None

    if request.method == "POST":

        ticket = request.form["ticket"]

        cleaned = clean_text(ticket)

        prediction = model.predict([cleaned])[0]

        confidence = model.predict_proba([cleaned]).max()

        priority = get_priority(ticket)

        review = "Needs Human Review" if confidence < 0.60 else "Auto Assigned"

        result = {
            "ticket": ticket,
            "category": prediction,
            "confidence": round(confidence * 100, 2),
            "priority": priority,
            "review": review,
        }

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)