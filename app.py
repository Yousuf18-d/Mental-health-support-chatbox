from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import logging
import random
from datetime import datetime, timedelta
from collections import defaultdict
import json
import os

app = Flask(_name_)
CORS(app)

# Logging setup
logging.basicConfig(level=logging.INFO)

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
model = AutoModelForSequenceClassification.from_pretrained("j-hartmann/emotion-english-distilroberta-base")

# Load FAQ data
FAQ_PATH = os.path.join(os.path.dirname(_file_), "faq.json")
if os.path.exists(FAQ_PATH):
    with open(FAQ_PATH, "r") as f:
        faq_data = json.load(f)
else:
    faq_data = []

# Constants
CRISIS_KEYWORDS = ["suicide", "kill myself", "end it", "hurt myself", "cut myself", "want to die", "can't go on"]
CONFIDENCE_THRESHOLD = 0.4
RATE_LIMIT_SECONDS = 2
user_last_request = defaultdict(lambda: datetime.min)

resources = {
    "sadness": "https://example.com/sadness-help",
    "joy": "https://example.com/joy-journal",
    "anger": "https://example.com/anger-management",
    "fear": "https://example.com/fear-support",
    "surprise": "https://example.com/mindfulness",
    "disgust": "https://example.com/talk-it-out",
    "love": "https://example.com/relationship-advice",
    "gratitude": "https://example.com/gratitude-journal",
    "neutral": "https://example.com/mental-health-check-in",
    "relief": "https://example.com/decompress",
    "curiosity": "https://example.com/discovery",
    "embarrassment": "https://example.com/self-compassion",
    "optimism": "https://example.com/positive-planning",
    "crisis": "https://www.opencounseling.com/suicide-hotlines",
    "rate_limited": "https://example.com/mental-health-check-in"
}

prewritten_responses = {
    "sadness": ["I'm sorry you're feeling this way. Want to talk more?"],
    "joy": ["That's wonderful!"],
    "anger": ["It's okay to feel angry. Want to talk about it?"],
    "fear": ["I'm here with you. What's scaring you?"],
    "surprise": ["That’s unexpected! How do you feel about it?"],
    "disgust": ["That must’ve been uncomfortable."],
    "love": ["Love is beautiful!"],
    "gratitude": ["Gratitude is powerful—thank you for sharing."],
    "neutral": ["Thanks for sharing. How can I support you?"],
    "relief": ["Glad to hear that!"],
    "curiosity": ["Curiosity is great. Want to explore something new?"],
    "embarrassment": ["We’ve all been there. You're not alone."],
    "optimism": ["Love the optimism! What are you hopeful for?"],
    "rate_limited": ["You're sending messages too fast. Take a deep breath."],
    "crisis": ["You're not alone. Help is available—please reach out."]
}

def is_rate_limited(ip: str) -> bool:
    now = datetime.utcnow()
    if now - user_last_request[ip] < timedelta(seconds=RATE_LIMIT_SECONDS):
        return True
    user_last_request[ip] = now
    return False

def detect_crisis(text: str) -> bool:
    return any(keyword in text.lower() for keyword in CRISIS_KEYWORDS)

def predict_emotion(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        confidence, label_id = torch.max(probs, dim=1)
        emotion = model.config.id2label[label_id.item()].lower()
    return emotion, confidence.item()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "").strip()
    user_ip = request.remote_addr or "unknown"

    if not user_input:
        return jsonify({
            "response": "It seems empty—can you share how you're feeling?",
            "emotion": "none",
            "resource": resources["neutral"]
        })

    if is_rate_limited(user_ip):
        return jsonify({
            "response": random.choice(prewritten_responses["rate_limited"]),
            "emotion": "rate_limited",
            "resource": resources["rate_limited"]
        })

    if detect_crisis(user_input):
        return jsonify({
            "response": random.choice(prewritten_responses["crisis"]),
            "emotion": "crisis",
            "resource": resources["crisis"]
        })

    for item in faq_data:
        if item["question"].lower() in user_input.lower():
            return jsonify({
                "response": item["answer"],
                "emotion": "faq",
                "confidence": "1.00",
                "resource": None,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            })

    try:
        emotion, confidence = predict_emotion(user_input)

        if confidence < CONFIDENCE_THRESHOLD:
            emotion = "neutral"

        reply = random.choice(prewritten_responses.get(emotion, prewritten_responses["neutral"]))
        return jsonify({
            "response": reply,
            "emotion": emotion,
            "confidence": f"{confidence:.2f}",
            "resource": resources.get(emotion),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        })

    except Exception as e:
        logging.error(f"Error: {e}")
        return jsonify({
            "response": "Something went wrong. Try again later.",
            "emotion": "error",
            "resource": resources["neutral"]
        })

if _name_ == "_main_":
    app.run(debug=True)
