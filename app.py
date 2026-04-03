from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle, re, os
import numpy as np

app = Flask(__name__)
CORS(app)

MAXLEN    = 1000
THRESHOLD = 0.8

print("Loading model...")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # silence TF warnings

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# compile=False skips optimizer loading — avoids batch_shape errors
model = load_model("saved_model.keras", compile=False)
print("Model loaded!")

print("Loading tokenizer...")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
print("API ready!")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

@app.route("/")
def home():
    return jsonify({
        "status":   "running",
        "model":    "virajyawale/fakenewsdetection",
        "type":     "Keras LSTM + Word2Vec",
        "accuracy": "98.8%",
        "tf":       tf.__version__
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Send JSON with a 'text' field"}), 400

    text = str(data["text"]).strip()
    if len(text) < 20:
        return jsonify({"error": "Text too short"}), 400

    cleaned   = clean_text(text)
    seq       = tokenizer.texts_to_sequences([cleaned])
    padded    = pad_sequences(seq, maxlen=MAXLEN)
    raw_score = float(model.predict(padded, verbose=0)[0][0])

    label      = "REAL" if raw_score > THRESHOLD else "FAKE"
    confidence = round((raw_score if label == "REAL" else 1 - raw_score) * 100, 1)

    return jsonify({
        "result":       label,
        "confidence":   f"{confidence}%",
        "raw_score":    round(raw_score, 4),
        "text_preview": text[:80] + "..." if len(text) > 80 else text
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)