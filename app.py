# ── app.py ───────────────────────────────────────────────────
# Flask API for Fake News Detection
# Model: Keras LSTM + Word2Vec + Keras Tokenizer
# Deploy: Render.com (free)

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle, re, os, numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
CORS(app)  # Allow browser extension to call this API

# ── Constants — must match Colab notebook exactly ────────────
MAXLEN    = 1000   # maxlen = 1000
THRESHOLD = 0.8    # > 0.8 → REAL, else → FAKE

# ── Load model files once at startup ────────────────────────
print("Loading LSTM model...")
model = load_model("saved_model.keras")

print("Loading tokenizer...")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

print("API ready!")

# ── Text cleaning — same as notebook ────────────────────────
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)  # remove special characters
    text = re.sub(r'\s+', ' ', text).strip()     # remove extra spaces
    return text

# ── Routes ───────────────────────────────────────────────────
@app.route("/")
def home():
    return jsonify({
        "status":  "running",
        "model":   "virajyawale/fakenewsdetection",
        "type":    "Keras LSTM + Word2Vec",
        "accuracy": "98.8%"
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "Send JSON with a 'text' field"}), 400

    text = str(data["text"]).strip()

    if len(text) < 20:
        return jsonify({"error": "Text too short — send at least 20 characters"}), 400

    # ── Pipeline (same as notebook) ──────────────────────────
    cleaned    = clean_text(text)
    seq        = tokenizer.texts_to_sequences([cleaned])
    padded     = pad_sequences(seq, maxlen=MAXLEN)
    raw_score  = float(model.predict(padded, verbose=0)[0][0])

    label      = "REAL" if raw_score > THRESHOLD else "FAKE"
    confidence = round((raw_score if label == "REAL" else 1 - raw_score) * 100, 1)

    return jsonify({
        "result":       label,
        "confidence":   f"{confidence}%",
        "raw_score":    round(raw_score, 4),
        "text_preview": text[:80] + "..." if len(text) > 80 else text
    })

# ── Run ──────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
