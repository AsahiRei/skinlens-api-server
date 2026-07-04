import json
import os
import pickle
import random
from pathlib import Path
from typing import Tuple

try:
    import numpy as np
    import tensorflow as tf
except ImportError:  # pragma: no cover - depends on runtime environment
    np = None
    tf = None

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_FOLDER = BASE_DIR / "models" / "chatbot"
INTENT_FOLDER = BASE_DIR / "datasets" / "chatbot"

model = None
vectorizer = None
label_encoder = None
all_intents = []


def _ensure_runtime_dependencies():
    if np is None or tf is None:
        raise ImportError("numpy and tensorflow are required to run chatbot prediction")
    if model is None or vectorizer is None or label_encoder is None:
        _load_models()


def _load_models():
    global model, vectorizer, label_encoder, all_intents

    if model is not None and vectorizer is not None and label_encoder is not None:
        return

    model = tf.keras.models.load_model(str(MODEL_FOLDER / "chatbot_model.keras"))
    vectorizer = tf.keras.models.load_model(str(MODEL_FOLDER / "vectorizer.keras"))

    with open(MODEL_FOLDER / "label_encoder.pkl", "rb") as handle:
        label_encoder = pickle.load(handle)

    all_intents = []
    for root, _, files in os.walk(INTENT_FOLDER):
        for file in files:
            if file.endswith(".json"):
                with open(os.path.join(root, file), "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                    all_intents.extend(data.get("intents", []))


def predict_intent(message: str) -> Tuple[str, float]:
    _ensure_runtime_dependencies()

    message = message.lower().strip()
    X = vectorizer.predict(tf.constant([message]), verbose=0)
    prediction = model.predict(X, verbose=0)[0]

    confidence = float(np.max(prediction))
    intent = label_encoder.inverse_transform([np.argmax(prediction)])[0]

    return intent, confidence


def get_response(intent: str) -> str:
    for item in all_intents:
        if item.get("tag") == intent:
            return random.choice(item.get("responses", ["Sorry, I don't know how to answer that."]))

    return "Sorry, I don't know how to answer that."


def get_chatbot_reply(message: str) -> Tuple[str, float, str]:
    _ensure_runtime_dependencies()

    intent, confidence = predict_intent(message)
    if confidence < 0.75:
        response = "Sorry, I didn't quite understand that."
    else:
        response = get_response(intent)

    return intent, confidence, response