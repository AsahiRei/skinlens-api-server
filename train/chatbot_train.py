import os
import json
import pickle
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder

# =====================================================
# SETTINGS
# =====================================================

INTENT_FOLDER = "datasets/chatbot"
MODEL_FOLDER = "models/chatbot"

MAX_TOKENS = 10000
MAX_LENGTH = 30
EMBEDDING_DIM = 128

EPOCHS = 300
BATCH_SIZE = 8

os.makedirs(MODEL_FOLDER, exist_ok=True)

# =====================================================
# LOAD DATASET
# =====================================================

all_intents = []

print("Loading intent files...\n")

for root, _, files in os.walk(INTENT_FOLDER):
    for file in files:
        if file.endswith(".json"):

            path = os.path.join(root, file)

            print(f"Loaded: {path}")

            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "intents" not in data:
                print(f"Skipped {file}: missing intents key")
                continue

            all_intents.extend(data["intents"])

print(f"\nTotal intents: {len(all_intents)}")

# =====================================================
# PREPARE DATA
# =====================================================

sentences = []
labels = []

for intent in all_intents:
    for pattern in intent["patterns"]:
        sentences.append(pattern.lower().strip())
        labels.append(intent["tag"])

print(f"Total patterns: {len(sentences)}")

# =====================================================
# LABEL ENCODER
# =====================================================

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

with open(os.path.join(MODEL_FOLDER, "label_encoder.pkl"), "wb") as f:
    pickle.dump(label_encoder, f)

print("Saved label_encoder.pkl")

# =====================================================
# TEXT VECTORIZATION
# =====================================================

vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=MAX_TOKENS,
    output_mode="int",
    output_sequence_length=MAX_LENGTH
)

vectorizer.adapt(sentences)

# Convert text to integer sequences
X = vectorizer(tf.constant(sentences))

# Save vocabulary
vocabulary = vectorizer.get_vocabulary()

with open(os.path.join(MODEL_FOLDER, "vocabulary.pkl"), "wb") as f:
    pickle.dump(vocabulary, f)

print("Saved vocabulary.pkl")

# =====================================================
# BUILD MODEL
# =====================================================

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(MAX_LENGTH,)),

    tf.keras.layers.Embedding(
        input_dim=MAX_TOKENS,
        output_dim=EMBEDDING_DIM
    ),

    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128)
    ),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(
        64,
        activation="relu"
    ),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(
        len(label_encoder.classes_),
        activation="softmax"
    )
])

# =====================================================
# COMPILE
# =====================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

print("\nTraining...\n")
history = model.fit(
    X,
    y,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    verbose=1
)
model.save(os.path.join(MODEL_FOLDER, "chatbot_model.keras"))
tf.keras.models.save_model(
    tf.keras.Sequential([vectorizer]),
    os.path.join(MODEL_FOLDER, "vectorizer.keras")
)

print("\n==============================")
print("Training Complete")
print("==============================")
print("Saved:")
print(" - chatbot_model.keras")
print(" - vectorizer.keras")
print(" - vocabulary.pkl")
print(" - label_encoder.pkl")