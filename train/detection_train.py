import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
SEED       = 42

TRAIN_DIR  = "datasets/detection/train"
VALID_DIR  = "datasets/detection/valid"
TEST_DIR   = "datasets/detection/test"
MODEL_PATH = "models/detection/detection_model.keras"

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

raw_train = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    label_mode="categorical",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    label_mode="categorical",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    VALID_DIR,
    label_mode="categorical",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    label_mode="categorical",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_labels = train_ds.class_names
num_class    = len(class_labels)
print("Class labels:", class_labels)

y_train = np.concatenate(
    [np.argmax(labels.numpy(), axis=1) for _, labels in raw_train]
)
class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = {i: float(class_weights_array[i]) for i in range(num_class)}
print("Class weights:", class_weight_dict)

def augment_and_normalize(images, labels):
    images = tf.image.random_flip_left_right(images)
    images = tf.image.random_flip_up_down(images)
    images = tf.image.random_brightness(images, max_delta=0.2)
    images = tf.image.random_contrast(images, lower=0.8, upper=1.2)
    images = tf.image.random_saturation(images, lower=0.8, upper=1.2)
    images = tf.clip_by_value(images, 0.0, 255.0)
    images = tf.keras.applications.mobilenet_v2.preprocess_input(images)
    return images, labels

def normalize_only(images, labels):
    images = tf.keras.applications.mobilenet_v2.preprocess_input(images)
    return images, labels

train_ds = (
    train_ds
    .shuffle(1000, seed=SEED, reshuffle_each_iteration=True)
    .map(augment_and_normalize, num_parallel_calls=tf.data.AUTOTUNE)
    .cache()
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)
val_ds = (
    val_ds
    .map(normalize_only, num_parallel_calls=tf.data.AUTOTUNE)
    .cache()
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)
test_ds = (
    test_ds
    .map(normalize_only, num_parallel_calls=tf.data.AUTOTUNE)
    .cache()
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

alpha_vals = 1.0 / (class_weights_array * num_class)
alpha_vals = (alpha_vals / alpha_vals.sum()).tolist()

focal_loss = tf.keras.losses.CategoricalFocalCrossentropy(
    gamma=2.0,
    alpha=alpha_vals
)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(256, activation="relu")(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(num_class, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=focal_loss,
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc")
    ]
)
model.summary()

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    MODEL_PATH,
    monitor="val_auc",
    mode="max",
    save_best_only=True,
    verbose=1
)
earlystop_cb = tf.keras.callbacks.EarlyStopping(
    monitor="val_auc",
    mode="max",
    patience=8,
    restore_best_weights=True,
    verbose=1
)
reducelr_cb = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

callbacks_phase1 = [checkpoint_cb, earlystop_cb, reducelr_cb]

callbacks_phase2 = [checkpoint_cb, earlystop_cb]

print("\n=== Phase 1: Training head (base frozen) ===")
history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    # FIX: no class_weight here — focal loss alpha already handles imbalance
    callbacks=callbacks_phase1
)

print("\n=== Phase 2: Fine-tuning top 40 layers ===")
base_model.trainable = True
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False                          # always freeze BN during fine-tune
    elif base_model.layers.index(layer) < len(base_model.layers) - 40:
        layer.trainable = False

FINETUNE_EPOCHS = 10
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-4,
    decay_steps=FINETUNE_EPOCHS * len(train_ds)
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss=focal_loss,
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc")
    ]
)

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINETUNE_EPOCHS,
    callbacks=callbacks_phase2
)

print("\n=== Test set evaluation ===")
results = model.evaluate(test_ds, verbose=1)
for name, value in zip(model.metrics_names, results):
    print(f"  {name}: {value:.4f}")

# Collect ground-truth and predictions
y_pred_probs = model.predict(test_ds, verbose=1)
y_pred       = np.argmax(y_pred_probs, axis=1)
y_true_onehot = np.concatenate([labels.numpy() for _, labels in test_ds])
y_true        = np.argmax(y_true_onehot, axis=1)

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_labels, digits=4))

print("\nPer-class ROC-AUC:")
try:
    roc_aucs = roc_auc_score(y_true_onehot, y_pred_probs, average=None)
    for cls, auc in zip(class_labels, roc_aucs):
        print(f"  AUC [{cls}]: {auc:.4f}")
    macro_auc = roc_auc_score(y_true_onehot, y_pred_probs, average="macro")
    print(f"  Macro AUC: {macro_auc:.4f}")
except ValueError as e:
    print(f"  Could not compute per-class AUC: {e}")

model.save(MODEL_PATH)
print(f"\nSaved model to {MODEL_PATH}")