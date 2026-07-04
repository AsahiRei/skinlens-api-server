import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import numpy as np
import requests
import cv2
import base64
import json
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

MODEL_PATH = "models/detection/detection_model.keras"
CLASS_PATH = "models/detection/class.json"
IMG_SIZE = (224, 224)

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_PATH, "r") as f:
    class_labels = json.load(f)

def make_gradcam_heatmap(img_array, model, prediction_class: str = None, confidence: float = None):
    base_model = model.get_layer("mobilenetv2_1.00_224")
    last_conv_layer = base_model.get_layer("Conv_1")
    base_grad_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=[last_conv_layer.output, base_model.output]
    )
    inputs = model.input
    last_conv_output, base_output = base_grad_model(inputs)
    x = base_output
    for layer in model.layers[2:]:
        x = layer(x)
    grad_model = tf.keras.Model(inputs, [last_conv_output, x])

    with tf.GradientTape() as tape:
        last_conv_layer_output, predictions = grad_model(img_array, training=False)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    if prediction_class and prediction_class.lower() == "normal":
        heatmap = np.zeros_like(heatmap)

    return heatmap

def apply_heatmap_overlay(original_img, heatmap, prediction_class: str = None):
    heatmap_resized = cv2.resize(
        heatmap,
        (original_img.shape[1], original_img.shape[0])
    )
    heatmap_resized = cv2.GaussianBlur(heatmap_resized, (11, 11), 0)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    alpha = 0.15 if (prediction_class and prediction_class.lower() == "normal") else 0.4
    overlay = cv2.addWeighted(original_img, 1 - alpha, heatmap_color, alpha, 0)
    return overlay, heatmap_uint8

def display_heatmap(original_img, heatmap):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(original_img)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Grad-CAM Heatmap")
    heatmap_resized = cv2.resize(
        heatmap,
        (original_img.shape[1], original_img.shape[0])
    )
    heatmap_resized = cv2.GaussianBlur(heatmap_resized, (11, 11), 0)
    plt.imshow(heatmap_resized, cmap="jet")
    plt.axis("off")
    overlay, _ = apply_heatmap_overlay(original_img, heatmap)
    plt.subplot(1, 3, 3)
    plt.title("Overlay")
    plt.imshow(overlay)
    plt.axis("off")
    plt.show()

def predict_skin_file(image_bytes, top_k=3):
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    resized_img = img.resize(IMG_SIZE)
    img_array = np.array(resized_img)
    img_array = np.expand_dims(img_array, axis=0)
    processed_img = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    predictions = model.predict(processed_img, verbose=0)[0]
    top_indices = np.argsort(predictions)[::-1][:top_k]
    top_predictions = [
        {
            "class": class_labels[i],
            "confidence": float(predictions[i] * 100)
        }
        for i in top_indices
    ]
    return {
        "prediction": top_predictions[0]["class"],
        "confidence": top_predictions[0]["confidence"],
        "top_predictions": top_predictions,
    }

def predict_skin_file_with_heatmap(image_bytes, top_k=3):
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    original_img = np.array(img)
    resized_img = img.resize(IMG_SIZE)
    img_array = np.array(resized_img)
    img_array = np.expand_dims(img_array, axis=0)
    processed_img = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    predictions = model.predict(processed_img, verbose=0)[0]
    top_indices = np.argsort(predictions)[::-1][:top_k]
    top_predictions = [
        {"class": class_labels[i], "confidence": float(predictions[i] * 100)}
        for i in top_indices
    ]
    prediction_class = top_predictions[0]["class"]

    heatmap = make_gradcam_heatmap(processed_img, model, prediction_class=prediction_class)
    overlay, _ = apply_heatmap_overlay(original_img, heatmap, prediction_class=prediction_class)

    output_image = Image.fromarray(overlay)
    output_image = output_image.resize(
        (original_img.shape[1] // 2, original_img.shape[0] // 2),
        Image.LANCZOS
    )
    buffer = BytesIO()
    output_image.save(buffer, format="JPEG", quality=80)
    heatmap_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return {
        "prediction": prediction_class,
        "confidence": top_predictions[0]["confidence"],
        "top_predictions": top_predictions,
        "heatmap": heatmap_base64
    }

def predict_skin_url(image_url, top_k=3):
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
    except Exception as e:
        return {"error": "Failed to fetch image", "details": str(e)}
    img = Image.open(BytesIO(response.content)).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    predictions = model.predict(img_array, verbose=0)[0]
    top_indices = np.argsort(predictions)[::-1][:top_k]
    top_predictions = [
        {
            "class": class_labels[i],
            "confidence": float(predictions[i] * 100)
        }
        for i in top_indices
    ]
    return {
        "prediction": top_predictions[0]["class"],
        "confidence": top_predictions[0]["confidence"],
        "top_predictions": top_predictions
    }

def predict_skin_url_with_heatmap(image_url, top_k=3):
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
    except Exception as e:
        return {
            "error": "Failed to fetch image",
            "details": str(e)
        }
    img = Image.open(BytesIO(response.content)).convert("RGB")
    original_img = np.array(img)
    resized_img = img.resize(IMG_SIZE)
    img_array = np.array(resized_img)
    img_array = np.expand_dims(img_array, axis=0)
    processed_img = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    predictions = model.predict(processed_img, verbose=0)[0]
    top_indices = np.argsort(predictions)[::-1][:top_k]
    top_predictions = [
        {
            "class": class_labels[i],
            "confidence": float(predictions[i] * 100)
        }
        for i in top_indices
    ]
    heatmap = make_gradcam_heatmap(processed_img, model)
    overlay, _ = apply_heatmap_overlay(original_img, heatmap)
    output_image = Image.fromarray(overlay)
    buffer = BytesIO()
    output_image.save(buffer, format="PNG")
    heatmap_base64 = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")
    return {
        "prediction": top_predictions[0]["class"],
        "confidence": top_predictions[0]["confidence"],
        "top_predictions": top_predictions,
        "heatmap": heatmap_base64
    }