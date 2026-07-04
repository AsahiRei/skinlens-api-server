
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
import numpy as np
import requests
import cv2
import base64
import json
from PIL import Image
from io import BytesIO
from typing import List, Optional
from app.config.settings import get_settings

settings = get_settings()


class DetectionService:
    _instance: Optional["DetectionService"] = None
    _model = None
    _class_labels = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_model()
        return cls._instance

    def _load_model(self):
        self._model = tf.keras.models.load_model(settings.MODEL_PATH)
        with open(settings.CLASS_PATH, "r") as f:
            self._class_labels = json.load(f)

    def _make_gradcam_heatmap(
        self, img_array, model, prediction_class: Optional[str] = None
    ):
        base_model = model.get_layer("mobilenetv2_1.00_224")
        last_conv_layer = base_model.get_layer("Conv_1")
        base_grad_model = tf.keras.Model(
            inputs=base_model.input,
            outputs=[last_conv_layer.output, base_model.output],
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

    def _apply_heatmap_overlay(
        self, original_img, heatmap, prediction_class: Optional[str] = None
    ):
        heatmap_resized = cv2.resize(
            heatmap, (original_img.shape[1], original_img.shape[0])
        )
        heatmap_resized = cv2.GaussianBlur(heatmap_resized, (11, 11), 0)
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        alpha = 0.15 if (prediction_class and prediction_class.lower() == "normal") else 0.4
        overlay = cv2.addWeighted(original_img, 1 - alpha, heatmap_color, alpha, 0)
        return overlay, heatmap_uint8

    def predict_file(
        self, image_bytes: bytes, top_k: int = 3
    ) -> dict:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        resized_img = img.resize(settings.IMG_SIZE)
        img_array = np.array(resized_img)
        img_array = np.expand_dims(img_array, axis=0)
        processed_img = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        predictions = self._model.predict(processed_img, verbose=0)[0]
        top_indices = np.argsort(predictions)[::-1][:top_k]
        top_predictions = [
            {"class": self._class_labels[i], "confidence": float(predictions[i] * 100)}
            for i in top_indices
        ]
        return {
            "prediction": top_predictions[0]["class"],
            "confidence": top_predictions[0]["confidence"],
            "top_predictions": top_predictions,
        }

    def predict_file_with_heatmap(
        self, image_bytes: bytes, top_k: int = 3
    ) -> dict:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        original_img = np.array(img)
        resized_img = img.resize(settings.IMG_SIZE)
        img_array = np.array(resized_img)
        img_array = np.expand_dims(img_array, axis=0)
        processed_img = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

        predictions = self._model.predict(processed_img, verbose=0)[0]
        top_indices = np.argsort(predictions)[::-1][:top_k]
        top_predictions = [
            {"class": self._class_labels[i], "confidence": float(predictions[i] * 100)}
            for i in top_indices
        ]
        prediction_class = top_predictions[0]["class"]

        heatmap = self._make_gradcam_heatmap(
            processed_img, self._model, prediction_class=prediction_class
        )
        overlay, _ = self._apply_heatmap_overlay(
            original_img, heatmap, prediction_class=prediction_class
        )

        output_image = Image.fromarray(overlay)
        output_image = output_image.resize(
            (original_img.shape[1] // 2, original_img.shape[0] // 2), Image.LANCZOS
        )
        buffer = BytesIO()
        output_image.save(buffer, format="JPEG", quality=80)
        heatmap_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {
            "prediction": prediction_class,
            "confidence": top_predictions[0]["confidence"],
            "top_predictions": top_predictions,
            "heatmap": heatmap_base64,
        }

    def predict_url(self, image_url: str, top_k: int = 3) -> dict:
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
        except Exception as e:
            return {"error": "Failed to fetch image", "details": str(e)}
        return self.predict_file(response.content, top_k)

    def predict_url_with_heatmap(
        self, image_url: str, top_k: int = 3
    ) -> dict:
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
        except Exception as e:
            return {"error": "Failed to fetch image", "details": str(e)}
        return self.predict_file_with_heatmap(response.content, top_k)

