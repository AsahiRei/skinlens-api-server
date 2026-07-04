
import numpy as np
import cv2
from typing import Optional
from insightface.app import FaceAnalysis


class FaceService:
    _instance: Optional["FaceService"] = None
    _face_app: Optional[FaceAnalysis] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def initialize(self):
        if self._face_app is None:
            self._face_app = FaceAnalysis(providers=["CPUExecutionProvider"])
            self._face_app.prepare(ctx_id=-1)

    def detect_faces(self, img: np.ndarray):
        if self._face_app is None:
            raise RuntimeError("FaceService not initialized. Call initialize() first.")
        return self._face_app.get(img)

    def has_face(self, image_bytes: bytes) -> bool:
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        faces = self.detect_faces(img)
        return len(faces) > 0

