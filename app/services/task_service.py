
import threading
import uuid
import numpy as np
import cv2
from typing import Dict, Optional
from app.services.detection_service import DetectionService
from app.services.face_service import FaceService


class TaskService:
    _instance: Optional["TaskService"] = None
    _tasks: Dict[str, dict] = {}
    _detection_service: Optional[DetectionService] = None
    _face_service: Optional[FaceService] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._detection_service = DetectionService()
            cls._instance._face_service = FaceService()
        return cls._instance

    def create_task(self) -> str:
        task_id = str(uuid.uuid4())
        self._tasks[task_id] = {
            "progress": 0,
            "status": "Queued",
            "result": None,
            "heatmap": None,
        }
        return task_id

    def get_task(self, task_id: str) -> Optional[dict]:
        return self._tasks.get(task_id)

    def process_image_task(self, task_id: str, contents: bytes):
        try:
            self._tasks[task_id]["progress"] = 10
            self._tasks[task_id]["status"] = "Reading image"
            image_array = np.frombuffer(contents, dtype=np.uint8)
            img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            self._tasks[task_id]["progress"] = 50
            self._tasks[task_id]["status"] = "Detecting face"
            faces = self._face_service.detect_faces(img)

            if len(faces) == 0:
                self._tasks[task_id]["progress"] = 100
                self._tasks[task_id]["status"] = "No face detected"
                self._tasks[task_id]["result"] = {"error": "No face detected in the image"}
                return

            self._tasks[task_id]["progress"] = 70
            self._tasks[task_id]["status"] = "Analyzing skin"
            result = self._detection_service.predict_file(contents)

            self._tasks[task_id]["progress"] = 90
            self._tasks[task_id]["status"] = "Completed"
            self._tasks[task_id]["result"] = result
            self._tasks[task_id]["status"] = "Generating heatmap"
            heatmap_result = self._detection_service.predict_file_with_heatmap(contents)
            self._tasks[task_id]["heatmap"] = heatmap_result.get("heatmap")

            self._tasks[task_id]["progress"] = 100
            self._tasks[task_id]["status"] = "Completed"

        except Exception as e:
            self._tasks[task_id]["progress"] = 100
            self._tasks[task_id]["status"] = "Failed"
            self._tasks[task_id]["result"] = {"error": str(e)}

    def start_processing(self, task_id: str, contents: bytes):
        threading.Thread(
            target=self.process_image_task, args=(task_id, contents), daemon=True
        ).start()

