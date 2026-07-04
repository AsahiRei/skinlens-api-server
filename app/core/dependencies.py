
from app.services.detection_service import DetectionService
from app.services.face_service import FaceService
from app.services.task_service import TaskService


def get_detection_service() -> DetectionService:
    return DetectionService()


def get_face_service() -> FaceService:
    return FaceService()


def get_task_service() -> TaskService:
    return TaskService()

