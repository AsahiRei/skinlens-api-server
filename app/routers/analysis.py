
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from app.schemas.common import (
    AnalyzeTaskCreateResponse,
    AnalyzeTaskStatus,
    PredictionResponse,
    PredictionWithHeatmapResponse,
    ImageUrlRequest,
    ErrorResponse,
    ChatbotRequest,
    ChatbotResponse,
)
from app.core.dependencies import get_task_service, get_detection_service
from app.services.task_service import TaskService
from app.services.detection_service import DetectionService
from utils.chatbot import get_chatbot_reply

router = APIRouter(prefix="/api/v1", tags=["Analysis"])


@router.post("/analyze", response_model=AnalyzeTaskCreateResponse)
async def analyze(
    image: UploadFile = File(...),
    task_service: TaskService = Depends(get_task_service),
):
    contents = await image.read()
    task_id = task_service.create_task()
    task_service.start_processing(task_id, contents)
    return AnalyzeTaskCreateResponse(task_id=task_id)


@router.get("/progress/{task_id}", response_model=AnalyzeTaskStatus)
def get_progress(
    task_id: str,
    task_service: TaskService = Depends(get_task_service),
):
    task = task_service.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return AnalyzeTaskStatus(**task)


@router.post(
    "/predict-skin-url",
    response_model=PredictionResponse,
    responses={400: {"model": ErrorResponse}},
)
def predict_url(
    request: ImageUrlRequest,
    detection_service: DetectionService = Depends(get_detection_service),
):
    result = detection_service.predict_url(request.image_url)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result)
    return PredictionResponse(**result)


@router.post(
    "/predict-skin-url-with-heatmap",
    response_model=PredictionWithHeatmapResponse,
    responses={400: {"model": ErrorResponse}},
)
def predict_skin_with_heatmap(
    request: ImageUrlRequest,
    detection_service: DetectionService = Depends(get_detection_service),
):
    result = detection_service.predict_url_with_heatmap(request.image_url)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result)
    return PredictionWithHeatmapResponse(**result)


@router.post("/predict-intent", response_model=ChatbotResponse)
def predict_intent(request: ChatbotRequest):
    try:
        intent, confidence, response = get_chatbot_reply(request.message)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if confidence < 0.75:
        return ChatbotResponse(
            intent="fallback",
            confidence=confidence,
            response="I'm sorry, I don't understand your question. Could you rephrase it?"
        )
    return ChatbotResponse(
        intent=intent,
        confidence=confidence,
        response=response,
    )