
from pydantic import BaseModel, Field
from typing import List, Optional, Union


class PredictionItem(BaseModel):
    class_name: str = Field(..., alias="class")
    confidence: float

    class Config:
        populate_by_name = True


class BasePredictionResponse(BaseModel):
    prediction: str
    confidence: float
    top_predictions: List[PredictionItem]


class PredictionResponse(BasePredictionResponse):
    pass


class PredictionWithHeatmapResponse(BasePredictionResponse):
    heatmap: str


class ImageUrlRequest(BaseModel):
    image_url: str


class ChatbotRequest(BaseModel):
    message: str = Field(..., min_length=1)


class ChatbotResponse(BaseModel):
    intent: str
    confidence: float
    response: str


class AnalyzeTaskCreateResponse(BaseModel):
    task_id: str


class ErrorResponse(BaseModel):
    error: str
    details: Optional[str] = None


class AnalyzeTaskStatus(BaseModel):
    progress: int
    status: str
    result: Optional[Union[BasePredictionResponse, ErrorResponse]] = None
    heatmap: Optional[str] = None


class HealthResponse(BaseModel):
    message: str
    version: str
    author: str