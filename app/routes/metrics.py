from fastapi import APIRouter
from app.services.yolo_service import model

router = APIRouter()

@router.get("/metrics")
def get_metrics():

    metrics = model.val()

    return {
        "Precision": metrics.box.mp,
        "Recall": metrics.box.mr,
        "mAP50": metrics.box.map50,
        "mAP50-95": metrics.box.map
    }