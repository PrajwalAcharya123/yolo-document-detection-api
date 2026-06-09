from fastapi import APIRouter, UploadFile, File
from app.services.yolo_service import model
from app.utils.helper import create_folder

import shutil
import os

router = APIRouter()

# Create uploads folder
create_folder("uploads")

@router.post("/predict")
async def predict(
    file: UploadFile = File(...),
    conf: float = 0.5,
    iou: float = 0.5
):

    # Save uploaded image
    file_path = f"uploads/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run YOLO prediction
    results = model.predict(
        source=file_path,
        conf=conf,
        iou=iou
    )

    detections = []

    for r in results:

        boxes = r.boxes

        for box in boxes:

            class_id = int(box.cls[0])

            confidence = float(box.conf[0])

            bbox = box.xyxy[0].tolist()

            detections.append({
                "class_id": class_id,
                "class_name": model.names[class_id],
                "confidence": confidence,
                "bounding_box": bbox
            })

    return {
        "filename": file.filename,
        "total_detections": len(detections),
        "detections": detections
    }