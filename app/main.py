from fastapi import FastAPI

from app.routes import predict
from app.routes import metrics
from app.routes import visualization

app = FastAPI(
    title="YOLO Document Detection API",
    description="FastAPI backend for document detection using YOLO",
    version="1.0"
)

# Include routes
app.include_router(predict.router)
app.include_router(metrics.router)
app.include_router(visualization.router)

@app.get("/")
def home():

    return {
        "message": "YOLO FastAPI Server Running"
    }