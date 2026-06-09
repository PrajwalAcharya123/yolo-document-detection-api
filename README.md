# YOLO Document Detection API

FastAPI backend for document layout detection using YOLO.

## Features

* YOLO object detection
* FastAPI backend
* Swagger UI
* Bounding box detection
* Metrics API
* Prediction image API
* CI/CD with GitHub Actions

## Run Locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```
