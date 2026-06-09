# from fastapi import APIRouter, UploadFile, File
# from fastapi.responses import FileResponse

# from app.services.yolo_service import model
# from app.utils.helper import create_folder

# import shutil
# import os

# router = APIRouter()

# create_folder("uploads")
# create_folder("results")

# @router.post("/predict-image")
# async def predict_image(file: UploadFile = File(...)):

#     file_path = f"uploads/{file.filename}"

#     with open(file_path, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)

#     # Predict and save image
#     results = model.predict(
#         source=file_path,
#         save=True,
#         project="results",
#         name="prediction"
#     )

#     saved_image_path = f"results/prediction/{file.filename}"

#     return FileResponse(saved_image_path)



# from fastapi import APIRouter, UploadFile, File
# from fastapi.responses import FileResponse

# from app.services.yolo_service import model
# from app.utils.helper import create_folder

# import shutil
# import os

# router = APIRouter()

# create_folder("uploads")

# @router.post("/predict-image")
# async def predict_image(file: UploadFile = File(...)):

#     # Save uploaded image
#     file_path = f"uploads/{file.filename}"

#     with open(file_path, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)

#     # Run prediction
#     results = model.predict(
#         source=file_path,
#         save=True
#     )

#     # Get actual saved image path
#     saved_image_path = results[0].save_dir / file.filename

#     print("Saved Image Path:", saved_image_path)

#     # Return image
#     return FileResponse(saved_image_path)





from fastapi import APIRouter, UploadFile, File
from fastapi.responses import FileResponse

from app.services.yolo_service import model
from app.utils.helper import create_folder

import shutil
import os

router = APIRouter()

# Create folders
create_folder("uploads")

@router.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):

    # Save uploaded image
    file_path = f"uploads/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run YOLO prediction
    results = model.predict(
        source=file_path,
        save=True
    )

    # Correct path handling
    saved_image_path = os.path.join(
        results[0].save_dir,
        file.filename
    )

    print("Saved Image Path:", saved_image_path)

    # Return predicted image
    return FileResponse(saved_image_path)

