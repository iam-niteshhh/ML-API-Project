from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
from pydantic import BaseModel
import numpy as np
import io

from app.services.image_classifier import ImageClassifier

router = APIRouter()
classifier = ImageClassifier()


class ImageAPI:
    @staticmethod
    @router.post("/predict-image", tags=["Image"])
    async def predict_image(file: UploadFile = File(...)):
        """
        Accepts an image file and returns the predicted coarse class.
        """
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            image_array = np.array(image)

            result = classifier.predict(image_array)

            return {
                "input_file": file.filename,
                "prediction": result["label"],
                "confidence": result["confidence"]
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
