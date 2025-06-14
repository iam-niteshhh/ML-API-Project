from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.text_classifier import TextClassifier

router = APIRouter()
classifier = TextClassifier()


class TextRequest(BaseModel):
    text: str


class TextAPI:
    @staticmethod
    @router.post("/predict-text", tags=["Text"])
    def predict_text(request: TextRequest):
        try:
            result = classifier.predict(request.text)
            return {
                "input": request.text,
                "prediction": result["label"],
                "confidence": result["confidence"]
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")