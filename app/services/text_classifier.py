import joblib
import os
from typing import List

# Custom Imports
import constants
from app.utils.text import TextProcessor


class TextClassifier:
    def __init__(self):
        model_path = constants.MODEL_LOAD_PATH
        labels_path = constants.MODEL_LOAD_PATH
        print("Loading model...", model_path)
        print("Loading labels...", labels_path)

        if not os.path.exists(model_path) and not os.path.exists(labels_path):
            raise FileNotFoundError("Model or labels not found. Please train the model first.")
        print("LOAD COMPLETE")
        self.model = joblib.load(model_path)
        self.labels: List[str] = joblib.load(labels_path)

    def predict(self, text: str) -> dict:
        """
           Predict the emotion label for a given input text.
           Returns a dict with predicted label and confidence score.
        """
        probs = self.model.predict_proba(text)[0]
        top_idx = probs.argmax()
        return {
            "label": self.model.classes_[top_idx],
            "confidence": round(probs[top_idx], 4)
        }


