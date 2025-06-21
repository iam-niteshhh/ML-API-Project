import os
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.image import resize
from tensorflow.keras.applications.imagenet_utils import preprocess_input

import constants


class ImageClassifier:
    """
    Loads a trained image classification model and predicts coarse categories for input images.
    """

    def __init__(self):
        self.model_path = constants.IMAGE_MODEL_LOAD_PATH
        self.label_path = constants.IMAGE_LABELS_LOAD_PATH
        self.model = None
        self.labels = None
        self.load_model()

    def load_model(self):
        """
        Loads the trained model and corresponding label list.
        """
        if not os.path.exists(self.model_path) or not os.path.exists(self.label_path):
            raise FileNotFoundError("Model or labels not found. Please train the model first.")

        self.model = load_model(self.model_path)
        self.labels = joblib.load(self.label_path)

    def preprocess(self, image):
        """
        Resizes and preprocesses an input image for prediction.
        """
        image = resize(image, [32, 32])  # CIFAR-100 image size
        image = preprocess_input(image)
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        return image

    def predict(self, image):
        """
        Predicts the coarse class label of the input image.
        """
        try:
            processed_image = self.preprocess(image)
            prediction = self.model.predict(processed_image)
            predicted_index = np.argmax(prediction)
            label = self.labels[predicted_index]
            confidence = float(np.max(prediction))

            return {
                "label": label,
                "confidence": confidence
            }
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")
