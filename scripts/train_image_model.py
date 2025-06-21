import os
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import constants


class ImageTrainer:
    """
    Trains a CNN image classifier on CIFAR-100 and maps predictions to coarse categories dynamically.
    """

    def __init__(self):
        self.model_path = constants.IMAGE_MODEL_STORE_PATH
        self.label_path = constants.IMAGE_LABEL_STORE_PATH
        self.model = None
        self.callbacks = None
        self.coarse_label_mapping = constants.COARSE_LABEL_MAPPING

    def loads_data(self):
        """
        Loads CIFAR-100 dataset and dynamically maps fine labels to coarse categories.
        Returns normalized image arrays and one-hot encoded coarse labels.
        """
        # Load CIFAR-100 with coarse labels
        (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='coarse')

        # Normalize images between 0 and 1
        x_train, x_test = x_train / 255.0, x_test / 255.0

        # One-hot encode the labels (20 coarse classes)
        y_train_encoded = to_categorical(y_train, num_classes=len(self.coarse_label_mapping))
        y_test_encoded = to_categorical(y_test, num_classes=len(self.coarse_label_mapping))

        return x_train, y_train_encoded, x_test, y_test_encoded

    def build_model_and_callbacks(self):
        """
        Builds a simple CNN model for image classification.
        """
        self.model = Sequential([
            Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            MaxPooling2D(2, 2),
            Dropout(0.25),

            Conv2D(64, (3, 3), activation='relu', padding='same'),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            MaxPooling2D(2, 2),
            Dropout(0.25),

            Conv2D(128, (3, 3), activation='relu', padding='same'),
            MaxPooling2D(2, 2),
            Dropout(0.25),

            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(len(self.coarse_label_mapping), activation='softmax')
        ])

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Training callbacks
        self.callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
            ModelCheckpoint(self.model_path, save_best_only=True)
        ]

    def train(self,x_train, x_test, y_train, y_test, epochs=15, batch_size=64):
        """
        Loads data, builds the model, trains it, and saves the model and label map.
        """
        if any(v is None for v in [x_train, x_test, y_train, y_test]):
            raise ValueError("All x_train, x_test, y_train, and y_test must be provided.")

        print("Starting training...")
        self.model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test, y_test),
            callbacks=self.callbacks
        )

    def save_model(self):
        """
           Saves the trained model and label list to disk.
        """

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        # Save trained model and labels
        self.model.save(self.model_path)
        joblib.dump(self.coarse_label_mapping, self.label_path)
        print(f"> Model saved to {self.model_path}")
        print(f"> Coarse label list saved to {self.label_path}")

    def check_gpu(self):
        """
        Checks if a GPU is available for TensorFlow and enables memory growth.
        Prints GPU details if available; otherwise notifies running on CPU.
        """
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"GPUs detected: {len(gpus)}")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu}")
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("Memory growth enabled for GPUs.")
            except RuntimeError as e:
                # Must be set before GPUs are initialized
                print(f"Could not set memory growth: {e}")
        else:
            print("No GPU detected, running on CPU.")

    def run(self):
        """
            Main entry point to execute the full training pipeline.
        """
        print("0. Checking for GPU availability...")
        print("0.0 Skipping Check for GPU availability...")
        # self.check_gpu()

        print("1. Loading data...")
        x_train, y_train, x_test, y_test = self.loads_data()

        print("2. Building model architecture...")
        self.build_model_and_callbacks()

        print("3. Training model...")
        self.train(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            epochs=30,
            batch_size=64
        )

        print("4. Saving model...")
        self.save_model()


if __name__ == "__main__":
    trainer = ImageTrainer()
    trainer.run()