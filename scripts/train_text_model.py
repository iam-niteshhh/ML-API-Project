import os
import joblib
import nltk
from datasets import load_dataset
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

# Custom Imports
import constants
from app.utils.text import TextProcessor

class TextClassifierTrainer:
    """
       Class to train and save a text classification model using:
       - GoEmotions dataseta
       - Preprocessing pipeline
       - Logistic Regression
       - Joblib for model export
   """
    def __init__(self):
        self.model_path = constants.TEXT_MODEL_STORE_PATH
        self.labels_path = constants.TEXT_LABELS_STORE_PATH
        self.pipeline = Pipeline([
            ('preprocess', TextProcessor()),
            ("tfidf", TfidfVectorizer(max_features=10000)),
            ("clf", LogisticRegression(max_iter=1000))
        ])
        self.id2label = None

    def load_data(self):
        """
           Loads GoEmotions dataset and filters to single-label samples.
           Returns a pandas DataFrame with text and string labels.
        """
        dataset = load_dataset(constants.TEXT_DATASET_NAME, constants.TEXT_DATASET_VERSION, split="train")
        print("GoEmotions dataset loaded")
        dataset = dataset.filter(lambda x: len(x["labels"]) == 1)
        print("GoEmotions dataset filtered")
        dataframe = dataset.to_pandas()

        #extraxt single label
        dataframe["label"] = dataframe["labels"].apply(lambda x: x[0])
        dataframe.drop(columns=["labels","id"], inplace=True)

        #map id->label
        self.id2label = dataset.features["labels"].feature.int2str

        # Convert to label strings
        dataframe["label"] = dataframe["label"].apply(self.id2label)

        return dataframe

    def train_and_evaluate(self, dataframe):
        """
            Splits the dataset, trains the model, and prints classification report.
            Returns the list of unique class labels.
        """
        X_train, X_test, y_train, y_test = train_test_split(dataframe["text"], dataframe["label"], test_size=0.2, random_state=42)
        self.pipeline.fit(X_train, y_train)
        y_pred = self.pipeline.predict(X_test)
        print("Classification report")
        print(classification_report(y_test, y_pred))


        print("Confusion matrix")
        print(confusion_matrix(y_test, y_pred))

        return sorted(dataframe["label"].unique())

    def save_model(self, label_list):
        """
           Saves the trained model and label list to disk.
        """
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.pipeline, self.model_path)
        joblib.dump(label_list, self.labels_path)
        print(f"> Model saved to {self.model_path}")
        print(f"> Labels saved to {self.labels_path}")

    def run(self):
        """
            Main entry point to execute the full training pipeline.
        """
        print("1. Loading data...")
        df = self.load_data()
        print("2. Training model...")
        label_list = self.train_and_evaluate(df)
        print("3. Saving model...")
        self.save_model(label_list)



if __name__ == "__main__":
    trainer = TextClassifierTrainer()
    trainer.run()