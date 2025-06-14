import re
import string
import nltk
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Custom Imports
import constants

# Download necessary NLTK resources
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt")
nltk.download("omw-1.4")

class TextProcessor(BaseEstimator, TransformerMixin):
    """
        Custom text preprocessing transformer for scikit-learn Pipeline.
        Includes:
          - Lowercasing
          - Removing digits and punctuation
          - Tokenizing
          - Stopword removal
          - Lemmatization
    """
    def __init__(self):
        self.stopwords = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text):
        text = text.lower()

        # Remove digits
        text = re.sub(constants.REGEX_WORDS_ONLY, " ", text)

        # remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        # tokenization
        tokens = word_tokenize(text)

        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stopwords]

        return " ".join(tokens)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self.clean_text(text) for text in X]

