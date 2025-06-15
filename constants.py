REGEX_WORDS_ONLY = r"\d+"
TEXT_MODEL_STORE_PATH = "../app/models/text_classifier.pkl"
TEXT_LABELS_STORE_PATH = "../app/models/text_labels.pkl"
TEXT_MODEL_LOAD_PATH = "./app/models/text_classifier.pkl"
TEXT_LABELS_LOAD_PATH = "./app/models/text_labels.pkl"
TEXT_DATASET_NAME = "google-research-datasets/go_emotions"
TEXT_DATASET_VERSION = "simplified"

IMAGE_MODEL_STORE_PATH = "./app/models/image_classifier.h5"
IMAGE_LABEL_STORE_PATH = "./app/models/image_labels.pkl"

COARSE_LABEL_MAPPING = {
    0: "aquatic mammals",
    1: "fish",
    2: "flowers",
    3: "food containers",
    4: "fruit and vegetables",
    5: "household electrical devices",
    6: "household furniture",
    7: "insects",
    8: "large carnivores",
    9: "large man-made outdoor things",
    10: "large natural outdoor scenes",
    11: "large omnivores and herbivores",
    12: "medium-sized mammals",
    13: "non-insect invertebrates",
    14: "people",
    15: "reptiles",
    16: "small mammals",
    17: "trees",
    18: "vehicles 1",
    19: "vehicles 2"
}
