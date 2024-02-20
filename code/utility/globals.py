import sys

import datasets

gettrace = getattr(sys, "gettrace", None)
REPO_NAME_CLASS = "./models/model"
REPO_NAME_NER = "./models/model_ner"
MODEL_FOLDER_CLASS = "/models/model"
MODEL_FOLDER_NER = "/models/model_ner"

if gettrace is None:
    print("no trace")
elif gettrace():
    REPO_NAME_CLASS = "./code/models/model"
    REPO_NAME_NER = "./code/models/model_ner"
    MODEL_FOLDER_CLASS = "/code/models/model"
    MODEL_FOLDER_NER = "/models/model_ner"


CLASSIFIER_LABELS = ["neg", "neut", "pos"]

MENU_BORDER_SIZE = 40

LABEL_LIST = [
    "O",
    "B-PERSON",
    "I-PERSON",
    "B-LOC",
    "I-LOC",
    "B-ORG",
    "I-ORG",
    "B-PRODUCT",
    "I-PRODUCT",
    "B-EVENT",
    "I-EVENT",
    "B-DATE",
    "I-DATE",
    "B-GPE",
    "I-GPE",
]

# Selecting the metric to use
METRIC = datasets.load_metric("seqeval", trust_remote_code=True)
