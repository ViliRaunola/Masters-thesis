import os

import datasets
import numpy as np
import pandas as pd
import torch
import transformers
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import common

# Private globals
_REPO_NAME = "./model"
_MODEL_FOLDER = "model"


################# Public Functions #################
def create_finbert():
    """
    Return 1 for failed attempt, 0 for success
    """

    try:
        if len(os.listdir(_MODEL_FOLDER)) != 0:
            print(
                f"{common.colors.CYELLOW} The folder {_MODEL_FOLDER} is not empty. Please clear it before training the model {common.colors.CEND}"
            )
            return 1
    except FileNotFoundError:
        _create_folder_for_model()

    tokenizer = _load_tokenizer()
    dataset_fin_sentiment = _prepare_fin_sentiment()
    data_sets_splits = _split_dataset(dataset_fin_sentiment)

    #!TODO needed?
    # np.unique(data_sets_splits["train"]["label"])
    # print(np.unique(data_sets_splits["train"]["label"]))

    #!TODO can be removed, jsut for checking how the data looks
    for i in range(10):
        print("text:", data_sets_splits["train"]["text"][i])
        print("label:", data_sets_splits["train"]["label"][i])

    tokenized_dataset = _tokenize_dataset(data_sets_splits, tokenizer)

    data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = _train_fin_bert(
        tokenized_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        repo_name=_REPO_NAME,
    )
    trainer.save_model(_REPO_NAME)

    _test_trained_model(trainer, tokenized_dataset)
    print(
        f"{common.colors.CGREEN}Creating the FinBERT model with FinSentiment data has been succesfull. The model has been saved to {_REPO_NAME}{common.colors.CEND}"
    )
    return 0


def get_sentiment_pipeline():
    if os.path.isdir(_REPO_NAME):
        if not os.listdir(_REPO_NAME):
            print(
                f"{common.colors.CYELLOW}The folder: {_REPO_NAME} is empty. This suggests that the model has not been trained yet. Please tain it before accessing it.{common.colors.CEND}"
            )
            return None
        else:
            sentiment_model = _load_model()
            tokenizer = _load_tokenizer()

            sentiment_pipeline = transformers.pipeline(
                task="text-classification", model=sentiment_model, tokenizer=tokenizer
            )
            return sentiment_pipeline
    else:
        print(
            f"{common.colors.CYELLOW}The folder: {_REPO_NAME} doesn't exist. This suggests that the model has not been trained yet. Please tain it before accessing it.{common.colors.CEND}"
        )
        return None


################# Private Functions #################
def _load_fin_bert():
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        "TurkuNLP/bert-base-finnish-cased-v1", num_labels=3
    )
    model.eval()
    if torch.cuda.is_available():
        print("Cuda is available, using it for the model.")
        model = model.cuda()
    return model


def _load_tokenizer():
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "TurkuNLP/bert-base-finnish-cased-v1"
    )
    return tokenizer


def _read_fin_sentiment():
    fin_sentiment = pd.read_csv(
        "data/finsen-v1-1-src/FinnSentiment-1.1.tsv",
        sep="\t",
        header=None,
        usecols=[*range(0, 5), *range(15, 20)],
        index_col=False,
    )
    return fin_sentiment


def _rename_fin_sentiment_columns(fin_sentiment):
    renaming_dic = {
        0: "A sentiment",
        1: "B sentiment",
        2: "C sentiment",
        3: "majority value",
        4: "derived values",
        14: "pre-annotated sentiment smiley",
        15: "pre-annotated sentiment product review",
        16: "split #",
        17: "batch #",
        18: "index in original corpus",
        19: "text",
    }

    fin_sentiment = fin_sentiment.rename(columns=renaming_dic)
    return fin_sentiment


def _map_sentiment_values_to_fin_sentiment(fin_sentiment, class_names):
    mapping = {
        1: class_names[0],
        2: class_names[0],
        3: class_names[1],
        4: class_names[2],
        5: class_names[2],
    }
    # Adding a new column called 'sentiment'. Value based on the derived value.
    fin_sentiment["label"] = fin_sentiment["derived values"].map(mapping)
    return fin_sentiment


def _create_dataset_from_fin_sentiment(fin_sentiment, class_names):
    dataset_fin_sentiment = datasets.Dataset.from_pandas(
        fin_sentiment[["text", "label"]],
        features=datasets.Features(
            {
                "text": datasets.Value("string"),
                "label": datasets.ClassLabel(names=class_names),
            }
        ),
        preserve_index=False,
    )
    return dataset_fin_sentiment


def _prepare_fin_sentiment():
    class_names = ["neg", "neut", "pos"]

    print("Preparing the FinSentiment dataset...")

    fin_sentiment = _read_fin_sentiment()
    fin_sentiment = _rename_fin_sentiment_columns(fin_sentiment)
    fin_sentiment = _map_sentiment_values_to_fin_sentiment(fin_sentiment, class_names)
    dataset_fin_sentiment = _create_dataset_from_fin_sentiment(
        fin_sentiment, class_names
    )

    return dataset_fin_sentiment


def _split_dataset(dataset_fin_sentiment):
    first_split = 0.2
    second_split = 0.5

    print("Splitting the dataset...")

    # split the data, https://stackoverflow.com/questions/76001128/splitting-dataset-into-train-test-and-validation-using-huggingface-datasets-fun
    train_dataset = dataset_fin_sentiment.train_test_split(
        test_size=first_split, seed=42, stratify_by_column="label"
    )
    test_dataset = dataset_fin_sentiment.train_test_split(
        test_size=second_split, seed=42, stratify_by_column="label"
    )

    data_sets_splits = datasets.DatasetDict(
        {
            "train": train_dataset["train"],
            "valid": test_dataset["train"],
            "test": test_dataset["test"],
        }
    )

    print(
        f"The data has been split to training: {1-first_split}, validation: {first_split*second_split} and testing: {first_split*second_split}"
    )
    print(data_sets_splits)

    return data_sets_splits


def _preprocess_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True)


def _tokenize_dataset(data_sets_splits, tokenizer):

    print("Tokenizing the dataset...")

    tokenized_dataset = data_sets_splits.map(
        lambda n: _preprocess_function(n, tokenizer), batched=True
    )
    return tokenized_dataset


def _train_fin_bert(repo_name, tokenized_dataset, tokenizer, data_collator):

    model = _load_fin_bert()

    epochs = 2
    learning_rate = 2e-5
    batch_size = 8
    weight_decay = 0.01

    # Source https://medium.com/@rakeshrajpurohit/customized-evaluation-metrics-with-hugging-face-trainer-3ff00d936f99
    training_args = transformers.TrainingArguments(
        output_dir=repo_name,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        push_to_hub=False,
        metric_for_best_model="f1",
        load_best_model_at_end=True,
        logging_dir="./logs",
    )

    #!TODO selvitä lisää tästä
    early_stopping = transformers.EarlyStoppingCallback(early_stopping_patience=2)

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["valid"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=_compute_metrics,
        callbacks=[early_stopping],
    )

    print(
        "Starting to train the model.\n"
        + "Parameters for training are:\n"
        + f"epochs: {epochs}\nlearning rate: {learning_rate}\nbatch size: {batch_size}\nweight_decay: {weight_decay}\n"
        "This takes ~30 min using RTX 2060 ...",
    )

    trainer.train()
    print(trainer.evaluate())
    return trainer


def _test_trained_model(trainer, tokenized_dataset):
    test_results = trainer.predict(test_dataset=tokenized_dataset["test"])
    print(test_results.metrics)


def _create_folder_for_model():
    print(f"Creating a new folder for the model: {_MODEL_FOLDER}")
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, _MODEL_FOLDER)
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)


# Source: https://medium.com/@rakeshrajpurohit/customized-evaluation-metrics-with-hugging-face-trainer-3ff00d936f99
def _compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    # Calculate accuracy
    accuracy = accuracy_score(labels, preds)

    # Calculate precision, recall, and F1-score
    precision = precision_score(labels, preds, average="weighted")
    recall = recall_score(labels, preds, average="weighted")
    f1 = f1_score(labels, preds, average="weighted")

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def _load_model():
    id2label = {0: "neg", 1: "neut", 2: "pos"}
    sentiment_model = transformers.AutoModelForSequenceClassification.from_pretrained(
        _REPO_NAME, id2label=id2label
    )
    return sentiment_model
