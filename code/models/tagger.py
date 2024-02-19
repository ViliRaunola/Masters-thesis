import itertools
import os

import datasets
import numpy as np
import pandas as pd
import torch
import transformers
import utility.common as common

# Private globals
_REPO_NAME = "./models/model_ner"
_MODEL_FOLDER = "/models/model_ner"

_LABEL_LIST = [
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


################# Public Functions #################
def create_ner_finbert():
    """
    Return 1 for failed attempt, 0 for success
    """
    try:
        if len(os.listdir(_MODEL_FOLDER)) != 0:
            print(
                f"{common.colors.CYELLOW}The folder {_MODEL_FOLDER} is not empty. Please clear it before training the model{common.colors.CEND}"
            )
            return 1
    except FileNotFoundError:
        _create_folder_for_model()

    tokenizer = _load_tokenizer()
    model = _load_fin_bert()

    train_dataset, dev_dataset, test_dataset = _prepare_data()

    train_tokenized_datasets, dev_tokenized_datasets, test_tokenized_datasets = (
        _tokenize_data(train_dataset, dev_dataset, test_dataset, tokenizer)
    )

    data_collator = transformers.DataCollatorForTokenClassification(tokenizer)

    trainer = _train_model(
        model=model,
        train_tokenized_datasets=train_tokenized_datasets,
        dev_tokenized_datasets=dev_tokenized_datasets,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Saving the model for later use
    trainer.save_model("./model_ner")

    _test_model(trainer, test_tokenized_datasets)
    print(
        f"{common.colors.CGREEN}Creating the FinBERT model with turku-one data has been succesfull. The model has been saved to {_REPO_NAME}{common.colors.CEND}"
    )
    return 0


def get_ner_pipeline():
    if os.path.isdir(_REPO_NAME):
        if not os.listdir(_REPO_NAME):
            print(
                f"{common.colors.CYELLOW}The folder: {_REPO_NAME} is empty. This suggests that the model has not been trained yet. Please tain it before accessing it.{common.colors.CEND}"
            )
            return None
        else:
            ner_model = _load_model()
            tokenizer = _load_tokenizer()

            sentiment_pipeline = transformers.pipeline(
                task="token-classification", model=ner_model, tokenizer=tokenizer
            )
            return sentiment_pipeline
    else:
        print(
            f"{common.colors.CYELLOW}The folder: {_REPO_NAME} doesn't exist. This suggests that the model has not been trained yet. Please tain it before accessing it.{common.colors.CEND}"
        )
        return None


################# Private Functions #################


def _load_model():
    ids_to_labels = _create_label_mapping_dics()[1]

    ner_model = transformers.AutoModelForTokenClassification.from_pretrained(
        _REPO_NAME, id2label=ids_to_labels
    )

    return ner_model


def _create_folder_for_model():
    print(f"Creating a new folder for the model: {_MODEL_FOLDER}")
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, _MODEL_FOLDER)
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)


def _tokenize_data(train_dataset, dev_dataset, test_dataset, tokenizer):
    train_tokenized_datasets = train_dataset.map(
        lambda n: _tokenize_and_align_labels(n, tokenizer=tokenizer), batched=True
    )
    dev_tokenized_datasets = dev_dataset.map(
        lambda n: _tokenize_and_align_labels(n, tokenizer=tokenizer), batched=True
    )
    test_tokenized_datasets = test_dataset.map(
        lambda n: _tokenize_and_align_labels(n, tokenizer=tokenizer), batched=True
    )

    return (train_tokenized_datasets, dev_tokenized_datasets, test_tokenized_datasets)


def _prepare_data():
    # Load data from files
    turku_one_data_train = _read_coll_file("../data_ner/data/conll/train.tsv")
    turku_one_data_dev = _read_coll_file("../data_ner/data/conll/dev.tsv")
    turku_one_data_test = _read_coll_file("../data_ner/data/conll/test.tsv")

    lables_to_ids = _create_label_mapping_dics()[0]

    # Change the labels to corresponding ids
    turku_one_data_train = _swap_labels_to_ids(turku_one_data_train, lables_to_ids)
    turku_one_data_dev = _swap_labels_to_ids(turku_one_data_dev, lables_to_ids)
    turku_one_data_test = _swap_labels_to_ids(turku_one_data_test, lables_to_ids)

    # Modifying the dataframes to datasets
    train_dataset = datasets.Dataset.from_pandas(turku_one_data_train)
    dev_dataset = datasets.Dataset.from_pandas(turku_one_data_dev)
    test_dataset = datasets.Dataset.from_pandas(turku_one_data_test)

    return (train_dataset, dev_dataset, test_dataset)


#!TODO source
def _test_model(trainer, test_tokenized_datasets):
    predictions, labels, metrics = trainer.predict(test_tokenized_datasets)

    print(metrics)

    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [_LABEL_LIST[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [_LABEL_LIST[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = METRIC.compute(predictions=true_predictions, references=true_labels)
    print(results)


def _train_model(
    model,
    train_tokenized_datasets,
    dev_tokenized_datasets,
    data_collator,
    tokenizer,
):

    epochs = 3
    learning_rate = 2e-5
    batch_size = 8
    weight_decay = 0.01

    args = transformers.TrainingArguments(
        output_dir=_REPO_NAME,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        push_to_hub=False,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir="./logs",
    )

    #!TODO selvitä lisää tästä
    early_stopping = transformers.EarlyStoppingCallback(early_stopping_patience=2)

    trainer = transformers.Trainer(
        model,
        args,
        train_dataset=train_tokenized_datasets,
        eval_dataset=dev_tokenized_datasets,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=_compute_metrics,
        callbacks=[early_stopping],
    )

    trainer.train()
    print(trainer.evaluate())
    return trainer


def _compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    # Remove ignored index (special tokens)
    true_predictions = [
        [_LABEL_LIST[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [_LABEL_LIST[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = METRIC.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def _load_fin_bert():
    model = transformers.AutoModelForTokenClassification.from_pretrained(
        "TurkuNLP/bert-base-finnish-cased-v1", num_labels=len(_LABEL_LIST)
    )
    if torch.cuda.is_available():
        print("Cuda is available, using it for the model.")
        model = model.cuda()
    return model


# Source: https://github.com/huggingface/notebooks/blob/main/examples/token_classification.ipynb
def _tokenize_and_align_labels(examples, tokenizer):
    label_all_tokens = True
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )

    labels = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def _swap_labels_to_ids(data_frame, lables_to_ids):
    for index, row in data_frame.iterrows():
        for index, label in enumerate(row["labels"]):
            row["labels"][index] = lables_to_ids.get(label, 0)
    return data_frame


def _create_label_mapping_dics():

    lables_to_ids = {}
    ids_to_labels = {}

    for index, label in enumerate(_LABEL_LIST):
        temp = {label: index}
        lables_to_ids.update(temp)

    for index, label in enumerate(_LABEL_LIST):
        temp = {index: label}
        ids_to_labels.update(temp)

    return (lables_to_ids, ids_to_labels)


def _read_coll_file(filename):
    with open(filename, "r", encoding="utf8") as f:
        lines = f.readlines()
        split_list = [
            list(y) for x, y in itertools.groupby(lines, lambda z: z == "\n") if not x
        ]
        tokens = [[x.split("\t")[0] for x in y] for y in split_list]
        entities = [[x.split("\t")[1][:-1] for x in y] for y in split_list]
    return pd.DataFrame({"tokens": tokens, "labels": entities})


def _load_tokenizer():
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "TurkuNLP/bert-base-finnish-cased-v1"
    )
    return tokenizer