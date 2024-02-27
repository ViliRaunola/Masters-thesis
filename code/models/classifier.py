import os
from typing import Type

import datasets
import pandas as pd
import torch
import transformers
import utility.common as common
import utility.globals as globals
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


################# Public Functions #################
def create_finbert():
    """
    Return 1 for failed attempt, 0 for success
    """

    try:
        if len(os.listdir(globals.REPO_NAME_CLASS)) != 0:
            print(
                f"{common.colors.CYELLOW} The folder {globals.MODEL_FOLDER_CLASS} is not empty. Please clear it before training the model {common.colors.CEND}"
            )
            return 1
    except FileNotFoundError:
        _create_folder_for_model()

    tokenizer = load_tokenizer()
    dataset_fin_sentiment = _prepare_fin_sentiment()
    data_sets_splits = _split_dataset(dataset_fin_sentiment)

    tokenized_dataset = _tokenize_dataset(data_sets_splits, tokenizer)

    data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = _train_fin_bert(
        tokenized_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        repo_name=globals.REPO_NAME_CLASS,
    )
    trainer.save_model(globals.REPO_NAME_CLASS)

    _test_trained_model(trainer, tokenized_dataset)
    print(
        f"{common.colors.CGREEN}Creating the FinBERT model with FinSentiment data has been succesfull. The model has been saved to {globals.REPO_NAME_CLASS}{common.colors.CEND}"
    )
    return 0


def get_sentiment_pipeline():
    if os.path.isdir(globals.REPO_NAME_CLASS):
        if not os.listdir(globals.REPO_NAME_CLASS):
            print(
                f"{common.colors.CYELLOW}The folder: {globals.REPO_NAME_CLASS} is empty. This suggests that the model has not been trained yet. Please tain it before accessing it.{common.colors.CEND}"
            )
            return None
        else:
            sentiment_model = load_model()
            tokenizer = load_tokenizer()

            sentiment_pipeline = transformers.pipeline(
                task="text-classification", model=sentiment_model, tokenizer=tokenizer
            )
            return sentiment_pipeline
    else:
        print(
            f"{common.colors.CYELLOW}The folder: {globals.REPO_NAME_CLASS} doesn't exist. This suggests that the model has not been trained yet. Please tain it before accessing it.{common.colors.CEND}"
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


def load_tokenizer():
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "TurkuNLP/bert-base-finnish-cased-v1"
    )

    return tokenizer


def tokenize_long_text(text: str, tokenizer: Type[transformers.AutoTokenizer]):
    tokens = tokenizer.encode_plus(text, add_special_tokens=False, return_tensors="pt")
    print(tokens)
    return tokens


def _read_fin_sentiment():
    fin_sentiment = pd.read_csv(
        "../data/finsen-v1-1-src/FinnSentiment-1.1.tsv",
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

    # Source https://huggingface.co/docs/transformers/main_classes/callback
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
    print(f"Creating a new folder for the model: {globals.MODEL_FOLDER_CLASS}")
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, globals.MODEL_FOLDER_CLASS)
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


def load_model():
    id2label = {0: "neg", 1: "neut", 2: "pos"}
    sentiment_model = transformers.AutoModelForSequenceClassification.from_pretrained(
        globals.REPO_NAME_CLASS, id2label=id2label
    )
    return sentiment_model


def get_long_text_classifier():
    return _long_text_classifier


def _long_text_classifier(text: str):

    tokenizer = load_tokenizer()
    tokens = tokenize_long_text(text, tokenizer)

    input_id_chunks = tokens["input_ids"][0].split(510)
    mask_chunks = tokens["attention_mask"][0].split(510)

    chunksize = 512

    input_id_chunks = list(input_id_chunks)
    mask_chunks = list(mask_chunks)

    input_ids, attention_mask = _split_tokens_into_chunks(
        input_id_chunks, mask_chunks, chunksize
    )

    input_dict = {"input_ids": input_ids.long(), "attention_mask": attention_mask.int()}

    model = load_model()

    outputs = model(**input_dict)
    probs = torch.nn.functional.softmax(outputs[0], dim=-1)
    mean = probs.mean(dim=0)

    label_id = torch.argmax(mean).item()
    id2label = {0: "neg", 1: "neut", 2: "pos"}

    return {"label": f"{id2label[label_id]}", "score": mean.detach().numpy()[label_id]}


# Source: https://www.youtube.com/watch?v=yDGo9z_RlnE
def _split_tokens_into_chunks(input_id_chunks: list, mask_chunks: list, chunksize: int):
    #!The CLC is 102 and SEP is 103!!!
    for i in range(len(input_id_chunks)):
        input_id_chunks[i] = torch.cat(
            [torch.Tensor([102]), input_id_chunks[i], torch.Tensor([103])]
        )

        mask_chunks[i] = torch.cat(
            [torch.Tensor([1]), mask_chunks[i], torch.Tensor([1])]
        )

        # Making sure that if it is less than 512 it gets filled with 0s
        pad_len = chunksize - input_id_chunks[i].shape[0]
        if pad_len > 0:
            input_id_chunks[i] = torch.cat(
                [input_id_chunks[i], torch.Tensor([0] * pad_len)]
            )
            mask_chunks[i] = torch.cat([mask_chunks[i], torch.Tensor([0] * pad_len)])

    input_ids = torch.stack(input_id_chunks)
    attention_mask = torch.stack(mask_chunks)

    return input_ids, attention_mask
