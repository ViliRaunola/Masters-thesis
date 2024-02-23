import pprint

import datasets
import evaluate
import models.tagger as tagger
import numpy as np
from transformers import TrainingArguments

metric = evaluate.load("seqeval")

my_data_val = tagger._read_coll_file("../own_data/results_annotated_from_testers.tsv")
lables_to_ids = tagger._create_label_mapping_dics()[0]
my_dataset = datasets.Dataset.from_pandas(my_data_val)

pipeline = tagger.get_ner_pipeline()
training_args = TrainingArguments("test_trainer")

task_evaluator = evaluate.evaluator("token-classification")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


results = task_evaluator.compute(
    model_or_pipeline=pipeline,
    data=my_dataset,
    metric=metric,
    input_column="tokens",
    label_column="labels",
)

for key in results:
    print(f"For {key}: {results[key]}")
