# %%
from transformers import (
    pipeline,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)
import evaluate
import numpy as np
from datasets import load_dataset
from datasets import DatasetDict


raw_dset = load_dataset("wiki")

checkpoint = "distilbert-base-cased"

label_list = raw_dset["train"].features["ner_tags"].feature.names


id2label = {i: label for i, label in enumerate(label_list)}
label2id = {label: i for i, label in enumerate(label_list)}

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForTokenClassification.from_pretrained(
    checkpoint, num_labels=len(label_list), id2label=id2label, label2id=label2id
)
print(raw_dset)


# conll-2003 Mapping:
# 0 : O-tag which means outside
# 1: B-PER   -> 2: I-PER
# 3: B-ORG   -> 4: I-ORG
# 5: B-LOC   -> 6: I-LOC
# 7: B-MISC  -> 8: I-MISC
B_TO_I = {1: 2, 3: 4, 5: 6, 7: 8}


def tokenize_fn(batch):
    tokenized_inputs = tokenizer(
        batch["tokens"], is_split_into_words=True, truncation=True
    )

    labels = []
    for i, old_labels in enumerate(batch["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None

        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != previous_word_idx:
                label_ids.append(old_labels[word_id])
            else:
                label = old_labels[word_id]
                label_ids.append(B_TO_I.get(label, label))

            previous_word_idx = word_id

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


tokenized_dataset = raw_dset.map(
    tokenize_fn,
    batched=True,
    remove_columns=raw_dset["train"].column_names,
)


# Load the metric
metric = evaluate.load("seqeval")

# Get the label names from the original dataset features
label_list = raw_dset["train"].features["ner_tags"].feature.names


def compute_metrics(logits_and_labels):
    logits, labels = logits_and_labels
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (-100) and convert to string labels
    true_labels = [[label_list[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


training_args = TrainingArguments(
    output_dir="distilbert-ner",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_steps=50,
)

data_collator = DataCollatorForTokenClassification(tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("ner_model")


nlp = pipeline("ner", model="ner_model", aggregation_strategy="simple")

results = nlp("hello world .")
