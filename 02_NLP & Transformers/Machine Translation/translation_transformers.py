import evaluate
import numpy as np
from datasets import Dataset, load_dataset
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, PreTrainedTokenizer,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments, pipeline)

bleu = evaluate.load("sacrebleu")




streamed_dset = load_dataset(
    "quickmt/quickmt-train.ar-en", split="train", streaming=True
)


dset = streamed_dset.take(1000)  # return iterable
dset = Dataset.from_list(list(dset)).train_test_split(test_size=0.2)



checkpoint = "Helsinki-NLP/opus-mt-en-ar"

tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)


def tokenize_fn(batch):
    input = batch["en"]
    target = batch["ar"]

    tokenized_input = tokenizer(input, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(target, truncation=True)

    tokenized_input["labels"] = labels["input_ids"]
    return tokenized_input


print(dset)

tokenized_dset = dset.map(
    tokenize_fn, batched=True, remove_columns=dset["train"].column_names
)


def compute_metrics(logits_and_labels):
    pred, labels = logits_and_labels

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    output = tokenizer.batch_decode(pred, skip_special_tokens=True)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # because BLEU expects:
    # predictions: list[str]
    # references: list[list[str]]
    # labels = [[l] for l in labels]  # sacrebleu format

    print("output ", output)
    score = bleu.compute(predictions=output, references=labels)
    return score


training_args = Seq2SeqTrainingArguments(
    "translation_model",
    save_strategy="epoch",
    eval_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    num_train_epochs=3,
    predict_with_generate=True,
    save_total_limit=2,
)


trainer = Seq2SeqTrainer(
    model,
    training_args,
    data_collator,
    tokenized_dset["train"],
    tokenized_dset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)


trainer.train()

model = pipeline("translation", "translation_model/checkpoint-200")


model("fake")


