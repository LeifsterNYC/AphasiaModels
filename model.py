import argparse
import hashlib
import json
from datetime import datetime

import numpy as np
from datasets import Dataset, DatasetDict
from evaluate import load
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from normalize import PREFIX

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="google/flan-t5-large")
parser.add_argument("--epochs", type=float, default=3)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--batch", type=int, default=32)
ARGS = parser.parse_args()

MODEL_NAME = ARGS.model
MAX_LENGTH = 96
# Generating for the whole held-out split every epoch dominates training
# time on the large model; a fixed subset is enough to pick a checkpoint.
EVAL_SUBSET = 3000

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bleu = load("sacrebleu")
chrf = load("chrf")


def load_data(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    # Split by hashing the target sentence: the corpus contains duplicate
    # lines and each line yields multiple corruption variants, so a random
    # pair-level split would leak eval targets into training.
    splits = {"train": {"source": [], "target": []},
              "test": {"source": [], "target": []}}
    for item in data:
        digest = hashlib.md5(item["original"].encode("utf-8")).hexdigest()
        split = "test" if int(digest, 16) % 10 == 0 else "train"
        splits[split]["source"].append(item["aphasic"])
        splits[split]["target"].append(item["original"])
    return DatasetDict({name: Dataset.from_dict(cols) for name, cols in splits.items()})


def preprocess_data(examples):
    inputs = [PREFIX + text for text in examples["source"]]
    targets = examples["target"]

    model_inputs = tokenizer(inputs, max_length=MAX_LENGTH, truncation=True)
    labels = tokenizer(text_target=targets, max_length=MAX_LENGTH, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # -100 is the ignore index used for padding in labels; the tokenizer
    # cannot decode it.
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    preds = [pred.strip() for pred in decoded_preds]
    refs = [[label.strip()] for label in decoded_labels]
    return {"bleu": bleu.compute(predictions=preds, references=refs)["score"],
            "chrf": chrf.compute(predictions=preds, references=refs)["score"]}


def main():
    dataset = load_data("aphasic_to_normal_synthetic.json")
    print({name: len(split) for name, split in dataset.items()})
    tokenized_dataset = dataset.map(preprocess_data, batched=True)
    eval_dataset = tokenized_dataset["test"].shuffle(seed=0).select(
        range(min(EVAL_SUBSET, len(tokenized_dataset["test"]))))
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_save_path = f"./models/model_{timestamp}_{MODEL_NAME.split('/')[-1]}"

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=model_save_path,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=ARGS.lr,
        warmup_steps=0.05,  # a float below 1 is a ratio of total steps
        per_device_train_batch_size=ARGS.batch,
        per_device_eval_batch_size=64,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=ARGS.epochs,
        bf16=True,
        predict_with_generate=True,
        generation_max_length=MAX_LENGTH,
        load_best_model_at_end=True,
        metric_for_best_model="chrf",
        greater_is_better=True,
        logging_steps=500,
        report_to=[],
        seed=42,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    final_metrics = trainer.evaluate()
    print(f"Final eval (best checkpoint): {final_metrics}")

    final_path = f"{model_save_path}/final"
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Saved best model to {final_path}")


if __name__ == "__main__":
    main()
