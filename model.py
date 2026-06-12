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

MODEL_NAME = "google/flan-t5-base"
PREFIX = "normalize in English: "
MAX_LENGTH = 128

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
metric = load("sacrebleu")


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

    result = metric.compute(
        predictions=[pred.strip() for pred in decoded_preds],
        references=[[label.strip()] for label in decoded_labels],
    )
    return {"bleu": result["score"]}


def main():
    dataset = load_data("aphasic_to_normal_synthetic.json")
    print({name: len(split) for name, split in dataset.items()})
    tokenized_dataset = dataset.map(preprocess_data, batched=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_save_path = f"./models/model_{timestamp}"

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=model_save_path,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-4,
        warmup_ratio=0.05,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=6,
        bf16=True,
        predict_with_generate=True,
        generation_max_length=MAX_LENGTH,
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        logging_steps=100,
        report_to=[],
        seed=42,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    final_metrics = trainer.evaluate()
    print(f"Final eval (best checkpoint): {final_metrics}")

    final_path = f"{model_save_path}/final"
    trainer.save_model(final_path)
    print(f"Saved best model to {final_path}")


if __name__ == "__main__":
    main()
