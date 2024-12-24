import json
from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer 

prefix = "normalize: "
tokenizer = AutoTokenizer.from_pretrained("t5-small")

def load_data(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    sources = []
    targets = []
    for item in data:
        sources.append(item["aphasic"])
        targets.append(item["original"])
    return Dataset.from_dict({"source": sources, "target": targets})

def preprocess_data(examples):
    inputs = [prefix + text for text in examples["source"]]
    targets = examples["target"]
    
    model_inputs = tokenizer(inputs, max_length=128, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True)
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
    


def main():
    dataset = load_data("aphasic_to_normal_synthetic.json")
    dataset = dataset.train_test_split(test_size=0.2)
    tokenized_dataset = dataset.map(preprocess_data, batched=True)

    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir="./models",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=1,
        fp16=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    trainer.train()


if __name__ == "__main__":
    main()