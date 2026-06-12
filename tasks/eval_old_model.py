"""Compute a model's BLEU on the hash-split test set of the current dataset.

Usage: python tasks/eval_old_model.py [model_dir]
"""
import hashlib
import json
import sys

import torch
from evaluate import load
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

OLD_MODEL = sys.argv[1] if len(sys.argv) > 1 else "./models/model_2025-01-04_13-49-09/checkpoint-12145"
PREFIX = "normalize in English: "

with open("aphasic_to_normal_synthetic.json", encoding="utf-8") as f:
    data = json.load(f)
test = [item for item in data
        if int(hashlib.md5(item["original"].encode("utf-8")).hexdigest(), 16) % 10 == 0]
print(len(test), "test pairs")

device = torch.device("cuda")
tokenizer = AutoTokenizer.from_pretrained(OLD_MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(OLD_MODEL).to(device).eval()

preds = []
for i in range(0, len(test), 64):
    batch = [PREFIX + item["aphasic"] for item in test[i:i + 64]]
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                       max_length=128).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=128)
    preds.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))

metric = load("sacrebleu")
result = metric.compute(predictions=[p.strip() for p in preds],
                        references=[[item["original"].strip()] for item in test])
print(f"{OLD_MODEL} BLEU on current test set: {result['score']:.2f}")
