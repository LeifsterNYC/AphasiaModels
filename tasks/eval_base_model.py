"""Zero-shot google/flan-t5-base on the same test set and hand-written inputs."""
import hashlib
import json

import torch
from evaluate import load
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL = "google/flan-t5-base"
PREFIX = "normalize in English: "

HAND_WRITTEN = [
    "want water",
    "yesterday go store buy bread",
    "not like hospital food",
    "wife visit tomorrow morning",
    "head hurt take medicine",
    "watch tv son house",
]

with open("aphasic_to_normal_synthetic.json", encoding="utf-8") as f:
    data = json.load(f)
test = [item for item in data
        if int(hashlib.md5(item["original"].encode("utf-8")).hexdigest(), 16) % 10 == 0]
print(len(test), "test pairs")

device = torch.device("cuda")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL).to(device).eval()

for text in HAND_WRITTEN:
    inputs = tokenizer(PREFIX + text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=128, num_beams=4)
    print(f"{text!r} -> {tokenizer.decode(outputs[0], skip_special_tokens=True)!r}")

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
print(f"zero-shot {MODEL} BLEU on test set: {result['score']:.2f}")
