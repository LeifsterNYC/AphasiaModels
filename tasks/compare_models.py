"""Side-by-side comparison of the old t5-small checkpoint vs the new model."""
import glob
import hashlib
import json
import os
import sys

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

OLD_MODEL = "./models/model_2025-01-04_13-49-09/checkpoint-12145"
PREFIX = "normalize in English: "

HAND_WRITTEN = [
    "want water",
    "yesterday go store buy bread",
    "not like hospital food",
    "wife visit tomorrow morning",
    "head hurt take medicine",
    "watch tv son house",
    # disfluent variants: fillers, repetitions, false starts, wrong function words
    "want want uh water",
    "yesterday go go store uh buy bread",
    "i i not like uh hospital food",
    "wife visit visit on tomorrow morning",
    "me head hurt um take take medicine",
]


def newest_final(models_root="./models"):
    candidates = glob.glob(os.path.join(models_root, "model_*", "final"))
    return max(candidates, key=os.path.getmtime) if candidates else None


def held_out_examples(n=8):
    with open("aphasic_to_normal_synthetic.json", encoding="utf-8") as f:
        data = json.load(f)
    examples = []
    seen = set()
    for item in data:
        digest = hashlib.md5(item["original"].encode("utf-8")).hexdigest()
        if int(digest, 16) % 10 == 0 and item["original"] not in seen:
            seen.add(item["original"])
            examples.append(item)
        if len(examples) >= n:
            break
    return examples


def generate(model, tokenizer, device, text):
    inputs = tokenizer(PREFIX + text, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=128, num_beams=4)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    if len(sys.argv) > 1:
        paths = sys.argv[1:]
    else:
        finals = sorted(glob.glob(os.path.join("./models", "model_*", "final")),
                        key=os.path.getmtime)
        paths = [OLD_MODEL] + finals
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = {}
    for i, path in enumerate(paths):
        name = f"m{i}"
        print(f"{name}: {path}")
        tok = AutoTokenizer.from_pretrained(path)
        mod = AutoModelForSeq2SeqLM.from_pretrained(path).to(device).eval()
        models[name] = (mod, tok)
    print()

    cases = [{"aphasic": t, "original": "(hand-written)"} for t in HAND_WRITTEN]
    cases += held_out_examples()

    for case in cases:
        print(f"input:    {case['aphasic']}")
        print(f"expected: {case['original']}")
        for name, (mod, tok) in models.items():
            print(f"{name}:       {generate(mod, tok, device, case['aphasic'])}")
        print()


if __name__ == "__main__":
    main()
