"""Evaluate a checkpoint on the hand-written set and a slice of the synthetic
held-out split. Usage: python tasks/eval_suite.py <model_dir> [--quiet] [--handset-only]

BLEU alone hid the failures that matter here (van Vaals et al. 2025 dropped
it for the same reason), so this also counts the three things the previous
model got wrong: flipped negation, invented names, and invented people ("you", "he", "they") the input never mentioned.
"""
import hashlib
import json
import re
import sys

import sacrebleu
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

sys.path.insert(0, ".")
from normalize import PREFIX, clean_input  # noqa: E402

NEG = re.compile(r"\b(not|no|never|cannot|nothing|nobody|none|without|neither|nor)\b|n't")
PERSON = re.compile(r"\b(you|your|yours|he|him|his|she|her|they|them|their|we|us|our)\b")


def words(text):
    return re.findall(r"[a-z0-9]+", text.lower().replace("'", ""))


def invented_names(src, hyp):
    src_words = set(words(src))
    tokens = re.findall(r"[A-Za-z']+", hyp)
    return [t for t in tokens[1:]
            if t[0].isupper() and t not in ("I", "I'm", "I'll", "I've", "I'd", "TV")
            and t.lower().replace("'s", "") not in src_words and t.lower() not in src_words]


def generate(model, tok, device, texts, beams=4, bs=64):
    out = []
    for i in range(0, len(texts), bs):
        enc = tok([PREFIX + t for t in texts[i:i + bs]], return_tensors="pt",
                  padding=True, truncation=True, max_length=96).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, max_length=96, num_beams=beams)
        out += tok.batch_decode(gen, skip_special_tokens=True)
    return out


def report(name, srcs, refs, hyps):
    n = len(srcs)
    neg_flip = sum(bool(NEG.search(s)) != bool(NEG.search(h.lower())) for s, h in zip(srcs, hyps))
    names = sum(bool(invented_names(s, h)) for s, h in zip(srcs, hyps))
    person = sum(bool(set(PERSON.findall(h.lower())) - set(PERSON.findall(s))) for s, h in zip(srcs, hyps))
    exact = sum(h.strip() == r.strip() for h, r in zip(hyps, refs))
    print(f"[{name}] n={n} chrF={sacrebleu.corpus_chrf(hyps, [refs]).score:.1f} "
          f"BLEU={sacrebleu.corpus_bleu(hyps, [refs]).score:.1f} exact={exact / n:.1%} "
          f"neg_flip={neg_flip / n:.1%} invented_name={names / n:.1%} invented_person={person / n:.1%}")


def main():
    path = sys.argv[1]
    quiet = "--quiet" in sys.argv
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSeq2SeqLM.from_pretrained(path).to(device).eval()

    rows = [line.rstrip("\n").split("\t") for line in open("tasks/handset.tsv", encoding="utf-8")
            if line.strip() and not line.startswith("#")]
    srcs = [clean_input(r[0]) for r in rows]
    refs = [r[1] for r in rows]
    hyps = generate(model, tok, device, srcs)
    if not quiet:
        for s, r, h in zip(srcs, refs, hyps):
            print(f"{s!r:52} -> {h}{'' if h == r else '   [ref: ' + r + ']'}")
    report("handset", srcs, refs, hyps)

    if "--handset-only" in sys.argv:
        return
    data = json.load(open("aphasic_to_normal_synthetic.json", encoding="utf-8"))
    test = [d for d in data
            if int(hashlib.md5(d["original"].encode("utf-8")).hexdigest(), 16) % 10 == 0][:3000]
    srcs = [d["aphasic"] for d in test]
    refs = [d["original"] for d in test]
    report("synthetic", srcs, refs, generate(model, tok, device, srcs))


if __name__ == "__main__":
    main()
