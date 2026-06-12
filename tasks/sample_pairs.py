import json
import random

d = json.load(open("aphasic_to_normal_synthetic.json", encoding="utf-8"))
print(len(d), "pairs total")
random.seed(7)
for p in random.sample(d, 20):
    print(repr(p["aphasic"]), "=>", repr(p["original"]))

print("--- negation ---")
negs = [p for p in d if " not " in (" " + p["original"].lower()) or "n't" in p["original"].lower()]
print(len(negs), "pairs with negation in original")
kept = sum(1 for p in negs if "not" in p["aphasic"].lower().split() or "never" in p["aphasic"].lower().split() or "no" in p["aphasic"].lower().split())
print(kept, "of those kept a negation word on the aphasic side")
for p in negs[:8]:
    print(repr(p["aphasic"]), "=>", repr(p["original"]))
