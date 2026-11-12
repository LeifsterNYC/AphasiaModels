"""Inference API for the demo site: POST /hfInference {input} -> {generated_text}.

Serves the CTranslate2 int8 export of the fine-tuned model on CPU. Deployed
copy lives in /srv/hfapi (hfapi.service); normalize.py is copied next to it.
"""
import os
import re

import ctranslate2
from fastapi import FastAPI, Request
from transformers import AutoTokenizer

from normalize import PREFIX, clean_input

MODEL_DIR = os.environ.get("APHASIA_MODEL", "/home/caduc3us/models/aphasia-v3-ct2")
MAX_CHARS = 200
BEAMS = 4

NEG = re.compile(r"\b(not|no|never|cannot|nothing|nobody|none|without|neither|nor)\b|n't")

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
translator = ctranslate2.Translator(MODEL_DIR, device="cpu", compute_type="int8",
                                    inter_threads=2, intra_threads=4)
app = FastAPI()


def restore(cleaned):
    tokens = tok.convert_ids_to_tokens(tok.encode(PREFIX + cleaned))
    result = translator.translate_batch([tokens], beam_size=BEAMS, num_hypotheses=BEAMS,
                                        max_decoding_length=96, return_scores=True)[0]
    texts = [tok.decode(tok.convert_tokens_to_ids(h), skip_special_tokens=True)
             for h in result.hypotheses]
    # Meaning guard: a restoration must not add or remove a negation. Take
    # the best-scoring beam that agrees with the input on that; if none
    # does, fall back to the top beam.
    negated = bool(NEG.search(cleaned))
    for text in texts:
        if bool(NEG.search(text.lower())) == negated:
            return text
    return texts[0]


@app.post("/hfInference")
async def infer(req: Request):
    data = await req.json()
    cleaned = clean_input(str(data.get("input", ""))[:MAX_CHARS])
    if not cleaned.replace("?", "").strip():
        return {"error": "Empty input"}
    return {"generated_text": restore(cleaned)}
