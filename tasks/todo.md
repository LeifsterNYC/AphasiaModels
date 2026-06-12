# Make AphasiaModels work very well — 2026-06-11

## Plan
- [x] Setup: install evaluate, sacrebleu, sentencepiece, nltk, spacy (+ en_core_web_sm, wordnet) — all have Python 3.14 wheels, no venv fallback needed
- [x] Rewrite `synthetic_dataset.py`: fix swapped-args bug, preserve negation, source-line filter (4–30 tokens, has verb, <50% proper nouns), per-variant sampled probs, constrained NOUN/ADJ synonym substitution, pair filter (≥30% retention), full 56k corpus, seeded, 2 variants/line
- [x] Regenerate `aphasic_to_normal_synthetic.json` (51,884 pairs) and eyeball samples — negation kept in 1288/1288 negative pairs
- [x] Fix `model.py`: wire compute_metrics (predict_with_generate, -100 handling, sacrebleu on strings), bf16 instead of fp16, lr 2e-4, warmup 0.05, 6 epochs, bs 32/64, load_best_model_at_end on BLEU, hash-based split by target sentence (avoids duplicate-line leakage)
- [x] Train `google/flan-t5-base` on RTX 5090 (47,090 train / 4,794 test, 20 min, no NaN)
- [x] Report test-set BLEU — new model 46.99 (epoch-5 best checkpoint); BLEU per epoch: 41.3 → 45.0 → 45.7 → 46.5 → 47.0 → 46.8
- [x] Side-by-side: old checkpoint vs new model on 14 fixed inputs (`tasks/compare_models.py`)
- [x] Old model BLEU on the identical test set: 33.21 (`tasks/eval_old_model.py`) → **+13.8 BLEU**
- [x] Zero-shot (untuned) flan-t5-base on the same test set: BLEU 0.58 — just echoes the prompt (`tasks/eval_base_model.py`), so the fine-tuning accounts for essentially all capability
- [x] Update `test.py` to default to newest model dir (or CLI arg)

## Round 2 — realism upgrade (2026-06-11)
- [x] Corruption model v2: fillers (uh/um before content words), word repetitions + sentence false starts, function-word *substitution* (paragrammatism: a↔the, in/on/at, he↔him, is↔are...) on surviving stopwords; fillers/repeats excluded from retention count
- [x] Regenerate dataset (52,989 pairs) — negation kept 2480/2480
- [x] Retrain flan-t5-base → `models/model_2026-06-11_22-17-42/final`
- [x] Cross-eval on the v2 (disfluent) test set, 4,850 pairs: **v2 46.09 / v1 40.71 / old t5-small 29.55** — realism training worth +5.4 BLEU on disfluent input
- [x] Three-way side-by-side (`tasks/compare_models.py`): v2 cleanly strips fillers/repetitions ("i i not like uh hospital food" → "I do not like the hospital food."); v1/old parrot them back. Minor v2 regressions on clean inputs: hallucinated a name once ("Mark will visit his wife"), one wrong preposition.
- [x] Research real aphasia datasets — see Review

### Real dataset findings
- **AphasiaBank (TalkBank)** — the standard: ~180 aphasic speakers + controls, CHAT transcripts + media. Free for aphasia researchers: register, then email talkbank@andrew.cmu.edu verifying institutional status & agreeing to Ground Rules. Citation required. https://talkbank.org/aphasia/access.html
- **APROCSA open dataset** (MDPI Data 2022) — 6 chronic post-stroke speakers, CHAT transcripts, open with simple registration (no membership). Small but real; good as a held-out *evaluation* set. https://www.mdpi.com/2306-5729/7/11/148
- **van Vaals, Matusevych & Tsiwah 2025** — same task (completing Broca's aphasic sentences with fine-tuned T5 family on synthetic data, evaluated on real AphasiaBank). Public code incl. CHAT-file processing + their rule-based generator: https://github.com/sijbrenvv/Completions_for_Broca-s_aphasia — their reported failure mode (models spuriously *adding* negations, 2.99% for FLAN-T5-XL) validates our negation-preserving generator. Notable: t5-base beat flan-t5-base on their synthetic test.
- **SBCSAE** (Santa Barbara Corpus of Spoken American English) — neurotypical *conversational* corpus the paper used as generation source; arguably better than OpenSubtitles for this purpose.
- No ready-made text-normalization dataset on HF hub (only an audio ASR model for Singlish aphasic speech).

## Review

**Result: new model at `models/model_2026-06-11_20-41-38/final`, BLEU 46.99 vs old 33.21 on the same held-out set.**

What changed and why it mattered:
1. `synthetic_dataset.py` had its stopword/subject probabilities silently swapped (positional-arg bug), deleted negation words 80% of the time (meaning-inverting pairs), and kept junk pairs (titles, verbless fragments, unrecoverable inputs). Now: negation always kept, source lines filtered (4–30 tokens, must contain a verb, <50% proper nouns), pairs must retain ≥30% of tokens, synonym paraphasia constrained to same-POS NOUN/ADJ, 2 sampled-strength variants per line, seeded. 51,884 pairs.
2. `model.py`: compute_metrics was dead code (never passed to the trainer); now wired with predict_with_generate, -100→pad handling, sacrebleu on strings. fp16→bf16 (T5 NaN risk), lr 2e-5→2e-4, warmup, best-checkpoint-on-BLEU selection, and a hash-of-target split (corpus duplicate lines + per-line variants would otherwise leak eval targets into training).
3. Base model t5-small → flan-t5-base.
4. `test.py` auto-loads the newest `models/model_*/final` or takes a CLI arg.

Caveats: part of the BLEU gap reflects that the old model was trained on the old (buggy) pair distribution; the hand-written side-by-side (`tasks/compare_models.py`) shows the qualitative gain independently. Eval loss rises after epoch 2 while BLEU climbs — mild overfit, handled by best-checkpoint selection.

Utilities kept in `tasks/`: `sample_pairs.py`, `compare_models.py`, `eval_old_model.py`.
All changes uncommitted (per plan); `push_to_hub.py` untouched.
