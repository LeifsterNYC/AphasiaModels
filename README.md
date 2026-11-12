# AphasiaModels

Restoring telegraphic, aphasic-style English to full sentences with a fine-tuned
sequence-to-sequence model.

People with non-fluent (Broca's-type) aphasia often produce "telegraphic" speech:
content words survive while function words, inflections, and sentence structure are
lost — *"want water"*, *"yesterday go store buy bread"*. This project trains a model
to map that kind of input back to fluent English:

```
input:  i i not like uh hospital food
model:  I do not like the hospital food.
```

There is no large public corpus of aphasic↔fluent sentence pairs, so the training
data here is **synthetic**: a rule-based generator corrupts fluent sentences with
linguistically motivated aphasic patterns, and the model learns the inverse mapping.

```
aphasic:  Exactly years ago today  bury time capsule
original: Exactly two years ago today, she and I buried a time capsule here.

aphasic:  parents parents want daughter  so er raise me like one
original: My parents wanted a daughter, so they raised me like one.
```

## Results

### v3: flan-t5-large, everyday data (November 2026)

The v2 model scored well on its own synthetic test set but failed on plain everyday
phrases: it invented people ("wife visit tomorrow morning" → "Mark will visit his
wife..."), turned statements into second-person questions, and mangled misplaced
negation ("he go not work today" → "But he is going to not work today."). All three
traced back to the training pairs, so v3 changes the data more than the model:

- **Sources**: subtitles are now mixed with everyday conversational sentences —
  a sample of the AAC-classified C4 set (Gaines & Vertanen, EMNLP 2025), Vertanen's
  COMM2 set, and 9,000 templated first-person needs/pain/family sentences
  (`everyday.py`). 186,892 pairs.
- **Nothing unrecoverable is deleted**: "you/he/she/his/their", wh-words, names and
  numbers are never dropped; only "I", "my" and expletives may go, so a subjectless
  input always restores to the first person.
- **Negation** is still never dropped, and is now sometimes misplaced
  ("he go not work") or reduced to "no" so the model learns to move it back.
- **Questions** keep a trailing `?` in the input.
- **Inputs are canonicalised** (`normalize.py`: lowercase, no punctuation except a
  final `?`) identically in training and serving.
- **Model**: `google/flan-t5-large`, 2 epochs, checkpoint chosen by chrF; exported
  to CTranslate2 int8 for CPU serving (`serve/app.py`, ~0.25 s per phrase on a
  Ryzen 5700X). The server also rejects any beam that adds or removes a negation.

Scores on `tasks/handset.tsv`, 80 hand-written phrases. 20 of the 80 references
also occur among the templated training targets, so the 60 that do not are the
fair comparison:

| Model (as served)            | chrF (all 80) | chrF (60 unseen) | exact (60) | invents a person (60) |
|------------------------------|---------------|------------------|------------|-----------------------|
| v2 flan-t5-base              | 71.5          | 69.9             | 20.0%      | 31.7%                 |
| **v3 flan-t5-large, int8**   | **86.3**      | **83.5**         | **43.3%**  | **0%**                |

On its own held-out synthetic split (3,000 pairs) v3 scores chrF 79.4 / BLEU 61.4;
that is not comparable with the v2 table below, which uses the v2 dataset.

Known failures: bare two- or three-word inputs are ambiguous without context
("book table"), "no can ..." is read as "No, I can ..." rather than "I cannot ...",
and tense is a guess when the input has no time word.

### v2 (June 2026)

BLEU on a held-out test set of 4,850 synthetic disfluent pairs (split by hash of the
target sentence, so no source sentence appears in both train and test):

| Model                                              | BLEU      |
|----------------------------------------------------|-----------|
| flan-t5-base, zero-shot (no fine-tuning)           | 0.58      |
| t5-small fine-tuned on v1 data                     | 29.55     |
| flan-t5-base fine-tuned on v1 (clean) data         | 40.71     |
| **flan-t5-base fine-tuned on v2 (disfluent) data** | **46.09** |

The zero-shot row shows the base model mostly echoes the prompt — fine-tuning
accounts for essentially all task capability. The v1→v2 gap (+5.4 BLEU) comes from
training on data that includes real-speech disfluencies (fillers, repetitions, false
starts): the v2 model cleanly strips them, while earlier models parrot them back.

## The data generator

`synthetic_dataset.py` turns fluent sentences into aphasic-style ones using spaCy
parses, simulating documented patterns of non-fluent aphasia:

- **Agrammatism** — stopwords dropped with high probability, grammatical subjects
  dropped, verbs reduced to their lemma (*buried* → *bury*)
- **Paragrammatism** — function words *substituted* rather than deleted, within the
  same word class (*a*↔*the*, *in/on/at*, *he*↔*him*, *is*↔*are*)
- **Semantic paraphasia** — nouns/adjectives replaced by a WordNet synonym,
  constrained to the same part of speech so substitutes stay plausible
- **Disfluencies** — word-finding fillers (*uh*, *um*) before content words,
  perseverative word repetitions, and false starts (first word attempted twice)

Guardrails keep the pairs learnable and meaning-preserving:

- **Negation is always preserved.** Dropping *not* silently inverts the target
  meaning; every generated pair keeps negation words (verified across all 2,480
  negative pairs in the dataset).
- **Source filtering** — sentences must be 4–30 tokens, contain a verb, and be <50%
  proper nouns (drops subtitle credits, titles, and name lists).
- **Recoverability filter** — a pair is discarded if the corrupted side retains <30%
  of the original's content tokens; unrecoverable inputs only teach the model to
  hallucinate.
- Each source sentence yields two corruption variants with independently sampled
  corruption strengths, seeded for reproducibility.

The source corpus is 100k lines of the OpenSubtitles English corpus
([OPUS](https://opus.nlpl.eu/OpenSubtitles.php), Lison & Tiedemann 2016), filtered by
`corpus_cleanup.py` (removes stage directions, speaker labels, credits, and quoted
dialogue). The result is **52,989 aphasic↔fluent pairs**
(`aphasic_to_normal_synthetic.json`).

## Training

`model.py` fine-tunes [`google/flan-t5-base`](https://huggingface.co/google/flan-t5-base)
with Hugging Face `Seq2SeqTrainer`:

- Prefix `"normalize in English: "`, max length 128
- bf16, learning rate 2e-4 with warmup, batch size 32, 6 epochs, seeded
- BLEU (sacrebleu) computed each epoch with `predict_with_generate`; best checkpoint
  selected by BLEU
- **Leakage-safe split**: train/test assignment by hash of the *target* sentence,
  since the corpus contains duplicate lines and each line yields multiple variants —
  a random pair-level split would leak eval targets into training

Trains in ~20 minutes on a single RTX 5090.

## Repository layout

```
corpus.txt                        100k lines of OpenSubtitles English
corpus_cleanup.py                 corpus filtering -> cleaned_corpus.txt
synthetic_dataset.py              aphasic-pair generator (spaCy + WordNet)
aphasic_to_normal_synthetic.json  52,989 generated training pairs
model.py                          flan-t5-base fine-tuning + BLEU eval
tasks/compare_models.py           side-by-side generation across checkpoints
tasks/eval_old_model.py           BLEU of a prior checkpoint on the current test set
tasks/eval_base_model.py          zero-shot BLEU of untuned flan-t5-base
tasks/sample_pairs.py             inspect random generated pairs
```

## Running it

```bash
pip install torch transformers datasets evaluate sacrebleu sentencepiece spacy nltk
python -m spacy download en_core_web_sm
python -m nltk.downloader wordnet omw-1.4

python corpus_cleanup.py       # corpus.txt -> cleaned_corpus.txt
python synthetic_dataset.py    # -> aphasic_to_normal_synthetic.json
python model.py                # fine-tune, eval each epoch, save best to models/
```

## Limitations and next steps

- Synthetic corruption is an approximation; the real test is transcripts of actual
  aphasic speech. Next step: evaluate on
  [AphasiaBank](https://talkbank.org/aphasia/) and the open
  [APROCSA](https://www.mdpi.com/2306-5729/7/11/148) transcripts.
- OpenSubtitles is scripted dialogue; a conversational corpus like the
  [Santa Barbara Corpus](https://www.linguistics.ucsb.edu/research/santa-barbara-corpus)
  would be a better generation source for spoken-style targets.
- Part of the gap over the old t5-small reflects that it was trained on earlier,
  buggier data; the qualitative side-by-side (`tasks/compare_models.py`) shows the
  improvement independently of that.
- Related work: [van Vaals, Matusevych & Tsiwah (2025)](https://github.com/sijbrenvv/Completions_for_Broca-s_aphasia)
  fine-tune T5-family models on synthetic data for the same task and report models
  spuriously *adding* negations — the failure mode this project's
  negation-preserving generator is designed against.

## License

Code and generated data are released under the [MIT License](LICENSE). The source
corpus derives from OpenSubtitles via [OPUS](https://opus.nlpl.eu/OpenSubtitles.php):

> P. Lison and J. Tiedemann, 2016, *OpenSubtitles2016: Extracting Large Parallel
> Corpora from Movie and TV Subtitles.* In Proceedings of LREC 2016.
