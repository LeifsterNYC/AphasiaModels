import json
import random

import spacy
from spacy.symbols import ORTH
from nltk.corpus import wordnet

from everyday import everyday_sentences
from normalize import clean_input

# run python -m spacy download en_core_web_sm
# run python -m nltk.downloader wordnet omw-1.4
# run python build_sources.py   (once, for data_src/c4_aac.txt)

# (path, max lines) of fluent source sentences. Subtitles alone are mostly
# second-person movie dialogue; the AAC-like corpora and the everyday
# templates supply the first-person needs/feelings/family sentences the model
# is actually asked to restore.
SOURCES = [
    ("cleaned_corpus.txt", 40000),
    ("data_src/c4_aac.txt", 70000),
    ("data_src/comm2.txt", None),
]
N_EVERYDAY = 9000
OUTPUT_PATH = "aphasic_to_normal_synthetic.json"

# Dropping these inverts the meaning of the sentence, so they are always kept.
NEGATION_WORDS = {"not", "no", "never", "n't", "neither", "nor", "none", "nothing", "cannot",
                  "nobody", "nowhere", "without"}

# Function words that carry information the model cannot guess back. If the
# input drops "you" or "his", any restoration is an invention: the previous
# model learned to make up subjects ("Mark will visit his wife") because the
# old generator deleted them. Only first-person and expletive forms may go.
PROTECTED = {"you", "he", "she", "we", "they", "him", "her", "them", "us", "your", "his",
             "their", "our", "yours", "hers", "theirs", "ours", "who", "what", "where",
             "when", "why", "how", "which", "whose", "yes"}

# Stopwords that still carry meaning; dropped far less often than "the"/"is".
SOFT_KEEP = {"but", "because", "if", "or", "before", "after", "until", "while", "with",
             "now", "here", "there", "again", "always", "only", "more", "less", "enough",
             "please", "maybe", "every", "all", "many", "much", "few", "other", "another",
             "first", "last", "next", "back", "up", "down", "off", "out", "together",
             "already", "still", "very", "too", "so", "can", "must", "should", "then"}

# Synonym substitution (semantic paraphasia), same-POS so it stays plausible.
WORDNET_POS = {"NOUN": wordnet.NOUN, "ADJ": wordnet.ADJ}

# Word-finding pauses, inserted before content words.
FILLERS = ["uh", "um", "er", "ah"]

# Paragrammatism: function words get *replaced* with a wrong one, not just
# deleted. Swaps stay within the same word class so the referent survives.
FUNCTION_WORD_SWAPS = {
    "a": ["the"], "an": ["the"], "the": ["a"],
    "in": ["on", "at"], "on": ["in", "at"], "at": ["in", "on"],
    "to": ["for"], "for": ["to"], "of": ["from"], "from": ["of"],
    "he": ["him"], "him": ["he"], "she": ["her"],
    "i": ["me"], "they": ["them"], "them": ["they"],
    "we": ["us"], "us": ["we"],
    "my": ["me"],
    "is": ["are", "be"], "are": ["is", "be"],
    "was": ["were", "is"], "were": ["was", "are"],
    "am": ["is"], "has": ["have"], "have": ["has"], "does": ["do"],
}

CLITICS = {"'s", "'m", "'re", "'ve", "'ll", "'d", "’s", "’m", "’re", "’ve", "’ll", "’d"}


def simple_synonym(word, pos):
    synsets = wordnet.synsets(word.lower(), pos=WORDNET_POS[pos])
    candidates = sorted({
        lemma.name().lower()
        for syn in synsets
        for lemma in syn.lemmas()
        if lemma.name().isalpha() and lemma.name().lower() != word.lower()
    })
    if candidates:
        return random.choice(candidates)
    return word


def is_negation(tok):
    return tok.dep_ == "neg" or tok.lower_ in NEGATION_WORDS


def aphasic_conversion(doc, stopword_prob, subject_drop_prob, lemma_prob, synonym_prob,
                       filler_prob, swap_prob, repeat_prob, restart_prob, neg_move_prob,
                       neg_no_prob, order_prob):
    # Each entry is [word, kind]; kind is "neg", "filler" or "word".
    out = []
    for tok in doc:
        if tok.is_punct or tok.is_space:
            continue
        low = tok.lower_
        if is_negation(tok):
            word = "not" if low in ("n't", "n’t") else low
            if word == "not" and random.random() < neg_no_prob:
                word = "no"
            out.append([word, "neg"])
            continue

        if low in CLITICS:
            # Possessive 's stays glued to its noun; auxiliary clitics drop
            # like any other function word.
            if tok.tag_ == "POS" and out and out[-1][1] == "word":
                out[-1][0] += "s"
            elif out and out[-1][1] == "word" and random.random() >= stopword_prob:
                out[-1][0] += low.strip("'’")
            continue

        protected = low in PROTECTED or tok.like_num or tok.pos_ in ("PROPN", "NUM")
        if not protected:
            if low == "i" and tok.dep_ in ("nsubj", "nsubjpass"):
                if random.random() < subject_drop_prob:
                    continue
            elif tok.is_stop:
                p = stopword_prob * (0.2 if low in SOFT_KEEP else 1.0)
                if random.random() < p:
                    continue

        if low in FUNCTION_WORD_SWAPS and random.random() < swap_prob:
            word = random.choice(FUNCTION_WORD_SWAPS[low])
        elif tok.pos_ in ("VERB", "AUX") and not tok.is_stop and random.random() < lemma_prob:
            word = tok.lemma_
        elif (tok.pos_ in WORDNET_POS and tok.is_alpha and not tok.is_stop
              and random.random() < synonym_prob):
            word = simple_synonym(tok.text, tok.pos_)
        else:
            word = tok.text

        # Word-finding pause before a content word.
        if tok.pos_ in ("NOUN", "VERB", "ADJ") and random.random() < filler_prob:
            out.append([random.choice(FILLERS), "filler"])
        out.append([word, "word"])
        # Perseverative repetition (never of a negation: "not not" reads as
        # a double negative rather than as noise).
        if random.random() < repeat_prob:
            out.append([word, "word"])

    # Misplaced negation: "he not go work" -> "he go not work". The target
    # keeps the natural placement, so the model learns to move it back.
    for i in range(len(out) - 1):
        if out[i][1] == "neg" and out[i + 1][1] == "word" and random.random() < neg_move_prob:
            out[i], out[i + 1] = out[i + 1], out[i]
            break

    # Light word-order disturbance between neighbouring words.
    for i in range(len(out) - 1):
        if out[i][1] == "word" and out[i + 1][1] == "word" and random.random() < order_prob:
            out[i], out[i + 1] = out[i + 1], out[i]

    # False start: the opening word gets attempted twice.
    if out and out[0][1] == "word" and random.random() < restart_prob:
        out = out[:1] + out

    text = " ".join(w for w, _ in out)
    # A question keeps its mark, otherwise "want eat food now" and
    # "want eat food now ?" would need the same input to mean two things.
    if doc.text.rstrip().endswith("?"):
        text += " ?"
    return clean_input(text)


def is_good_source(doc):
    tokens = [t for t in doc if not t.is_punct and not t.is_space]
    if not 3 <= len(tokens) <= 30:
        return False
    if not any(t.pos_ in ("VERB", "AUX") for t in tokens):
        return False
    # Mostly proper nouns means a title, name list, or credits line.
    if sum(1 for t in tokens if t.pos_ == "PROPN") / len(tokens) >= 0.5:
        return False
    return True


def load_sources():
    lines = []
    for path, limit in SOURCES:
        with open(path, encoding="utf-8") as f:
            chunk = [line.strip() for line in f if line.strip()]
        if limit and len(chunk) > limit:
            chunk = random.sample(chunk, limit)
        print(f"{path}: {len(chunk)} lines")
        lines += chunk
    everyday = everyday_sentences(N_EVERYDAY)
    print(f"everyday templates: {len(everyday)} lines")
    lines += everyday
    return list(dict.fromkeys(lines))


def process(lines, output_path, nlp):
    sentence_pairs = []
    n_neg = 0
    for original, doc in zip(lines, nlp.pipe(lines, batch_size=512)):
        if not is_good_source(doc):
            continue
        n_original = sum(1 for t in doc if not t.is_punct and not t.is_space)
        negated = any(is_negation(t) for t in doc)
        seen = set()
        # Negated sentences are the hardest to restore and the costliest to
        # get wrong, so they get an extra variant.
        for _ in range(3 if negated else 2):
            # Sample corruption strength per variant for a more diverse distribution.
            text = aphasic_conversion(
                doc,
                stopword_prob=random.uniform(0.4, 0.9),
                subject_drop_prob=random.uniform(0.3, 0.8),
                lemma_prob=random.uniform(0.5, 1.0),
                synonym_prob=random.uniform(0.0, 0.05),
                filler_prob=random.uniform(0.0, 0.12),
                swap_prob=random.uniform(0.05, 0.25),
                repeat_prob=random.uniform(0.0, 0.06),
                restart_prob=random.uniform(0.0, 0.25),
                neg_move_prob=random.uniform(0.1, 0.5),
                neg_no_prob=random.uniform(0.0, 0.3),
                order_prob=random.uniform(0.0, 0.06),
            )
            # Fillers and repetitions carry no content, so they do not count
            # toward retention. An input that lost most of its content words
            # is unrecoverable and only teaches the model to hallucinate.
            words = [w for w in text.split() if w != "?"]
            n_aphasic = len(set(words) - set(FILLERS))
            if n_aphasic < 2 or n_aphasic < 0.35 * n_original:
                continue
            if text in seen:
                continue
            seen.add(text)
            n_neg += negated
            sentence_pairs.append({"aphasic": text, "original": original})

    with open(output_path, "w", encoding="utf-8") as outfile:
        json.dump(sentence_pairs, outfile, ensure_ascii=False)
    print(f"Wrote {len(sentence_pairs)} pairs ({n_neg} negated) to {output_path}")


def main():
    random.seed(42)
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    gonna = [{ORTH: "gonna"}]
    nlp.tokenizer.add_special_case("gonna", gonna)
    process(load_sources(), OUTPUT_PATH, nlp)


if __name__ == "__main__":
    main()
