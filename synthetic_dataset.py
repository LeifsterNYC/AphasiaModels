import json
import random
import string

import spacy
from spacy.symbols import ORTH
from nltk.corpus import wordnet

# run python -m spacy download en_core_web_sm
# run python -m nltk.downloader wordnet omw-1.4

INPUT_PATH = "cleaned_corpus.txt"
OUTPUT_PATH = "aphasic_to_normal_synthetic.json"

# Dropping these inverts the meaning of the sentence, so they are always kept.
NEGATION_WORDS = {"not", "no", "never", "n't", "neither", "nor", "none", "nothing", "cannot"}

# Synonym substitution (semantic paraphasia) is only applied to these POS,
# using same-POS synsets so the substitute stays plausible.
WORDNET_POS = {"NOUN": wordnet.NOUN, "ADJ": wordnet.ADJ}

# Word-finding pauses, inserted before content words.
FILLERS = ["uh", "um", "er", "ah"]

# Paragrammatism: function words get *replaced* with a wrong one, not just
# deleted. Swaps stay within the same word class so output stays plausible.
FUNCTION_WORD_SWAPS = {
    "a": ["the"], "an": ["the"], "the": ["a"],
    "in": ["on", "at"], "on": ["in", "at"], "at": ["in", "on"],
    "to": ["for"], "for": ["to"], "of": ["from"], "from": ["of"],
    "he": ["him"], "him": ["he"], "she": ["her"], "her": ["she"],
    "i": ["me"], "me": ["i"], "they": ["them"], "them": ["they"],
    "we": ["us"], "us": ["we"],
    "his": ["him"], "my": ["me"], "their": ["them"],
    "is": ["are", "be"], "are": ["is", "be"],
    "was": ["were", "is"], "were": ["was", "are"],
    "am": ["is"], "has": ["have"], "have": ["has"], "does": ["do"],
}


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


def aphasic_conversion(doc, stopword_prob, subject_drop_prob, lemma_prob, synonym_prob,
                       filler_prob, swap_prob, repeat_prob, restart_prob):
    tokens = []
    for tok in doc:
        is_negation = tok.dep_ == "neg" or tok.lower_ in NEGATION_WORDS
        if is_negation:
            word = "not" if tok.lower_ == "n't" else tok.text
        else:
            if tok.is_stop and random.random() < stopword_prob:
                continue
            if tok.dep_ in ("nsubj", "nsubjpass") and random.random() < subject_drop_prob:
                continue
            if tok.lower_ in FUNCTION_WORD_SWAPS and random.random() < swap_prob:
                word = random.choice(FUNCTION_WORD_SWAPS[tok.lower_])
            elif tok.pos_ == "VERB" and random.random() < lemma_prob:
                word = tok.lemma_
            elif (tok.pos_ in WORDNET_POS and tok.is_alpha and not tok.is_stop
                  and random.random() < synonym_prob):
                word = simple_synonym(tok.text, tok.pos_)
            else:
                word = tok.text

        # Word-finding pause before a content word.
        if tok.pos_ in ("NOUN", "VERB", "ADJ") and random.random() < filler_prob:
            tokens.append(random.choice(FILLERS))
        tokens.append(word)
        # Perseverative repetition. Never repeat a negation: "not not" reads
        # as a double negative rather than as noise.
        if not is_negation and random.random() < repeat_prob:
            tokens.append(word)

    # False start: the opening word gets attempted twice.
    if tokens and random.random() < restart_prob:
        tokens = tokens[:1] + tokens

    text = " ".join(tokens)
    return text.translate(str.maketrans("", "", string.punctuation)).strip()


def is_good_source(doc):
    tokens = [t for t in doc if not t.is_punct and not t.is_space]
    if not 4 <= len(tokens) <= 30:
        return False
    if not any(t.pos_ in ("VERB", "AUX") for t in tokens):
        return False
    # Mostly proper nouns means a title, name list, or credits line.
    if sum(1 for t in tokens if t.pos_ == "PROPN") / len(tokens) >= 0.5:
        return False
    return True


def process_file(input_path, output_path, nlp, variants=2):
    with open(input_path, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    sentence_pairs = []
    for original, doc in zip(lines, nlp.pipe(lines, batch_size=256)):
        if not is_good_source(doc):
            continue
        n_original = sum(1 for t in doc if not t.is_punct and not t.is_space)
        seen = set()
        for _ in range(variants):
            # Sample corruption strength per variant for a more diverse distribution.
            text = aphasic_conversion(
                doc,
                stopword_prob=random.uniform(0.5, 0.9),
                subject_drop_prob=random.uniform(0.3, 0.7),
                lemma_prob=random.uniform(0.6, 1.0),
                synonym_prob=random.uniform(0.05, 0.2),
                filler_prob=random.uniform(0.0, 0.12),
                swap_prob=random.uniform(0.05, 0.25),
                repeat_prob=random.uniform(0.0, 0.06),
                restart_prob=random.uniform(0.0, 0.3),
            )
            # Fillers and repetitions carry no content, so they do not count
            # toward retention. An input that lost most of its content words
            # is unrecoverable and only teaches the model to hallucinate.
            words = text.split()
            n_aphasic = len(set(w.lower() for w in words) - set(FILLERS))
            if n_aphasic < 2 or n_aphasic < 0.3 * n_original:
                continue
            if text in seen:
                continue
            seen.add(text)
            sentence_pairs.append({"aphasic": text, "original": original})

    with open(output_path, "w", encoding="utf-8") as outfile:
        json.dump(sentence_pairs, outfile, ensure_ascii=False)
    print(f"Wrote {len(sentence_pairs)} pairs to {output_path}")


def main():
    random.seed(42)
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    gonna = [{ORTH: "gonna"}]
    nlp.tokenizer.add_special_case("gonna", gonna)
    process_file(INPUT_PATH, OUTPUT_PATH, nlp)


if __name__ == "__main__":
    main()
