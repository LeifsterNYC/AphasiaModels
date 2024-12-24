import json
import random
import string
import spacy
from spacy.symbols import ORTH

# run python -m spacy download en_core_web_sm
    
# def subject_drop(doc, prob):
#     if random.random() < prob:
#         return " ".join([tok.text for tok in doc if tok.dep_ != "nsubj"])
#     return doc.text
    
# def remove_stopwords(doc, prob):
#     if random.random() < prob:
#         return " ".join([tok.text for tok in doc if not tok.is_stop])
#     return doc.text

# def verb_lemmatization(doc):
#     return " ".join([tok.lemma_ if tok.pos_ == "VERB" else tok.text for tok in doc])

# def remove_punctuation(text):
#     return text.translate(str.maketrans('', '', string.punctuation)).strip()                  

def aphasic_conversion(doc, stopword_prob=0.8, subject_drop_prob=0.5):
    tokens = []
    for tok in doc:
        if random.random() < stopword_prob and tok.is_stop:             # stop word removal
            continue
        if random.random() < subject_drop_prob and tok.dep_ == "nsubj": # subject drop
            continue
        if tok.pos_ == "VERB":                                          # verb lemmatization
            tokens.append(tok.lemma_)
        else:
            tokens.append(tok.text)
        
    text = " ".join(tokens)

    text = text.translate(str.maketrans('', '', string.punctuation)).strip()   # punctuation removal

    return text
    
def process_file(input_path, output_path, nlp, stopword_prob=0.8, subject_drop_prob=0.5):
    sentence_pairs = []
    
    lines = []
    with open(input_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 10000:
                break
            lines.append(line.strip())
            
    docs = list(nlp.pipe(lines)) # batch process
    sentence_pairs = []
    for original, doc in zip(lines, docs):              
        text = aphasic_conversion(doc, subject_drop_prob, stopword_prob)
        sentence_pairs.append({"aphasic": text, "original": original})
        
    with open(output_path, "w", encoding="utf-8") as outfile:
        json.dump(sentence_pairs, outfile, ensure_ascii=False)

def main():
    nlp = spacy.load("en_core_web_sm", disable=["ner"]) # disable unneeded features
    gonna = [{ORTH: "gonna"}]
    nlp.tokenizer.add_special_case("gonna", gonna)      # how to add special cases
    process_file("cleaned_corpus.txt", "aphasic_to_normal_synthetic.json", nlp)

if __name__ == "__main__":
    main()
