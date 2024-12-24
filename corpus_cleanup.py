import re

def clean_sentences(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
        for line in infile:
            if re.search(r"\[.*?\]", line):  # [LAUGHS]
                continue
            if re.search(r"\(.*?\)", line):  # (LAUGHS)
                continue
            if re.match(r"^\s*[-/]", line):  # Starts with -
                continue
            if re.match(r"^\s*\w+:", line):  # Starts with Word:
                continue
            if re.match(r"^\s*(Presented by|Produced by|In association with)", line, re.IGNORECASE):
                continue
            if re.search(r"[\"']", line):  # Quotes
                continue
            
            outfile.write(line)

input_file = "corpus.txt" 
output_file = "cleaned_corpus.txt"
clean_sentences(input_file, output_file)
