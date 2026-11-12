"""Build the extra fluent source corpora (run once; outputs go to data_src/).

- data_src/c4_aac.txt: sample of everyday conversational sentences from
  figmtu/aac_c4_deberta_classified_0.90 (Gaines & Vertanen, EMNLP 2025).
- data_src/comm2.txt is Vertanen's COMM2 set (CC BY 4.0), checked in as-is.
"""
import os
import random
import re
import urllib.request

import pyarrow.parquet as pq

URL = ("https://huggingface.co/datasets/figmtu/aac_c4_deberta_classified_0.90"
       "/resolve/main/data/dev-00000-of-00032.parquet")
PARQUET = "data_src/c4_dev0.parquet"
OUT = "data_src/c4_aac.txt"
N = 70000

OK = re.compile(r"^[A-Z][A-Za-z0-9 ,'’\-]*[.?!]$")
# Keep only sentences about the people in the conversation; this drops most
# of the product-description and encyclopedia text that slips through.
PERSONAL = re.compile(r"\b(I|I'm|I'll|I've|my|me|you|your|we|he|she|they|our)\b")
JUNK = re.compile(r"\b(click|website|blog|post|thread|forum|email|password|login|download|"
                  r"http|www|com|subscribe|coupon|sir|madam|pls|plz)\b", re.I)


def main():
    os.makedirs("data_src", exist_ok=True)
    if not os.path.exists(PARQUET):
        urllib.request.urlretrieve(URL, PARQUET)
    random.seed(42)
    f = pq.ParquetFile(PARQUET)
    groups = random.sample(range(f.metadata.num_row_groups), 700)
    seen, out = set(), []
    for g in groups:
        for s in f.read_row_group(g).column("text").to_pylist():
            s = s.strip()
            n = len(s.split())
            if not 3 <= n <= 16 or not OK.match(s) or JUNK.search(s) or not PERSONAL.search(s):
                continue
            if sum(c.isupper() for c in s) > 3 or s in seen:
                continue
            seen.add(s)
            out.append(s)
    random.shuffle(out)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out[:N]) + "\n")
    print(f"kept {min(len(out), N)} of {len(out)} candidates -> {OUT}")


if __name__ == "__main__":
    main()
