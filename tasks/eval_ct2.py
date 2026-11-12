"""Score the served pipeline (CTranslate2 int8 + negation guard) on the
hand-written set and time it. Usage: python tasks/eval_ct2.py [model_dir]"""
import os
import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, "serve")
if len(sys.argv) > 1:
    os.environ["APHASIA_MODEL"] = sys.argv[1]
import app  # noqa: E402
from normalize import clean_input  # noqa: E402
from tasks.eval_suite import report  # noqa: E402

rows = [line.rstrip("\n").split("\t") for line in open("tasks/handset.tsv", encoding="utf-8")
        if line.strip() and not line.startswith("#")]
srcs = [clean_input(r[0]) for r in rows]
refs = [r[1] for r in rows]
app.restore(srcs[0])
times, hyps = [], []
for s in srcs:
    t0 = time.perf_counter()
    hyps.append(app.restore(s))
    times.append(time.perf_counter() - t0)
for s, r, h in zip(srcs, refs, hyps):
    print(f"{s!r:52} -> {h}{'' if h == r else '   [ref: ' + r + ']'}")
report("handset/ct2-int8", srcs, refs, hyps)
times.sort()
print(f"latency s: median={times[len(times) // 2]:.2f} p95={times[int(len(times) * .95)]:.2f} max={times[-1]:.2f}")
