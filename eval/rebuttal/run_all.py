"""Reproduce all rebuttal analyses. Run from the repository root:

    python eval/rebuttal/run_all.py

Results (JSON) and generated LaTeX table rows go to eval/rebuttal/results/.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)


def run(script, *args, stdout=None):
    print(f"\n### {script}")
    subprocess.run([sys.executable, str(HERE / script), *map(str, args)], check=True, stdout=stdout)


run("step1_check.py", ROOT)
run("step2_loo.py", ROOT, OUT / "loo.json")
run("step3_bootstrap.py", ROOT, OUT / "boot.json", 10000)
run("step4_baselines.py", ROOT, OUT / "baselines.json")
run("step4b_subset.py", ROOT, OUT / "subset.json")
run("holm.py", OUT)
run("run_alt_test.py", ROOT, OUT / "alt_test.json")
run("step5_family.py", ROOT, OUT / "family.json")
run("step6_f1_pareto.py", ROOT, OUT / "f1_pareto.json")
run("step7_disagreement.py", ROOT, OUT / "disagreement.json")
run("step8_overestimate.py", ROOT, OUT / "overestimate.json")
run("step9_article_plaus.py", ROOT, OUT / "article_plausibility.json")
with open(OUT / "table_rows.tex", "w") as f:
    run("make_tables.py", OUT, stdout=f)
print(f"\nDone. Results in {OUT}")
