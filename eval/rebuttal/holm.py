"""Holm-adjusted p-values for each model vs the individual-expert baseline and vs the steering baseline."""
import json
import sys


def holm(p):
    keys = sorted(p, key=p.get)
    m, run, out = len(keys), 0.0, {}
    for j, k in enumerate(keys):
        run = max(run, min(1.0, (m - j) * p[k]))
        out[k] = run
    return out


d = sys.argv[1]
boot = json.load(open(f"{d}/boot.json"))["loo_diff_vs_human"]
sub = json.load(open(f"{d}/subset.json"))["vs_steering"]
res = {"vs_human_holm": holm({k: v["p"] for k, v in boot.items()}),
       "vs_steering_holm": holm({k: v["p_kappa"] for k, v in sub.items()})}
json.dump(res, open(f"{d}/holm.json", "w"), indent=1)
for name, r in res.items():
    print(name)
    for k, v in sorted(r.items(), key=lambda kv: kv[1]):
        print(f"  {k:18s} {v:.4f}")
