"""A3: do OpenAI models (same family as the gpt-4.1-mini generator) agree more, or rate more leniently?"""
import json, sys
import numpy as np
from load import MODELS, ALIAS, load_humans_iter, load_llm
from metrics import kappa, mae, bias, median_int

root, out = sys.argv[1], sys.argv[2]
B = 10000
rng = np.random.default_rng(11)
med = median_int(load_humans_iter(root)).astype(float)
X = {m: load_llm(root, m) for m in MODELS}
OAI = ["gpt5", "4o", "o3", "o3_mini"]
OTH = [m for m in MODELS if m not in OAI]
N = len(med)

def group_stats(s):
    r = {}
    for k, f in {"kappa_w": kappa, "mae": mae, "bias": bias}.items():
        a = np.mean([f(med[s], X[m][s]) for m in OAI]); b = np.mean([f(med[s], X[m][s]) for m in OTH])
        r[k] = (a, b, a - b)
    return r

pt = group_stats(np.arange(N))
idx = rng.integers(0, N, size=(B, N))
boots = [group_stats(s) for s in idx]
res = {"openai": [ALIAS[m] for m in OAI], "others": [ALIAS[m] for m in OTH], "B": B}
for k in ["kappa_w", "mae", "bias"]:
    d = np.array([b[k][2] for b in boots])
    res[k] = {"openai_mean": pt[k][0], "others_mean": pt[k][1], "diff": pt[k][2],
              "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
              "p": float(min(1, 2 * min(np.mean(d <= 0), np.mean(d >= 0))))}
# ranks
ranked = sorted(MODELS, key=lambda m: -kappa(med, X[m]))
res["kappa_ranks_openai"] = {ALIAS[m]: ranked.index(m) + 1 for m in OAI}
res["kappa_ranks_others"] = {ALIAS[m]: ranked.index(m) + 1 for m in OTH}
# permutation test over model labels: is the OpenAI-vs-others kappa gap unusual among all 4-vs-6 splits?
from itertools import combinations
ks = {m: kappa(med, X[m]) for m in MODELS}
gaps = [np.mean([ks[m] for m in g]) - np.mean([ks[m] for m in MODELS if m not in g]) for g in combinations(MODELS, 4)]
res["kappa_gap_label_permutation_p"] = float(np.mean(np.abs(gaps) >= abs(pt["kappa_w"][2]) - 1e-12))
bs = {m: bias(med, X[m]) for m in MODELS}
bgaps = [np.mean([bs[m] for m in g]) - np.mean([bs[m] for m in MODELS if m not in g]) for g in combinations(MODELS, 4)]
res["bias_gap_label_permutation_p"] = float(np.mean(np.abs(bgaps) >= abs(pt["bias"][2]) - 1e-12))
json.dump(res, open(out, "w"), indent=1)
for k in ["kappa_w", "mae", "bias"]:
    r = res[k]; print(f"{k:8s} OpenAI {r['openai_mean']:+.3f}  others {r['others_mean']:+.3f}  diff {r['diff']:+.3f} [{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}] p={r['p']:.3f}")
print("kappa ranks OpenAI:", res["kappa_ranks_openai"]); print("kappa ranks others:", res["kappa_ranks_others"])
print("label-permutation p (kappa gap):", round(res["kappa_gap_label_permutation_p"], 3), " (bias gap):", round(res["bias_gap_label_permutation_p"], 3))
