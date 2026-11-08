"""A1: sensitivity to annotator disagreement.

Strata by the range (max - min) of the three expert ratings: 0 = unanimous, 1 = adjacent, >=2 = split.
Distributional scoring: (a) share of excerpts where the score lies within the experts' range,
(b) mean absolute distance to each individual expert (expected error against a random expert).
Humans get the same treatment leave-one-out, against the other two experts.
"""
import json, sys
from collections import Counter
import numpy as np
from scipy.stats import spearmanr
from load import MODELS, ALIAS, load_humans_disagg, load_humans_iter, load_llm
from metrics import kappa, mae, median_int

root, out = sys.argv[1], sys.argv[2]
I = load_humans_iter(root)                      # 3 x 120
med = median_int(I).astype(float)
_, D = load_humans_disagg(root)
X = {ALIAS[m]: load_llm(root, m) for m in MODELS}
names = list(X)
rng_ = I.max(0) - I.min(0)
strata = {"unanimous (range 0)": rng_ == 0, "adjacent (range 1)": rng_ == 1, "split (range >= 2)": rng_ >= 2}
res = {"range_counts": {str(k): int(v) for k, v in sorted(Counter(rng_.astype(int).tolist()).items())},
       "strata_n": {k: int(v.sum()) for k, v in strata.items()}, "by_stratum": {}, "distributional": {}, "rank_robustness": {}}

# leave-one-out slots for humans (and LLMs in the same slots)
si, sh, so = [], [], []
for i in range(D.shape[1]):
    r = np.where(np.isfinite(D[:, i]))[0]
    for a in r:
        si.append(i); sh.append(D[a, i]); so.append(D[[b for b in r if b != a], i])
si, sh, so = np.array(si), np.array(sh), np.array(so)    # so: 360 x 2
sref = so.mean(1)

for k, s in strata.items():
    row = {}
    ss = np.isin(si, np.where(s)[0])
    row["Human (leave-one-out)"] = {"mae": mae(sref[ss], sh[ss]), "kappa_w": kappa(sref[ss], sh[ss]) if s.sum() > 5 else None}
    for n in names:
        row[n] = {"mae": mae(med[s], X[n][s]), "kappa_w": kappa(med[s], X[n][s]) if s.sum() > 5 else None}
    res["by_stratum"][k] = row

# distributional scoring
lo, hi = I.min(0), I.max(0)
h_in = np.mean((sh >= so.min(1)) & (sh <= so.max(1)))
h_dist = np.mean(np.abs(sh[:, None] - so).mean(1))
res["distributional"]["Human (vs other two)"] = {"within_range": float(h_in), "mean_dist_to_experts": float(h_dist)}
for n in names:
    x = X[n]
    # LLM against all three experts
    w3 = np.mean((x >= lo) & (x <= hi))
    d3 = np.mean(np.abs(x[None, :] - I).mean(0))
    # LLM in the human's leave-one-out slots (against the same two experts)
    xs = x[si]
    w2 = np.mean((xs >= so.min(1)) & (xs <= so.max(1)))
    d2 = np.mean(np.abs(xs[:, None] - so).mean(1))
    res["distributional"][n] = {"within_range_3": float(w3), "mean_dist_3": float(d3), "within_range": float(w2), "mean_dist_to_experts": float(d2)}

# ranking robustness: kappa on all items vs excluding split items
full = {n: kappa(med, X[n]) for n in names}
keep = rng_ <= 1
sub = {n: kappa(med[keep], X[n][keep]) for n in names}
res["rank_robustness"] = {"n_kept": int(keep.sum()), "full": full, "excl_split": sub,
                          "spearman": float(spearmanr([full[n] for n in names], [sub[n] for n in names])[0]),
                          "top3_full": sorted(names, key=lambda n: -full[n])[:3], "top3_excl_split": sorted(names, key=lambda n: -sub[n])[:3]}
json.dump(res, open(out, "w"), indent=1)

print("range counts:", res["range_counts"], " strata:", res["strata_n"])
for k, row in res["by_stratum"].items():
    print(f"\n{k} (n={res['strata_n'][k]})")
    for n, r in sorted(row.items(), key=lambda kv: kv[1]["mae"]):
        kw = "" if r["kappa_w"] is None else f" kappa={r['kappa_w']:.3f}"
        print(f"  {n:24s} mae={r['mae']:.3f}{kw}")
print("\nDistributional (two-expert slots; and vs all three for LLMs)")
for n, r in sorted(res["distributional"].items(), key=lambda kv: -kv[1]["within_range"]):
    extra = f"  | vs 3: within={r['within_range_3']:.3f} dist={r['mean_dist_3']:.3f}" if "within_range_3" in r else ""
    print(f"  {n:24s} within={r['within_range']:.3f} dist={r['mean_dist_to_experts']:.3f}{extra}")
rr = res["rank_robustness"]
print(f"\nRanking: Spearman full vs excl. split = {rr['spearman']:.3f} (n kept {rr['n_kept']}); top3 full {rr['top3_full']}, excl {rr['top3_excl_split']}")
for n in sorted(names, key=lambda n: -sub[n]): print(f"  {n:18s} full {full[n]:.3f}  excl split {sub[n]:.3f}")
