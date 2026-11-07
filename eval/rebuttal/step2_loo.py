"""Step 2: leave-one-out human baseline, LLMs in the same slot, LLM vs individual annotators.

Slot = (item, held-out annotator): 120 items x 3 annotators = 360 slots.
Human score in a slot  = the held-out annotator's rating.
LLM score in a slot    = the LLM's rating of that item.
Reference in a slot    = median (= mean) of the other two annotators' ratings (may be x.5).
Metrics are pooled over all 360 slots, so humans and LLMs face exactly the same references.
"""
import json
import sys
from itertools import combinations

import numpy as np

from load import MODELS, ALIAS, load_humans_disagg, load_llm
from metrics import all_metrics

root, out = sys.argv[1], sys.argv[2]
names, D = load_humans_disagg(root)
A, N = D.shape
llm = {m: load_llm(root, m) for m in MODELS}

slot_item, slot_ann, slot_human, slot_ref = [], [], [], []
for i in range(N):
    raters = np.where(np.isfinite(D[:, i]))[0]
    for a in raters:
        others = [b for b in raters if b != a]
        slot_item.append(i)
        slot_ann.append(a)
        slot_human.append(D[a, i])
        slot_ref.append(np.mean(D[others, i]))
slot_item, slot_ann = np.array(slot_item), np.array(slot_ann)
slot_human, slot_ref = np.array(slot_human), np.array(slot_ref)
print("slots:", len(slot_ref), " half-point references:", int((slot_ref % 1 != 0).sum()))

res = {"loo": {}, "per_annotator_loo": {}, "llm_vs_individual": {}, "human_pairwise": {}}

res["loo"]["Human (held-out annotator)"] = all_metrics(slot_ref, slot_human)
for m in MODELS:
    res["loo"][ALIAS[m]] = all_metrics(slot_ref, llm[m][slot_item])

for a, nm in enumerate(names):
    s = slot_ann == a
    res["per_annotator_loo"][nm] = {"n": int(s.sum()), **all_metrics(slot_ref[s], slot_human[s])}

# LLM vs each individual annotator, on the items that annotator rated; plus pooled over 360 ratings
for m in MODELS:
    row = {}
    for a, nm in enumerate(names):
        s = np.isfinite(D[a])
        row[nm] = all_metrics(D[a, s], llm[m][s])
    pooled_ref = np.concatenate([D[a, np.isfinite(D[a])] for a in range(A)])
    pooled_x = np.concatenate([llm[m][np.isfinite(D[a])] for a in range(A)])
    row["pooled"] = all_metrics(pooled_ref, pooled_x)
    res["llm_vs_individual"][ALIAS[m]] = row

# Human vs human: every pair of annotators on the items both rated (pooled, symmetric)
px, py = [], []
for a, b in combinations(range(A), 2):
    s = np.isfinite(D[a]) & np.isfinite(D[b])
    if s.sum():
        res["human_pairwise"][f"{names[a]}-{names[b]}"] = {"n": int(s.sum()), **all_metrics(D[a, s], D[b, s])}
        px += list(D[a, s]) + list(D[b, s])
        py += list(D[b, s]) + list(D[a, s])
res["human_pairwise_pooled"] = all_metrics(np.array(px), np.array(py))
res["human_pairwise_pooled"]["n_pairs"] = len(px) // 2

json.dump(res, open(out, "w"), indent=1)

print(f"\nLeave-one-out (360 slots)\n{'rater':28s} kappa   rho    bias    mae")
for k, r in sorted(res["loo"].items(), key=lambda kv: -kv[1]["kappa_w"]):
    print(f"{k:28s} {r['kappa_w']:.3f}  {r['rho']:.3f}  {r['bias']:+.3f}  {r['mae']:.3f}")
print("\nPer annotator, held out vs median of other two")
for k, r in res["per_annotator_loo"].items():
    print(f"  {k}: n={r['n']:3d} kappa={r['kappa_w']:.3f} bias={r['bias']:+.3f} mae={r['mae']:.3f}")
print("\nHuman-human pairwise (pooled over", res["human_pairwise_pooled"]["n_pairs"], "pairs):",
      {k: round(v, 3) for k, v in res["human_pairwise_pooled"].items() if k != "n_pairs"})
print("\nLLM vs individual annotators, pooled over 360 ratings (kappa_w)")
for k, r in sorted(res["llm_vs_individual"].items(), key=lambda kv: -kv[1]["pooled"]["kappa_w"]):
    per = " ".join(f"{nm}:{r[nm]['kappa_w']:.2f}" for nm in names)
    print(f"  {k:18s} {r['pooled']['kappa_w']:.3f}   [{per}]")
