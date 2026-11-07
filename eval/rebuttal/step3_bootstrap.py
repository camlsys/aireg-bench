"""Step 3: item-level bootstrap CIs and paired bootstrap comparisons.

Resampling unit = excerpt (120). For the leave-one-out comparison, each resampled excerpt
brings its three slots with it, so the human and the LLMs are always scored on the same draws.
"""
import json
import sys
from itertools import combinations

import numpy as np

from load import MODELS, ALIAS, load_humans_disagg, load_humans_iter, load_llm
from metrics import METRICS, median_int

root, out = sys.argv[1], sys.argv[2]
B = int(sys.argv[3]) if len(sys.argv) > 3 else 10000
rng = np.random.default_rng(20261107)

I = load_humans_iter(root)
med = median_int(I).astype(float)
_, D = load_humans_disagg(root)
N = len(med)
X = {ALIAS[m]: load_llm(root, m) for m in MODELS}
names = list(X)

# Leave-one-out slots, indexed by item
slot_item, slot_h, slot_ref = [], [], []
for i in range(N):
    r = np.where(np.isfinite(D[:, i]))[0]
    for a in r:
        slot_item.append(i); slot_h.append(D[a, i]); slot_ref.append(np.mean(D[[b for b in r if b != a], i]))
slot_item, slot_h, slot_ref = map(np.array, (slot_item, slot_h, slot_ref))
slots_of = [np.where(slot_item == i)[0] for i in range(N)]

idx = rng.integers(0, N, size=(B, N))


def ci(a):
    a = np.asarray(a)
    a = a[np.isfinite(a)]
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


res = {"B": B, "table2": {}, "loo": {}, "loo_diff_vs_human": {}, "paired": {}, "rank1_prob": {}}

# --- Table 2 (vs median of three) ---
boot = {n: {k: np.empty(B) for k in METRICS} for n in names}
for b in range(B):
    s = idx[b]
    for n in names:
        for k, f in METRICS.items():
            boot[n][k][b] = f(med[s], X[n][s])
for n in names:
    res["table2"][n] = {k: {"point": METRICS[k](med, X[n]), "ci": ci(boot[n][k])} for k in METRICS}

# Probability of ranking first on kappa
best = np.array([max(names, key=lambda n: boot[n]["kappa_w"][b]) for b in range(B)])
res["rank1_prob"] = {n: float(np.mean(best == n)) for n in names}

# --- Paired comparisons on kappa and on per-item absolute error ---
for a, c in combinations(names, 2):
    dk = boot[a]["kappa_w"] - boot[c]["kappa_w"]
    dm = boot[a]["mae"] - boot[c]["mae"]
    pk = 2 * min(np.mean(dk <= 0), np.mean(dk >= 0))
    pm = 2 * min(np.mean(dm <= 0), np.mean(dm >= 0))
    res["paired"][f"{a} | {c}"] = {
        "d_kappa": METRICS["kappa_w"](med, X[a]) - METRICS["kappa_w"](med, X[c]), "d_kappa_ci": ci(dk), "p_kappa": float(min(pk, 1)),
        "d_mae": METRICS["mae"](med, X[a]) - METRICS["mae"](med, X[c]), "d_mae_ci": ci(dm), "p_mae": float(min(pm, 1)),
    }

# Holm correction across the 45 pairs, separately for kappa and MAE
for key in ["p_kappa", "p_mae"]:
    pairs = sorted(res["paired"], key=lambda k: res["paired"][k][key])
    m, running = len(pairs), 0.0
    for j, k in enumerate(pairs):
        running = max(running, min(1.0, (m - j) * res["paired"][k][key]))
        res["paired"][k][key + "_holm"] = running

# --- Leave-one-out with CIs, and LLM minus human difference ---
loo_boot = {n: np.empty(B) for n in names + ["Human"]}
loo_mae = {n: np.empty(B) for n in names + ["Human"]}
for b in range(B):
    ss = np.concatenate([slots_of[i] for i in idx[b]])
    ref = slot_ref[ss]
    loo_boot["Human"][b] = METRICS["kappa_w"](ref, slot_h[ss]); loo_mae["Human"][b] = METRICS["mae"](ref, slot_h[ss])
    for n in names:
        x = X[n][slot_item[ss]]
        loo_boot[n][b] = METRICS["kappa_w"](ref, x); loo_mae[n][b] = METRICS["mae"](ref, x)
pt = {"Human": METRICS["kappa_w"](slot_ref, slot_h)}
pt.update({n: METRICS["kappa_w"](slot_ref, X[n][slot_item]) for n in names})
for n in ["Human"] + names:
    res["loo"][n] = {"kappa_w": pt[n], "ci": ci(loo_boot[n]), "mae_ci": ci(loo_mae[n])}
for n in names:
    d = loo_boot[n] - loo_boot["Human"]
    p = 2 * min(np.mean(d <= 0), np.mean(d >= 0))
    res["loo_diff_vs_human"][n] = {"d_kappa": pt[n] - pt["Human"], "ci": ci(d), "p": float(min(p, 1))}

json.dump(res, open(out, "w"), indent=1)

print(f"Table 2 with 95% CIs (B={B})")
for n in sorted(names, key=lambda n: -res["table2"][n]["kappa_w"]["point"]):
    r = res["table2"][n]
    print(f"  {n:18s} k={r['kappa_w']['point']:.3f} [{r['kappa_w']['ci'][0]:.3f},{r['kappa_w']['ci'][1]:.3f}]"
          f"  mae={r['mae']['point']:.3f} [{r['mae']['ci'][0]:.3f},{r['mae']['ci'][1]:.3f}]"
          f"  bias={r['bias']['point']:+.3f} [{r['bias']['ci'][0]:+.3f},{r['bias']['ci'][1]:+.3f}]  P(rank1)={res['rank1_prob'][n]:.2f}")
print("\nLeave-one-out kappa with CIs; LLM - human difference")
print(f"  {'Human':18s} {pt['Human']:.3f} [{res['loo']['Human']['ci'][0]:.3f},{res['loo']['Human']['ci'][1]:.3f}]")
for n in sorted(names, key=lambda n: -pt[n]):
    d = res["loo_diff_vs_human"][n]
    print(f"  {n:18s} {pt[n]:.3f} [{res['loo'][n]['ci'][0]:.3f},{res['loo'][n]['ci'][1]:.3f}]   diff {d['d_kappa']:+.3f} [{d['ci'][0]:+.3f},{d['ci'][1]:+.3f}] p={d['p']:.3f}")
sig = [(k, v) for k, v in res["paired"].items() if v["p_kappa_holm"] < 0.05]
print(f"\nPairs with Holm-adjusted p<0.05 on kappa: {len(sig)} of {len(res['paired'])}")
top = names_sorted = sorted(names, key=lambda n: -res["table2"][n]["kappa_w"]["point"])
for k, v in res["paired"].items():
    if top[0] in k:
        print(f"  {k:40s} dk={v['d_kappa']:+.3f} [{v['d_kappa_ci'][0]:+.3f},{v['d_kappa_ci'][1]:+.3f}] p={v['p_kappa']:.3f} holm={v['p_kappa_holm']:.3f} | dMAE p_holm={v['p_mae_holm']:.3f}")
