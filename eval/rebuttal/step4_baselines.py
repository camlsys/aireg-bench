"""Step 4: steering-label and constant baselines, label distributions, steered-subset metrics.
Step 5 checks: Krippendorff alpha variant, alpha without two annotators, SD figures.
"""
import json
import sys
from collections import Counter
from itertools import product

import krippendorff
import numpy as np

from load import MODELS, ALIAS, ARTS, load_humans_disagg, load_humans_iter, load_llm, load_steering, item_meta
from metrics import all_metrics, median_int, kappa, mae

root, out = sys.argv[1], sys.argv[2]
I = load_humans_iter(root)
med = median_int(I)
names, D = load_humans_disagg(root)
steer = load_steering(root)
use, art, _ = item_meta()
X = {ALIAS[m]: load_llm(root, m) for m in MODELS}
res = {}

# ---- distributions ----
res["median_dist"] = {str(k): int(v) for k, v in sorted(Counter(med.tolist()).items())}
res["median_dist_by_steer"] = {
    "steered_compliant": {str(k): int(v) for k, v in sorted(Counter(med[steer == 1].tolist()).items())},
    "steered_noncompliant": {str(k): int(v) for k, v in sorted(Counter(med[steer == 0].tolist()).items())},
}

# ---- constant predictors ----
res["constant"] = {str(c): all_metrics(med, np.full(len(med), c)) for c in range(1, 6)}

# ---- steering-label predictors ----
# (a) a priori mapping: compliant profile -> 5, violation profile -> 2 (scale midpoint of "low/very low")
# (b) a priori: compliant -> 5, violation -> 1
# (c) oracle: best (value for compliant, value for violation) pair on kappa, chosen with the labels (upper bound)
def steer_pred(c_val, v_val):
    return np.where(steer == 1, c_val, v_val)

res["steering"] = {"5/2": all_metrics(med, steer_pred(5, 2)), "5/1": all_metrics(med, steer_pred(5, 1))}
best = max(product(range(1, 6), range(1, 6)), key=lambda p: kappa(med, steer_pred(*p)))
res["steering"]["oracle"] = {"map": list(best), **all_metrics(med, steer_pred(*best))}
# (d) oracle on MAE: median of median labels within each steered group
mc, mv = int(np.median(med[steer == 1])), int(np.median(med[steer == 0]))
res["steering"]["group_median"] = {"map": [mc, mv], **all_metrics(med, steer_pred(mc, mv))}
# Binary agreement: steer label vs expert median >= 4
res["steering_binary_agreement"] = float(np.mean((med >= 4) == (steer == 1)))

# ---- LLMs within each steered subset ----
res["subset"] = {}
for nm, s in {"steered_noncompliant": steer == 0, "steered_compliant": steer == 1}.items():
    res["subset"][nm] = {"n": int(s.sum())}
    for n, x in X.items():
        res["subset"][nm][n] = all_metrics(med[s], x[s])
    res["subset"][nm]["constant_best_mae"] = min(
        ((c, mae(med[s], np.full(s.sum(), c))) for c in range(1, 6)), key=lambda t: t[1])

# ---- Step 5: alpha and SDs ----
res["alpha"] = {}
for lvl in ["ordinal", "interval", "nominal"]:
    res["alpha"][lvl] = float(krippendorff.alpha(reliability_data=D, level_of_measurement=lvl))
# eval.py computes alpha on human_annotations_disaggregated with block_len=15, n_blocks=8 -> same matrix
keep = [i for i, n in enumerate(names) if n not in ("TJ3", "YIQ")]
res["alpha"]["ordinal_without_TJ3_YIQ"] = float(krippendorff.alpha(reliability_data=D[keep], level_of_measurement="ordinal"))
sd0 = np.nanstd(D, axis=0, ddof=0)
sd1 = np.nanstd(D, axis=0, ddof=1)
res["sd_by_article"] = {f"Art {a}": {"ddof0": float(sd0[art == a].mean()), "ddof1": float(sd1[art == a].mean())} for a in ARTS}
res["sd_by_use_article_max"] = max(
    ((u, a, float(sd0[(use == u) & (art == a)].mean()), float(sd1[(use == u) & (art == a)].mean()))
     for u in range(1, 9) for a in ARTS), key=lambda t: t[3])
# Annotator co-assignment structure
cooc = {}
for i in range(D.shape[1]):
    trio = tuple(sorted(n for n, v in zip(names, D[:, i]) if np.isfinite(v)))
    cooc[trio] = cooc.get(trio, []) + [int(use[i])]
res["annotator_trios"] = {"/".join(k): sorted(set(v)) for k, v in cooc.items()}
res["annotator_items"] = {n: int(np.isfinite(D[j]).sum()) for j, n in enumerate(names)}

json.dump(res, open(out, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))

print("median distribution:", res["median_dist"])
print("by steer:", res["median_dist_by_steer"])
print("steer binary agreement with median>=4:", round(res["steering_binary_agreement"], 3))
print("\nconstants:")
for c, r in res["constant"].items():
    print(f"  {c}: kappa={r['kappa_w']:.3f} mae={r['mae']:.3f} bias={r['bias']:+.3f}")
print("steering baselines:")
for k, r in res["steering"].items():
    print(f"  {k:12s} kappa={r['kappa_w']:.3f} rho={r['rho']:.3f} mae={r['mae']:.3f} bias={r['bias']:+.3f} {r.get('map', '')}")
for nm in res["subset"]:
    sub = res["subset"][nm]
    print(f"\n{nm} (n={sub['n']}), best constant MAE {sub['constant_best_mae']}")
    for n in sorted(X, key=lambda n: -np.nan_to_num(sub[n]["kappa_w"])):
        r = sub[n]
        print(f"  {n:18s} kappa={r['kappa_w']:.3f} rho={r['rho']:.3f} mae={r['mae']:.3f} bias={r['bias']:+.3f}")
print("\nalpha:", {k: round(v, 3) for k, v in res["alpha"].items()})
print("SD by article:", {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in res["sd_by_article"].items()})
print("max SD (use, art, ddof0, ddof1):", res["sd_by_use_article_max"])
print("annotator trios -> uses:", res["annotator_trios"])
