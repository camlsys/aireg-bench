"""A5: per-Article uncertainty in expert disagreement. A6: does plausibility relate to agreement?"""
import json, sys
from collections import Counter
import numpy as np
from scipy.stats import spearmanr
from load import MODELS, ALIAS, ARTS, load_humans_disagg, load_humans_iter, load_llm, load_steering, item_meta
from metrics import median_int, kappa, mae

root, out = sys.argv[1], sys.argv[2]
rng = np.random.default_rng(13); B = 10000
_, D = load_humans_disagg(root)
I = load_humans_iter(root); med = median_int(I).astype(float)
use, art, _ = item_meta(); steer = load_steering(root)
X = {ALIAS[m]: load_llm(root, m) for m in MODELS}
sd = np.nanstd(D, axis=0, ddof=0)
res = {"A5": {}, "A6": {}}

# A5: mean per-excerpt SD by Article, with CIs; permutation test for any difference
for a in ARTS:
    v = sd[art == a]
    bs = [rng.choice(v, len(v)).mean() for _ in range(B)]
    res["A5"][f"Art {a}"] = {"n": int(len(v)), "mean_sd": float(v.mean()), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
stat = lambda lab: np.var([sd[lab == a].mean() for a in ARTS])
s0 = stat(art)
res["A5"]["omnibus_perm_p"] = float(np.mean([stat(rng.permutation(art)) >= s0 for _ in range(B)]))
cells = [(u, a, float(sd[(use == u) & (art == a)].mean())) for u in range(1, 9) for a in ARTS]
res["A5"]["cells_per_item"] = 3
res["A5"]["top_cells"] = sorted(cells, key=lambda t: -t[2])[:5]
# per-Article model MAE (pooled over models) with CIs, to caveat Fig. 3
E = np.vstack([np.abs(X[n] - med) for n in X]).mean(0)
res["A5"]["pooled_mae_by_article"] = {}
for a in ARTS:
    v = E[art == a]; bs = [rng.choice(v, len(v)).mean() for _ in range(B)]
    res["A5"]["pooled_mae_by_article"][f"Art {a}"] = {"mae": float(v.mean()), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}

# A6: plausibility
P = load_humans_iter(root, marker="Plausibility [1-5]")
pmed = np.median(P, axis=0)
res["A6"]["plaus_median_dist"] = {str(k): int(v) for k, v in sorted(Counter(pmed.tolist()).items())}
res["A6"]["rho_plaus_vs_pooled_model_error"] = list(map(float, spearmanr(pmed, E)))
res["A6"]["rho_plaus_vs_expert_sd"] = list(map(float, spearmanr(pmed, sd)))
res["A6"]["rho_plaus_vs_compliance_median"] = list(map(float, spearmanr(pmed, med)))
hi, lo = pmed >= 4, pmed <= 3
res["A6"]["n_hi"], res["A6"]["n_lo"] = int(hi.sum()), int(lo.sum())
res["A6"]["by_model"] = {}
for n, x in X.items():
    res["A6"]["by_model"][n] = {"mae_hi": mae(med[hi], x[hi]), "mae_lo": mae(med[lo], x[lo]),
                                "kappa_hi": kappa(med[hi], x[hi]), "kappa_lo": kappa(med[lo], x[lo])}
d = E[hi].mean() - E[lo].mean()
idx_h, idx_l = np.where(hi)[0], np.where(lo)[0]
bs = [E[rng.choice(idx_h, len(idx_h))].mean() - E[rng.choice(idx_l, len(idx_l))].mean() for _ in range(B)]
res["A6"]["pooled_mae_hi_minus_lo"] = {"diff": float(d), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                       "hi": float(E[hi].mean()), "lo": float(E[lo].mean())}
res["A6"]["steer_vs_plaus"] = {"compliant_mean_plaus": float(pmed[steer == 1].mean()), "noncompliant_mean_plaus": float(pmed[steer == 0].mean())}
json.dump(res, open(out, "w"), indent=1)

print("A5 mean per-excerpt SD by Article:")
for a in ARTS:
    r = res["A5"][f"Art {a}"]; print(f"  Art {a:2d}: {r['mean_sd']:.3f} [{r['ci'][0]:.2f}, {r['ci'][1]:.2f}]")
print("  omnibus permutation p:", round(res["A5"]["omnibus_perm_p"], 3)); print("  top cells:", res["A5"]["top_cells"])
print("  pooled model MAE by Article:", {k: (round(v['mae'], 3), [round(c, 2) for c in v['ci']]) for k, v in res["A5"]["pooled_mae_by_article"].items()})
print("\nA6 plausibility median dist:", res["A6"]["plaus_median_dist"])
for k in ["rho_plaus_vs_pooled_model_error", "rho_plaus_vs_expert_sd", "rho_plaus_vs_compliance_median"]:
    print(f"  {k}: rho={res['A6'][k][0]:.3f} p={res['A6'][k][1]:.3f}")
print("  pooled MAE hi vs lo plausibility:", res["A6"]["pooled_mae_hi_minus_lo"], "n", res["A6"]["n_hi"], res["A6"]["n_lo"])
print("  steer vs plaus:", res["A6"]["steer_vs_plaus"])
for n, r in res["A6"]["by_model"].items(): print(f"   {n:18s} mae hi {r['mae_hi']:.2f} lo {r['mae_lo']:.2f}")
