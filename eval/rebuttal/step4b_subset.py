"""Step 4b: CIs for steering baseline vs LLMs, and the leave-one-out human baseline within steered subsets."""
import json
import sys

import numpy as np

from load import MODELS, ALIAS, load_humans_disagg, load_humans_iter, load_llm, load_steering
from metrics import kappa, mae, median_int

root, out = sys.argv[1], sys.argv[2]
B = 10000
rng = np.random.default_rng(7)
med = median_int(load_humans_iter(root)).astype(float)
_, D = load_humans_disagg(root)
steer = load_steering(root)
X = {ALIAS[m]: load_llm(root, m) for m in MODELS}
SB = np.where(steer == 1, 5.0, 2.0)
N = len(med)


def ci(a):
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


res = {"vs_steering": {}, "subset_ci": {}, "subset_loo": {}}

# Paired bootstrap: each LLM minus steering baseline (5/2), full set
idx = rng.integers(0, N, size=(B, N))
kb = np.array([kappa(med[s], SB[s]) for s in idx])
mb = np.array([mae(med[s], SB[s]) for s in idx])
res["steering_ci"] = {"kappa": ci(kb), "mae": ci(mb)}
for n, x in X.items():
    kx = np.array([kappa(med[s], x[s]) for s in idx])
    mx = np.array([mae(med[s], x[s]) for s in idx])
    dk, dm = kx - kb, mx - mb
    res["vs_steering"][n] = {
        "d_kappa": kappa(med, x) - kappa(med, SB), "d_kappa_ci": ci(dk),
        "p_kappa": float(min(1, 2 * min(np.mean(dk <= 0), np.mean(dk >= 0)))),
        "d_mae": mae(med, x) - mae(med, SB), "d_mae_ci": ci(dm),
        "p_mae": float(min(1, 2 * min(np.mean(dm <= 0), np.mean(dm >= 0)))),
    }

# Within-subset: LLM kappa CIs and LOO human baseline on the same slots
for nm, mask in {"steered_noncompliant": steer == 0, "steered_compliant": steer == 1}.items():
    items = np.where(mask)[0]
    si, sh, sr = [], [], []
    for i in items:
        r = np.where(np.isfinite(D[:, i]))[0]
        for a in r:
            si.append(i); sh.append(D[a, i]); sr.append(np.mean(D[[b for b in r if b != a], i]))
    si, sh, sr = map(np.array, (si, sh, sr))
    res["subset_loo"][nm] = {"Human": {"kappa": kappa(sr, sh), "mae": mae(sr, sh)}}
    for n, x in X.items():
        res["subset_loo"][nm][n] = {"kappa": kappa(sr, x[si]), "mae": mae(sr, x[si])}
    sub_idx = rng.choice(items, size=(B, len(items)))
    res["subset_ci"][nm] = {}
    for n, x in X.items():
        ks = np.array([kappa(med[s], x[s]) for s in sub_idx])
        res["subset_ci"][nm][n] = {"kappa": kappa(med[items], x[items]), "ci": ci(ks[np.isfinite(ks)])}

json.dump(res, open(out, "w"), indent=1)
print("steering baseline kappa CI:", [round(v, 3) for v in res["steering_ci"]["kappa"]])
print("\nLLM minus steering baseline (5/2), full set")
for n in sorted(X, key=lambda n: -res["vs_steering"][n]["d_kappa"]):
    r = res["vs_steering"][n]
    print(f"  {n:18s} dk={r['d_kappa']:+.3f} [{r['d_kappa_ci'][0]:+.3f},{r['d_kappa_ci'][1]:+.3f}] p={r['p_kappa']:.3f}"
          f" | dMAE={r['d_mae']:+.3f} [{r['d_mae_ci'][0]:+.3f},{r['d_mae_ci'][1]:+.3f}] p={r['p_mae']:.3f}")
for nm in res["subset_loo"]:
    print(f"\n{nm}: leave-one-out kappa / MAE (human vs LLMs), and kappa vs median-of-3 with CI")
    L = res["subset_loo"][nm]
    print(f"  {'Human':18s} loo k={L['Human']['kappa']:.3f} mae={L['Human']['mae']:.3f}")
    for n in sorted(X, key=lambda n: -L[n]["kappa"]):
        c = res["subset_ci"][nm][n]
        print(f"  {n:18s} loo k={L[n]['kappa']:.3f} mae={L[n]['mae']:.3f} | k_med3={c['kappa']:.3f} [{c['ci'][0]:.3f},{c['ci'][1]:.3f}]")
