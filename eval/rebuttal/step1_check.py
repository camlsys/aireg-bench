"""Step 1: align data sources and reproduce Table 2."""
import sys
from collections import Counter

import numpy as np

from load import MODELS, ALIAS, load_humans_disagg, load_humans_iter, load_llm, load_steering
from metrics import all_metrics, kappa, kappa_sklearn, median_int

root = sys.argv[1]
names, D = load_humans_disagg(root)
I = load_humans_iter(root)

print("annotators:", names, "ratings each:", np.isfinite(D).sum(1).tolist())
assert (np.isfinite(D).sum(0) == 3).all()
# Same three scores per item in both files?
bad = [i for i in range(D.shape[1]) if sorted(D[np.isfinite(D[:, i]), i]) != sorted(I[:, i])]
print("items whose disaggregated scores differ from human_annotations.xlsx:", bad)

med = median_int(I)
print("median from disaggregated == median from iter:", bool((median_int(D) == med).all()))
print("median label distribution:", dict(sorted(Counter(med.tolist()).items())))
print("all 360 labels:", dict(sorted(Counter(I.ravel().astype(int).tolist()).items())))

steer = load_steering(root)
print("steered compliant:", int(steer.sum()), "of", len(steer))

print(f"\n{'model':18s} kappa   rho    bias   mae   (sklearn kappa check)")
for m in MODELS:
    x = load_llm(root, m)
    assert np.isfinite(x).all() and len(x) == 120
    r = all_metrics(med, x)
    print(f"{ALIAS[m]:18s} {r['kappa_w']:.3f}  {r['rho']:.3f}  {r['bias']:+.3f}  {r['mae']:.3f}   {kappa_sklearn(med, x):.3f}")
