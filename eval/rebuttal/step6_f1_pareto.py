"""A4: uncertainty for Table 3 (exact/over/under, F1) and for the Fig. 4 Pareto frontier.

F1 follows eval/eval.py: 'compliant' = score >= 4, for both the expert median and the LLM.
Pareto frontier follows eval/eval.py: a model is on it if no other model is at most as expensive
and at least as good on kappa_w, with one of the two strictly better. Prices are eval.py's PRICES (Gemini 2.0 Flash at $0.40).
"""
import json, sys
import numpy as np
from load import MODELS, ALIAS, load_humans_iter, load_llm
from metrics import kappa, median_int

root, out = sys.argv[1], sys.argv[2]
B = 10000
rng = np.random.default_rng(5)
med = median_int(load_humans_iter(root)).astype(float)
X = {m: load_llm(root, m) for m in MODELS}
PRICES = {"grok3_mini": 0.5, "flash": 0.4, "sonnet": 15, "o3_mini": 4.4, "4o": 10, "grok4": 15, "pro": 10, "gpt5": 10, "o3": 8}
N = len(med)


def f1(y, x, t=4):
    g, p = y >= t, x >= t
    tp, fp, fn = np.sum(p & g), np.sum(p & ~g), np.sum(~p & g)
    d = 2 * tp + fp + fn
    return np.nan if d == 0 else 2 * tp / d


def pct(y, x):
    return 100 * np.mean(x == y), 100 * np.mean(x > y), 100 * np.mean(x < y)


def frontier(ks):
    ms = [m for m in MODELS if m in PRICES]
    on = set()
    for m in ms:
        dom = any((PRICES[o] <= PRICES[m] and ks[o] >= ks[m]) and (PRICES[o] < PRICES[m] or ks[o] > ks[m]) for o in ms if o != m)
        if not dom:
            on.add(m)
    return on


def ci(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


idx = rng.integers(0, N, size=(B, N))
res = {"table3": {}, "f1_best_prob": {}, "f1_pairs": {}, "pareto_point": [], "pareto_prob": {}}
F = {m: np.array([f1(med[s], X[m][s]) for s in idx]) for m in MODELS}
P = {m: np.array([pct(med[s], X[m][s]) for s in idx]) for m in MODELS}
for m in MODELS:
    e, o, u = pct(med, X[m])
    res["table3"][ALIAS[m]] = {"exact": e, "exact_ci": ci(P[m][:, 0]), "over": o, "over_ci": ci(P[m][:, 1]),
                               "under": u, "under_ci": ci(P[m][:, 2]), "f1": float(f1(med, X[m])), "f1_ci": ci(F[m])}
Fm = np.vstack([F[m] for m in MODELS])
best = np.nanargmax(Fm, axis=0)
res["f1_best_prob"] = {ALIAS[m]: float(np.mean(best == i)) for i, m in enumerate(MODELS)}
for a, b in [("flash", "pro"), ("flash", "gpt5"), ("flash", "o3"), ("pro", "gpt5")]:
    d = F[a] - F[b]; d = d[np.isfinite(d)]
    res["f1_pairs"][f"{ALIAS[a]} - {ALIAS[b]}"] = {"diff": float(f1(med, X[a]) - f1(med, X[b])), "ci": ci(d),
                                                    "p": float(min(1, 2 * min(np.mean(d <= 0), np.mean(d >= 0))))}
ks = {m: kappa(med, X[m]) for m in MODELS}
res["pareto_point"] = sorted(ALIAS[m] for m in frontier(ks))
cnt = {m: 0 for m in PRICES}
for s in idx:
    for m in frontier({m: kappa(med[s], X[m][s]) for m in MODELS}):
        cnt[m] += 1
res["pareto_prob"] = {ALIAS[m]: c / B for m, c in cnt.items()}
json.dump(res, open(out, "w"), indent=1)

print("Table 3 with CIs")
for m in MODELS:
    r = res["table3"][ALIAS[m]]
    print(f"  {ALIAS[m]:18s} exact {r['exact']:.1f} [{r['exact_ci'][0]:.1f},{r['exact_ci'][1]:.1f}]  over {r['over']:.1f} [{r['over_ci'][0]:.1f},{r['over_ci'][1]:.1f}]"
          f"  under {r['under']:.1f} [{r['under_ci'][0]:.1f},{r['under_ci'][1]:.1f}]  F1 {r['f1']:.3f} [{r['f1_ci'][0]:.3f},{r['f1_ci'][1]:.3f}]  P(best F1)={res['f1_best_prob'][ALIAS[m]]:.2f}")
print("F1 paired diffs:", {k: (round(v['diff'], 3), [round(c, 3) for c in v['ci']], round(v['p'], 3)) for k, v in res["f1_pairs"].items()})
print("Pareto frontier (point):", res["pareto_point"])
print("P(on frontier):", {k: round(v, 3) for k, v in sorted(res["pareto_prob"].items(), key=lambda kv: -kv[1])})
