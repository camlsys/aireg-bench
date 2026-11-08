"""A2: where do models over-estimate compliance?"""
import json, sys
import numpy as np
from load import MODELS, ALIAS, ARTS, load_humans_iter, load_llm, load_steering, item_meta
from metrics import median_int

root, out = sys.argv[1], sys.argv[2]
I = load_humans_iter(root); med = median_int(I).astype(float)
steer = load_steering(root); use, art, scen = item_meta()
X = {ALIAS[m]: load_llm(root, m) for m in MODELS}
names = list(X)
O = np.vstack([X[n] > med for n in names])          # 10 x 120 over-estimates
M = np.vstack([X[n] - med for n in names])          # signed error
res = {}

def rate(mask):
    return {"n": int(mask.sum()), "over_rate_pooled": float(O[:, mask].mean()),
            "mean_over_size": float(M[:, mask][O[:, mask]].mean()) if O[:, mask].any() else None,
            "per_model": {n: float(O[j, mask].mean()) for j, n in enumerate(names)}}

res["by_steer"] = {"steered_compliant": rate(steer == 1), "steered_noncompliant": rate(steer == 0)}
res["by_median"] = {str(v): rate(med == v) for v in range(1, 6)}
res["by_article"] = {f"Art {a}": rate(art == a) for a in ARTS}
res["by_article_noncompliant"] = {f"Art {a}": rate((art == a) & (steer == 0)) for a in ARTS}
res["by_use"] = {f"Use {u}": rate(use == u) for u in range(1, 9)}
# items over-estimated by many models
cnt = O.sum(0)
order = np.argsort(-cnt)
res["top_items"] = [{"use": int(use[i]), "art": int(art[i]), "scen": str(scen[i]), "steer": int(steer[i]),
                     "experts": I[:, i].astype(int).tolist(), "median": int(med[i]), "n_models_over": int(cnt[i]),
                     "mean_llm": float(np.mean([X[n][i] for n in names]))} for i in order[:12]]
res["items_over_by_majority"] = int((cnt >= 6).sum())
# the violation-steered excerpts that experts nonetheless rated 4+
odd = np.where((steer == 0) & (med >= 4))[0]
res["noncompliant_rated_4plus"] = [{"use": int(use[i]), "art": int(art[i]), "scen": str(scen[i]), "experts": I[:, i].astype(int).tolist(),
                                    "llm_scores": {n: int(X[n][i]) for n in names}} for i in odd]
# where on the scale over-estimates land: on violation-steered items, share of LLM scores >= 4 when median <= 3
v = (steer == 0) & (med <= 3)
res["noncompliant_med_le3_llm_ge4"] = {n: float(np.mean(X[n][v] >= 4)) for n in names}
res["noncompliant_med_le3_llm_ge4_pooled"] = float(np.mean([X[n][v] >= 4 for n in names]))
res["n_noncompliant_med_le3"] = int(v.sum())
json.dump(res, open(out, "w"), indent=1)

def show(title, d):
    print(f"\n{title}")
    for k, r in d.items():
        sz = "" if r["mean_over_size"] is None else f" size {r['mean_over_size']:.2f}"
        top = sorted(r["per_model"].items(), key=lambda kv: -kv[1])[:3]
        print(f"  {k:22s} n={r['n']:3d} over {r['over_rate_pooled']*100:5.1f}%{sz}   highest: " + ", ".join(f"{a} {b*100:.0f}%" for a, b in top))
show("By steering", res["by_steer"]); show("By expert median", res["by_median"]); show("By article", res["by_article"])
show("By article (non-compliant-steered only)", res["by_article_noncompliant"]); show("By use", res["by_use"])
print("\nItems over-estimated by >=6 of 10 models:", res["items_over_by_majority"])
for t in res["top_items"]: print("  ", t)
print("\nViolation-steered but median >= 4:")
for t in res["noncompliant_rated_4plus"]: print("  ", t["use"], t["art"], t["scen"], t["experts"], "LLM mean", round(np.mean(list(t["llm_scores"].values())), 2))
print(f"\nOn {res['n_noncompliant_med_le3']} violation-steered excerpts with median <= 3, share of LLM scores >= 4 (i.e. called compliant):",
      f"pooled {res['noncompliant_med_le3_llm_ge4_pooled']*100:.1f}%", {k: round(v*100, 1) for k, v in sorted(res['noncompliant_med_le3_llm_ge4'].items(), key=lambda kv: -kv[1])})
