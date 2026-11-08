"""Generate LaTeX table rows directly from the JSON results, so no number is retyped by hand."""
import json
import sys

d = sys.argv[1]
boot = json.load(open(f"{d}/boot.json"))
loo = json.load(open(f"{d}/loo.json"))
base = json.load(open(f"{d}/baselines.json"))
sub = json.load(open(f"{d}/subset.json"))

ORDER = ["GPT-5", "GPT-4o", "o3", "o3 mini", "Claude Sonnet 4", "Gemini 2.5 Pro", "Gemini 2.0 Flash", "Gemma 3", "Grok 4", "Grok 3 mini"]
CITE = {"GPT-5": "openai2025gpt5", "GPT-4o": "openai2024gpt4ocard", "o3": "openai2025o3o4mini", "o3 mini": "openai2025o3mini",
        "Claude Sonnet 4": "anthropic2025claude4", "Gemini 2.5 Pro": "comanici2025gemini25pushingfrontier",
        "Gemini 2.0 Flash": "google2024gemini2", "Gemma 3": "kamath2025gemma3", "Grok 4": "xai2025grok4",
        "Grok 3 mini": "xai2025grok3"}


def f(x, s=False):
    return f"{x:+.3f}".replace("+", "" if not s else "+") if s else f"{x:.3f}"


def ci(c, nd=2):
    return f"{{\\scriptsize[{c[0]:.{nd}f}, {c[1]:.{nd}f}]}}"


def cell(x):
    return f"\\centering {x}"


out = []
# ---- Table 2 with CIs and baselines ----
t2 = boot["table2"]
best = {k: (max if k in ("kappa_w", "rho") else min)(ORDER, key=lambda n: abs(t2[n][k]["point"]) if k == "bias" else t2[n][k]["point"]) for k in ["kappa_w", "rho", "bias", "mae"]}
out.append("%% ---- Table 2 rows ----")
for n in ORDER:
    r = t2[n]
    cols = []
    for k in ["kappa_w", "rho", "bias", "mae"]:
        v = f"{r[k]['point']:.3f}"
        if best[k] == n:
            v = f"\\underline{{{v}}}"
        cols.append(cell(f"{v} {ci(r[k]['ci'])}"))
    out.append(f"{n} \\citep{{{CITE[n]}}} & " + " & ".join(cols) + " \\tabularnewline")
out.append("\\hline")
s = base["steering"]["5/2"]
out.append("\\textit{Baseline: steering label} (5 if steered compliant, else 2) & " +
           " & ".join(cell(f"{s[k]:.3f}") for k in ["kappa_w", "rho", "bias", "mae"]) + " \\tabularnewline")
c = base["constant"]["3"]
out.append("\\textit{Baseline: constant} (always 3) & " + " & ".join(
    [cell(f"{c['kappa_w']:.3f}"), cell("--"), cell(f"{c['bias']:.3f}"), cell(f"{c['mae']:.3f}")]) + " \\tabularnewline")

# ---- Leave-one-out table ----
out.append("\n%% ---- Leave-one-out rows ----")
L = boot["loo"]
h = L["Human"]
out.append(f"Individual expert (held out) & {cell(format(h["kappa_w"], '.3f') + ' ' + ci(h['ci']))} & {cell('--')} & {cell(format(loo["loo"]["Human (held-out annotator)"]["mae"], '.3f'))} \\tabularnewline")
out.append("\\hline")
for n in sorted(ORDER, key=lambda n: -L[n]["kappa_w"]):
    dv = boot["loo_diff_vs_human"][n]
    p = "$<$0.001" if dv["p"] < 0.001 else "=" + format(dv["p"], ".3f")
    out.append(f"{n} & {cell(format(L[n]["kappa_w"], '.3f') + ' ' + ci(L[n]['ci']))} & "
               f"{cell(format(dv["d_kappa"], '+.3f') + ' ' + ci(dv['ci']) + f' ($p${p})')} & {cell(format(loo["loo"][n]["mae"], '.3f'))} \\tabularnewline")

# ---- Individual annotators table ----
out.append("\n%% ---- LLM vs individual annotators rows (kappa_w) ----")
names = ["L1Q", "SGV", "YIQ", "TJ3", "YO3", "GW7"]
hp = loo["human_pairwise_pooled"]["kappa_w"]
for n in sorted(ORDER, key=lambda n: -loo["llm_vs_individual"][n]["pooled"]["kappa_w"]):
    r = loo["llm_vs_individual"][n]
    out.append(f"{n} & " + " & ".join(cell(f"{r[a]['kappa_w']:.2f}") for a in names) + f" & {cell(format(r["pooled"]["kappa_w"], '.3f'))} \\tabularnewline")
out.append("\\hline")
out.append("Leave-one-out expert & " + " & ".join(cell(f"{loo['per_annotator_loo'][a]['kappa_w']:.2f}") for a in names) +
           f" & {cell(format(boot["loo"]["Human"]["kappa_w"], '.3f'))} \\tabularnewline")
out.append(f"%% human-human pairwise pooled kappa_w = {hp:.3f}")
out.append("%% items per annotator: " + ", ".join(f"{a}={loo['per_annotator_loo'][a]['n']}" for a in names))
out.append("%% per-annotator bias vs mean of other two: " + ", ".join(f"{a}={loo['per_annotator_loo'][a]['bias']:+.3f}" for a in names))

# ---- Steered subset table ----
out.append("\n%% ---- Steered non-compliant subset rows ----")
S, SL = sub["subset_ci"]["steered_noncompliant"], sub["subset_loo"]["steered_noncompliant"]
SM = base["subset"]["steered_noncompliant"]
out.append(f"Individual expert (held out) & {cell('--')} & {cell(format(SL["Human"]["kappa"], '.3f'))} & {cell('--')} & {cell(format(SL["Human"]["mae"], '.3f'))} \\tabularnewline")
out.append("\\hline")
for n in sorted(ORDER, key=lambda n: -S[n]["kappa"]):
    out.append(f"{n} & {cell(format(S[n]["kappa"], '.3f') + ' ' + ci(S[n]['ci']))} & {cell(format(SL[n]["kappa"], '.3f'))} & "
               f"{cell(format(SM[n]["bias"], '+.3f'))} & {cell(format(SM[n]["mae"], '.3f'))} \\tabularnewline")
bc = SM["constant_best_mae"]
out.append("\\hline")
out.append(f"\\textit{{Baseline: constant}} (always {bc[0]}) & {cell('0.000')} & {cell('0.000')} & {cell('--')} & {cell(f'{bc[1]:.3f}')} \\tabularnewline")

print("\n".join(out))
