"""Alternative Annotator Test (Calderon et al., 2025) on all six annotators and 120 excerpts.

Uses alt_test.py unchanged, in two scoring modes:
  continuous - alignment scored by negative RMSE (main result; treats the 1-5 scale as ordered)
  discrete   - alignment scored by exact match (reported for comparison)
epsilon = 0.1, Benjamini-Yekutieli FDR at 0.05.

Usage (from the repository root): python eval/rebuttal/run_alt_test.py . eval/rebuttal/results/alt_test.json
"""
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import alt_test as at  # noqa: E402

root, out = Path(sys.argv[1]), sys.argv[2]
MODELS = ["gpt5", "4o", "o3", "o3_mini", "sonnet", "pro", "flash", "gemma", "grok4", "grok3_mini"]
H = at.load_human_annotation_matrix(root / "human_annotations_disaggregated.xlsx", "Compliance [1-5]")
res = {}
for mode in ["continuous", "discrete"]:
    res[mode] = {}
    for m in MODELS:
        L = at.load_llm_annotation_series(m, "Compliance [1-5]", root / "llm_annotations")
        h, l = at._prepare_alt_test_inputs(H, L)
        r = at.alt_test(h, l, task_type=mode, epsilon=0.1, fdr_level=0.05)
        res[mode][at.MODEL_ALIASES.get(m, m)] = {
            "n_items": int(h.shape[0]), "n_annotators": int(h.shape[1]),
            "winning_rate": float(r["winning_rate"]),
            "average_advantage_probability": float(r["average_advantage_probability"]),
            "can_replace": bool(r["can_replace"]),
            "corrected_p_values": [float(x) for x in r["corrected_p_values"]],
        }
    print(f"\n{mode}")
    for name, r in sorted(res[mode].items(), key=lambda kv: (-kv[1]["winning_rate"], -kv[1]["average_advantage_probability"])):
        print(f"  {name:18s} winning rate {r['winning_rate']:.2f}  AAP {r['average_advantage_probability']:.3f}  can replace: {r['can_replace']}")
json.dump(res, open(out, "w"), indent=1)
