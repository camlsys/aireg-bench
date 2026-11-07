# Rebuttal analyses (TMLR revision)

Reproduces the uncertainty estimates, human baseline and baselines added in the revision.
From the repository root:

    pip install -r requirements.txt
    python eval/rebuttal/run_all.py

Takes about two minutes. Outputs go to `eval/rebuttal/results/`.

| Script | What it computes |
| --- | --- |
| `step1_check.py` | Loads all annotations, checks the two human files agree, reproduces Table 2, prints the median-label distribution |
| `step2_loo.py` | Leave-one-out human baseline (each expert vs the median of the other two; each LLM against the same references), LLM agreement with individual experts, expert-expert pairwise agreement |
| `step3_bootstrap.py` | 95% bootstrap CIs (10,000 resamples of the 120 excerpts) for Table 2 and the leave-one-out comparison; paired bootstrap tests between models with Holm correction; probability of ranking first |
| `step4_baselines.py` | Steering-label and constant baselines; metrics on excerpts steered towards compliance / non-compliance; Krippendorff's alpha variants; per-article standard deviations; annotator assignment |
| `step4b_subset.py` | Each LLM vs the steering baseline (paired bootstrap); CIs and leave-one-out human baseline on the steered subsets |
| `holm.py` | Holm-adjusted p-values for each model vs the individual expert and vs the steering baseline |
| `make_tables.py` | LaTeX table rows generated from the JSON results |

Item order: index = use × 15 + article × 3 + scenario (Uses 1–8; Articles 9, 10, 12, 14, 15; Scenarios A–C),
matching `eval/eval.py`. The steering label of each excerpt comes from `rand_assign.txt` (1 = compliant profile).

Quadratically weighted kappa is computed as 1 − mean((x−y)²) / mean over all pairs (x_i − y_j)², which equals
scikit-learn's quadratic kappa for integer ratings and is also defined when the reference is a half-point
median of two ratings.
