"""Shared loaders for the AIReg-Bench rebuttal analyses.

Item order everywhere: index = use*15 + article*3 + scenario, with
use in 0..7 (Uses 1-8), article in [9, 10, 12, 14, 15], scenario in A, B, C.
This matches eval/eval.py (blocks read column by column, 15 rows each).
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

ARTS = [9, 10, 12, 14, 15]
SCEN = ["A", "B", "C"]
N_USES = 8
N = N_USES * len(ARTS) * len(SCEN)  # 120

MODELS = ["gpt5", "4o", "o3", "o3_mini", "sonnet", "pro", "flash", "gemma", "grok4", "grok3_mini"]
ALIAS = {"gpt5": "GPT-5", "4o": "GPT-4o", "o3": "o3", "o3_mini": "o3 mini", "sonnet": "Claude Sonnet 4",
         "pro": "Gemini 2.5 Pro", "flash": "Gemini 2.0 Flash", "gemma": "Gemma 3", "grok4": "Grok 4",
         "grok3_mini": "Grok 3 mini"}


def _sheet_matrix(df, marker="Compliance [1-5]"):
    """Return a length-120 vector (NaN where empty) from one sheet laid out like the repo files."""
    sub = df.iloc[1].fillna("").astype(str)
    cols = [c for c in df.columns if sub[c] == marker][:N_USES]
    assert len(cols) == N_USES, f"expected {N_USES} use blocks, found {len(cols)}"
    block = df.iloc[2:17][cols].apply(pd.to_numeric, errors="coerce").to_numpy()  # 15 x 8
    assert block.shape == (15, N_USES)
    return block.T.reshape(-1)  # use-major


def load_humans_disagg(root, marker="Compliance [1-5]"):
    """Annotators x items matrix, NaN where an annotator did not rate an item."""
    path = Path(root) / "human_annotations_disaggregated.xlsx"
    xls = pd.ExcelFile(path)
    names, rows = [], []
    for sh in xls.sheet_names:
        rows.append(_sheet_matrix(pd.read_excel(xls, sheet_name=sh, header=None), marker))
        names.append(sh.replace("Annotator ", ""))
    return names, np.vstack(rows)


def load_humans_iter(root, marker="Compliance [1-5]"):
    """3 x items matrix from human_annotations.xlsx (the file eval.py uses)."""
    path = Path(root) / "human_annotations.xlsx"
    xls = pd.ExcelFile(path)
    return np.vstack([_sheet_matrix(pd.read_excel(xls, sheet_name=s, header=None), marker)
                      for s in xls.sheet_names])


def load_llm(root, model, marker="Compliance [1-5]"):
    path = Path(root) / "llm_annotations" / f"{model}_annotations.xlsx"
    xls = pd.ExcelFile(path)
    return _sheet_matrix(pd.read_excel(xls, sheet_name=xls.sheet_names[0], header=None), marker)


def load_steering(root):
    """1 if the excerpt was generated from the Compliance profile, 0 if from a Violation profile."""
    mapping, use, art = {}, None, None
    for line in (Path(root) / "rand_assign.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        m = re.match(r"Use\s+(\d+),\s*Article\s+(\d+)", line)
        if m:
            use, art = map(int, m.groups())
            continue
        m = re.match(r"(\d+)\s*--\s*([ABC])", line)
        if m:
            mapping[(use, art, m.group(2))] = int(m.group(1))
    steer = np.zeros(N, dtype=int)
    for u in range(N_USES):
        for a_i, a in enumerate(ARTS):
            for s_i, s in enumerate(SCEN):
                steer[u * 15 + a_i * 3 + s_i] = 1 if mapping[(u + 1, a, s)] == 1 else 0
    return steer


def item_meta():
    use = np.repeat(np.arange(1, N_USES + 1), 15)
    art = np.tile(np.repeat(ARTS, 3), N_USES)
    scen = np.tile(SCEN, N_USES * len(ARTS))
    return use, art, scen
