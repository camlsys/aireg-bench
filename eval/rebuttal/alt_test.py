from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

def _calculate_s_acc(other_annotator_scores: np.ndarray, target_annotation: Union[str, int]) -> float:
    """Calculate the alignment score S for accuracy (discrete tasks)."""
    if other_annotator_scores.size == 0:
        return 0.0
    return np.mean(other_annotator_scores == target_annotation)

def _calculate_s_rmse(other_annotator_scores: np.ndarray, target_annotation: float) -> float:
    """Calculate the alignment score S for -RMSE (continuous tasks)."""
    if other_annotator_scores.size == 0:
        return 0.0
    return -np.sqrt(np.mean((other_annotator_scores - target_annotation)**2))

def _calculate_s_sim(other_annotator_scores: np.ndarray, target_annotation: str, similarity_metric: callable) -> float:
    """Calculate the alignment score S for similarity (free-text tasks)."""
    if other_annotator_scores.size == 0:
        return 0.0
    similarities = [similarity_metric(target_annotation, str(h_anno)) for h_anno in other_annotator_scores]
    return np.mean(similarities)

def alt_test(
    human_annotations: pd.DataFrame,
    llm_annotations: pd.Series,
    task_type: str,
    epsilon: float = 0.1,
    fdr_level: float = 0.05,
    similarity_metric: Optional[Callable[[str, str], float]] = None,
) -> Dict[str, Any]:
    """
    Performs the Alternative Annotator Test (alt-test).

    Args:
        human_annotations: DataFrame where columns are annotators and rows are items.
        llm_annotations: Series with LLM annotations, indexed by item.
        task_type: 'discrete', 'continuous', or 'free-text'.
        epsilon: Cost-benefit hyperparameter.
        fdr_level: False Discovery Rate level for correction.
        similarity_metric: A function for 'free-text' tasks to compute similarity.

    Returns:
        A dictionary with the test results.
    """
    annotators = list(human_annotations.columns)
    p_values: list[float] = []
    rho_f_values: list[float] = []

    if task_type == 'discrete':
        score_func = _calculate_s_acc
    elif task_type == 'continuous':
        score_func = _calculate_s_rmse
    elif task_type == 'free-text':
        if not callable(similarity_metric):
            raise ValueError("A similarity_metric function must be provided for free-text tasks.")
        score_func = lambda annotations, llm_annotation: _calculate_s_sim(annotations, llm_annotation, similarity_metric)
    else:
        raise ValueError("task_type must be one of 'discrete', 'continuous', or 'free-text'")

    for j_annotator in annotators:
        other_annotators = [h for h in annotators if h != j_annotator]
        
        # Get items annotated by j_annotator
        items_annotated_by_j = human_annotations[j_annotator].dropna().index
        
        d_i_j = []
        w_f_i_j_list = []

        for item in items_annotated_by_j:
            # Annotations from other humans for this item
            h_minus_j_annotations = human_annotations.loc[item, other_annotators].dropna().values
            
            if len(h_minus_j_annotations) < 1:
                continue

            llm_anno = llm_annotations.loc[item]
            
            if task_type == 'free-text':
                s_f = score_func(h_minus_j_annotations, llm_anno)
                s_h = score_func(h_minus_j_annotations, human_annotations.loc[item, j_annotator])
            else:
                s_f = score_func(h_minus_j_annotations, llm_anno)
                s_h = score_func(h_minus_j_annotations, human_annotations.loc[item, j_annotator])


            w_f_i_j = 1 if s_f >= s_h else 0
            w_h_i_j = 1 if s_h >= s_f else 0
            
            w_f_i_j_list.append(w_f_i_j)
            d_i_j.append(w_h_i_j - w_f_i_j)

        if not d_i_j:
            p_values.append(1.0)
            rho_f_values.append(0.0)
            continue
            
        rho_f_j = np.mean(w_f_i_j_list)
        rho_f_values.append(rho_f_j)

        # Paired t-test
        diff = np.array(d_i_j)
        
        if len(diff) < 30:
            # Per Calderon et al. (2025), switch to a Wilcoxon signed-rank test when the normality
            # assumption of the paired t-test is implausible (small sample sizes).
            try:
                _, p_value = stats.wilcoxon(diff - epsilon, alternative="less")
            except ValueError:
                # All differences are identical; treat as non-significant.
                p_value = 1.0
        else:
            _, p_value = stats.ttest_1samp(diff, epsilon, alternative="less")
            if np.isnan(p_value):
                p_value = 1.0

        p_values.append(p_value)

    # Multiple comparison correction (Benjamini-Yekutieli)
    rejected, corrected_p_values, _, _ = multipletests(p_values, alpha=fdr_level, method='fdr_by')

    winning_rate = np.sum(rejected) / len(annotators)
    
    avg_advantage_prob = np.mean(rho_f_values)

    return {
        "winning_rate": winning_rate,
        "average_advantage_probability": avg_advantage_prob,
        "p_values": p_values,
        "corrected_p_values": corrected_p_values,
        "rejected_hypotheses": rejected,
        "can_replace": winning_rate >= 0.5
    }


DATA_START_ROW = 2
HEADER_ROW_USE = 0
HEADER_ROW_MARKER = 1
LABEL_COLUMN = 0

HUMAN_ANNOTATIONS_FILE = Path("human_annotations_disaggregated.xlsx")
LLM_ANNOTATIONS_DIR = Path("llm_annotations")

DEFAULT_MODELS: list[str] = [
    "gpt5",
    "4o",
    "o3",
    "o3_mini",
    "sonnet",
    "pro",
    "flash",
    "gemma",
    "grok4",
    "grok3_mini",
    "llama3-70b",
    "kimi-k2-instruct",
]

MODEL_ALIASES: dict[str, str] = {
    "grok3_mini": "Grok 3 mini",
    "gemma": "Gemma 3",
    "flash": "Gemini 2.5 Flash",
    "sonnet": "Sonnet 4",
    "o3_mini": "o3 mini",
    "4o": "GPT-4o",
    "pro": "Gemini 2.5 Pro",
    "grok4": "Grok 4",
    "gpt5": "GPT-5",
    "o3": "o3",
    "llama3-70b": "Llama 3 70B",
    "kimi-k2-instruct": "Kimi K2 Instruct",
}

TASK_CONFIGS: dict[str, dict[str, Union[str, float]]] = {
    "compliance": {
        "score_marker": "Compliance [1-5]",
        "task_type": "discrete",
        "default_epsilon": 0.1,
    },
    "plausibility": {
        "score_marker": "Plausibility [1-5]",
        "task_type": "discrete",
        "default_epsilon": 0.1,
    },
}


def _sanitize_string(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if pd.isna(value):
        return ""
    return str(value).strip()


def _extract_score_series(workbook_path: Path, sheet_name: str, score_marker: str) -> pd.Series:
    df = pd.read_excel(workbook_path, sheet_name=sheet_name, header=None)
    if df.empty:
        return pd.Series(dtype=float, name=sheet_name)

    # Ensure the label column is forward-filled so every row has a scenario name.
    df.iloc[:, LABEL_COLUMN] = df.iloc[:, LABEL_COLUMN].ffill()
    scenario_labels = df.iloc[DATA_START_ROW:, LABEL_COLUMN].astype(str)

    scores: dict[str, float] = {}
    for col_idx in range(df.shape[1]):
        marker_value = _sanitize_string(df.iloc[HEADER_ROW_MARKER, col_idx])
        if marker_value != score_marker:
            continue

        use_name = _sanitize_string(df.iloc[HEADER_ROW_USE, col_idx]) or f"Use@{col_idx}"
        col_scores = pd.to_numeric(df.iloc[DATA_START_ROW:, col_idx], errors="coerce")

        for label, value in zip(scenario_labels, col_scores):
            if pd.isna(value):
                continue
            label_clean = label.strip()
            if not label_clean or label_clean.lower() == "nan":
                continue
            item_id = f"{use_name}|{label_clean}"
            scores[item_id] = float(value)

    series = pd.Series(scores, dtype=float, name=sheet_name)
    return series.sort_index()


def load_human_annotation_matrix(workbook_path: Path, score_marker: str) -> pd.DataFrame:
    xls = pd.ExcelFile(workbook_path)
    series_list: list[pd.Series] = []
    for sheet in xls.sheet_names:
        series = _extract_score_series(workbook_path, sheet, score_marker)
        if series.empty:
            continue
        series.name = sheet
        series_list.append(series)

    if not series_list:
        raise ValueError(f"No human annotations found for marker '{score_marker}'.")

    humans_df = pd.concat(series_list, axis=1).sort_index()
    humans_df.index.name = "item_id"
    return humans_df


def _resolve_llm_path(model_name: str, base_dir: Path) -> Path:
    candidates = [
        base_dir / f"{model_name}_annotations_new.xlsx",
        base_dir / f"{model_name}_annotations.xlsx",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Annotation file for model '{model_name}' not found in {base_dir}.")


def load_llm_annotation_series(model_name: str, score_marker: str, base_dir: Path) -> pd.Series:
    llm_path = _resolve_llm_path(model_name, base_dir)
    sheet_name = pd.ExcelFile(llm_path).sheet_names[0]
    series = _extract_score_series(llm_path, sheet_name, score_marker)
    series.name = model_name
    return series


def _prepare_alt_test_inputs(
    human_df: pd.DataFrame,
    llm_series: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    llm_series = llm_series.rename("LLM")
    combined = human_df.join(llm_series, how="inner")
    if "LLM" not in combined.columns or combined.empty:
        return pd.DataFrame(), pd.Series(dtype=float)

    combined = combined.dropna(subset=["LLM"])
    human_only = combined.drop(columns=["LLM"])
    # Require at least two human annotations per item so leave-one-out comparisons work.
    mask = human_only.notna().sum(axis=1) >= 2
    human_only = human_only.loc[mask]
    llm_clean = combined.loc[mask, "LLM"]

    human_only = human_only.dropna(axis=1, how="all")
    if human_only.empty or llm_clean.empty:
        return pd.DataFrame(), pd.Series(dtype=float)

    return human_only, llm_clean


def run_alt_test_suite(
    human_df: pd.DataFrame,
    models: Sequence[str],
    score_marker: str,
    task_type: str,
    epsilon: float,
    fdr_level: float,
    llm_dir: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_name in models:
        try:
            llm_series = load_llm_annotation_series(model_name, score_marker, llm_dir)
        except FileNotFoundError as exc:
            print(f"[WARN] {exc}")
            continue

        aligned_humans, aligned_llm = _prepare_alt_test_inputs(human_df, llm_series)
        if aligned_humans.empty or aligned_llm.empty:
            print(f"[WARN] Insufficient overlapping data for model '{model_name}'. Skipping.")
            continue

        if aligned_humans.shape[1] < 3:
            print(
                f"[WARN] Model '{model_name}' has fewer than three effective annotators after filtering."
                " Alt-test requires at least three.",
            )
            continue

        result = alt_test(
            human_annotations=aligned_humans,
            llm_annotations=aligned_llm,
            task_type=task_type,
            epsilon=epsilon,
            fdr_level=fdr_level,
        )

        rows.append(
            {
                "model_id": model_name,
                "model": MODEL_ALIASES.get(model_name, model_name),
                "num_items": int(aligned_humans.shape[0]),
                "num_annotators": int(aligned_humans.shape[1]),
                "winning_rate": float(result["winning_rate"]),
                "average_advantage_probability": float(result["average_advantage_probability"]),
                "can_replace": bool(result["can_replace"]),
            },
        )

    return pd.DataFrame(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Alternative Annotator Test on real annotations.")
    parser.add_argument(
        "--task",
        choices=list(TASK_CONFIGS.keys()),
        default="compliance",
        help="Which annotation dimension to evaluate (default: compliance).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Subset of model IDs to evaluate (default: all supported models).",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="Override the cost-benefit epsilon (defaults to per-task recommendation).",
    )
    parser.add_argument(
        "--fdr",
        type=float,
        default=0.05,
        help="Target false discovery rate for the Benjamini-Yekutieli correction (default: 0.05).",
    )
    parser.add_argument(
        "--human-file",
        type=Path,
        default=HUMAN_ANNOTATIONS_FILE,
        help="Path to the disaggregated human annotation workbook.",
    )
    parser.add_argument(
        "--llm-dir",
        type=Path,
        default=LLM_ANNOTATIONS_DIR,
        help="Directory containing LLM annotation workbooks.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional path to save the summary table as CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    task_cfg = TASK_CONFIGS[args.task]
    epsilon = args.epsilon if args.epsilon is not None else float(task_cfg["default_epsilon"])

    print(f"[INFO] Loading human annotations from {args.human_file} ({args.task}).")
    human_df = load_human_annotation_matrix(args.human_file, str(task_cfg["score_marker"]))

    models = args.models or DEFAULT_MODELS
    print(f"[INFO] Evaluating models: {', '.join(models)}")

    summary_df = run_alt_test_suite(
        human_df=human_df,
        models=models,
        score_marker=str(task_cfg["score_marker"]),
        task_type=str(task_cfg["task_type"]),
        epsilon=epsilon,
        fdr_level=args.fdr,
        llm_dir=args.llm_dir,
    )

    if summary_df.empty:
        print("[WARN] No results were produced. Please check data availability.")
        return

    summary_df = summary_df.sort_values("winning_rate", ascending=False).reset_index(drop=True)
    print("\n=== Alt-Test Results ===")
    print(summary_df.to_string(index=False))

    if args.output_csv:
        summary_df.to_csv(args.output_csv, index=False)
        print(f"[INFO] Saved results to {args.output_csv}")


if __name__ == "__main__":
    main()
