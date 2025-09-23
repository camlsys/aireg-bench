"""
This script compares the compliance assessments of humans (from "human_annotations.xlsx") and multiple LLMs (from "{model}_annotations.xlsx"). 
It computes several metrics related to agreement and accuracy between the LLMs and the median human rating.
It also calculates several other metrics, such as inter-rater reliability for the human data.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score
from scipy.stats import spearmanr
import krippendorff

# ---------- Parameters ----------------------------------------------
models = ["gpt5","4o","o3","o3_mini","sonnet","pro","flash","gemma","grok4","grok3_mini"]
human_file = Path("human_annotations.xlsx")
human_file_disaggregated = Path("human_annotations_disaggregated.xlsx")
llm_pattern = "llm_annotations/{model}_annotations.xlsx"
ABLATION = True
MODEL_ALIASES = {
    "grok3_mini": "Grok 3 mini",
    "gemma": "Gemma 3",
    "flash": "Gemini 2.5 Flash",
    "sonnet":  "Sonnet 4",
    "o3_mini": "o3 mini",
    "4o": "GPT-4o",
    "pro": "Gemini 2.5 Pro",
    "grok4": "Grok 4",
    "gpt5": "GPT-5",
    "o3": "o3"}
PRICES = {
   "grok3_mini": 0.5,
   "flash": 2.5,
   "sonnet":  15,
   "o3_mini": 4.4,
   "4o": 10,
   "grok4": 15,
   "pro": 10,
   "gpt5": 10,
   "o3": 8}
ART_LABELS = ["Art 9", "Art 10", "Art 12", "Art 14", "Art 15"]
ITEMS_PER_ARTICLE = 3
ARTICLES_PER_USE = len(ART_LABELS)
ITEMS_PER_USE = ARTICLES_PER_USE * ITEMS_PER_ARTICLE
NUM_USES = 8
COMPLIANCE_MARKERS = ["Compliance [1-5]"]
PLAUSIBILITY_MARKERS = ["Plausibility [1-5]"]

# ---------- Parse and load  ----------------------------------------------
def find_marker_positions(df: pd.DataFrame, markers: set[str]) -> list[tuple[int, int]]:
    mask = df.isin(markers)
    rows, cols = np.where(mask.to_numpy())
    return list(zip(rows, cols))

def extract_block(df: pd.DataFrame, r: int, c: int, block_len: int | None = None) -> np.ndarray:
    values = pd.to_numeric(df.iloc[r + 1 :, c], errors="coerce")
    values = values.dropna().astype(float).to_numpy()
    if block_len is not None:
        if len(values) >= block_len:
            return values[:block_len]
        return np.pad(values, (0, block_len - len(values)), constant_values=np.nan)
    return values

def extract_blocks(df: pd.DataFrame, markers: set[str], block_len: int | None = None, n_blocks: int | None = None) -> np.ndarray:
    positions = find_marker_positions(df, markers)
    vecs = [extract_block(df, r, c, block_len) for r, c in positions]
    if n_blocks is not None:
        vecs = vecs[:n_blocks]
        while len(vecs) < n_blocks:
            filler = np.full(block_len, np.nan, dtype=float) if block_len else np.array([], dtype=float)
            vecs.append(filler)
    return np.concatenate(vecs) if vecs else np.array([], dtype=float)

def extract_sheet_vectors(path: Path, sheet_name: str, block_len: int | None = None, n_blocks: int | None = None) -> dict[str, np.ndarray]:
    df = pd.read_excel(path, sheet_name=sheet_name, header=None)
    comp = extract_blocks(df, COMPLIANCE_MARKERS, block_len, n_blocks)
    plau = extract_blocks(df, PLAUSIBILITY_MARKERS, block_len, n_blocks)
    return {"compliance": comp, "plausibility": plau}

def _build_human_matrix(humans: list[np.ndarray], max_n: int) -> np.ndarray:
    H = np.full((len(humans), max_n), np.nan, dtype=float)
    for i, v in enumerate(humans):
        m = min(len(v), max_n)
        H[i, :m] = v[:m]
    return H

def load_humans(human_path: Path):
    xls = pd.ExcelFile(human_path)
    c_list, p_list = [], []
    for sh in xls.sheet_names:
        hv = extract_sheet_vectors(human_path, sh)
        c_list.append(hv["compliance"])
        p_list.append(hv["plausibility"])
    return c_list, p_list

def load_humans_with_nans(human_path: Path, block_len: int = 15, n_blocks: int = 8):
    xls = pd.ExcelFile(human_path)
    c_list, p_list = [], []
    for sh in xls.sheet_names:
        hv = extract_sheet_vectors(human_path, sh, block_len=block_len, n_blocks=n_blocks)
        c_list.append(hv["compliance"])
        p_list.append(hv["plausibility"])
    shape = (0, n_blocks * block_len)
    Hc = np.vstack(c_list) if c_list else np.empty(shape)
    Hp = np.vstack(p_list) if p_list else np.empty(shape)
    return Hc, Hp

# ---------- Calculate and collect statistical measures ----------------------------------
def median(humans: list[np.ndarray], llm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    max_n = max([len(v) for v in humans] + [len(llm)])
    H = _build_human_matrix(humans, max_n=max_n)
    med_int = np.nanmedian(H, axis=0).astype(int)
    llm_int = llm.astype(int)
    return llm_int, med_int

def alpha(humans: list[np.ndarray]) -> float:
    max_n = max(len(v) for v in humans)
    H = _build_human_matrix(humans, max_n=max_n)
    return krippendorff.alpha(reliability_data=H, level_of_measurement="ordinal")

def kappa_vs_median(humans: np.ndarray, llm: np.ndarray) -> float:
    return float(cohen_kappa_score(llm, humans, weights="quadratic"))

def spearman_vs_median(humans: np.ndarray, llm: np.ndarray) -> float:
    if np.all(humans == humans[0]) or np.all(llm == llm[0]):
        return float("nan")
    rho, _ = spearmanr(llm, humans)
    return float(rho)

def mean_diff_vs_median(humans: np.ndarray, llm: np.ndarray) -> float:
    return float(np.mean(llm - humans))

def mae_vs_median(humans: np.ndarray, llm: np.ndarray) -> float:
    return float(np.mean(np.abs(llm - humans)))

def exact_match_vs_median(humans: np.ndarray, llm: np.ndarray) -> float:
    return float(100.0 * np.mean(llm == humans))

def pct_over_vs_median(humans: np.ndarray, llm: np.ndarray) -> float:
    return float(100.0 * np.mean(llm > humans))

def pct_under_vs_median(humans: np.ndarray, llm: np.ndarray) -> float:
    return float(100.0 * np.mean(llm < humans))

def f1_vs_median(humans: np.ndarray | list[np.ndarray], llm: np.ndarray, high_threshold: int = 4) -> float:
    humans = np.asarray(humans)
    gt_pos = humans >= high_threshold
    pred_pos = llm >= high_threshold
    TP = np.sum(pred_pos & gt_pos)
    FP = np.sum(pred_pos & ~gt_pos)
    FN = np.sum(~pred_pos & gt_pos)
    denom = 2 * TP + FP + FN
    return np.nan if denom == 0 else float(2 * TP / denom)

def mae_by_article_use(humans: list[np.ndarray],
                       llm: np.ndarray,
                       num_uses: int = NUM_USES,
                       items_per_use: int = ITEMS_PER_USE,
                       items_per_article: int = ITEMS_PER_ARTICLE,
                       art_labels: list[str] = ART_LABELS) -> pd.DataFrame:
    llm_round, med_round = median(humans, llm)
    n_expected = num_uses * items_per_use
    n = min(len(llm_round), len(med_round), n_expected)
    maes = np.full((len(art_labels), num_uses), np.nan, dtype=float)
    for u in range(num_uses):
        base = u * items_per_use
        for a_idx, _ in enumerate(art_labels):
            s = base + a_idx * items_per_article
            e = min(s + items_per_article, n)
            llm_chunk = llm_round[s:e]
            med_chunk = med_round[s:e]
            if llm_chunk.size == 0 or med_chunk.size == 0:
                continue
            mask = ~np.isnan(llm_chunk) & ~np.isnan(med_chunk)
            if np.any(mask):
                maes[a_idx, u] = np.mean(np.abs(llm_chunk[mask] - med_chunk[mask]))
    cols = [f"Use{u+1}" for u in range(num_uses)]
    return pd.DataFrame(maes, index=art_labels, columns=cols)

def score_counts_by_usecase(human_path: Path,
                            block_len: int = ITEMS_PER_USE,
                            n_blocks: int = NUM_USES) -> pd.DataFrame:
    Hc, _ = load_humans_with_nans(human_path, block_len=block_len, n_blocks=n_blocks)
    num_annotators, total_items = Hc.shape
    assert total_items == n_blocks * block_len, "Unexpected shape for Hc"
    results = {}
    for use in range(n_blocks):
        start = use * block_len
        end = start + block_len
        ratings = Hc[:, start:end].ravel()
        ratings = ratings[~np.isnan(ratings)].astype(int)
        counts = pd.Series(ratings).value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)
        results[f"Use{use+1}"] = counts
    df_counts = pd.DataFrame(results)
    return df_counts

def evaluate_model(human_c_list, human_p_list, model_name: str, ablation_suffix: str | None = None):
    metrics = {
        "Spearman ρ": spearman_vs_median,
        "Cohen’s κ (w)": kappa_vs_median,
        "Mean difference": mean_diff_vs_median,
        "MAE": mae_vs_median,
        "Exact match (%)": exact_match_vs_median,
        "Overestimation (%)": pct_over_vs_median,
        "Underestimation (%)": pct_under_vs_median,
        "F1 score": f1_vs_median,
    }
    if ablation_suffix:
        llm_path = Path(f"llm_annotations/{model_name}_annotations_ablation_{ablation_suffix}.xlsx")
        tag = f"{model_name}-Ablation-{ablation_suffix}"
    else:
        llm_path = Path(llm_pattern.format(model=model_name))
        tag = model_name
    sheet = pd.ExcelFile(llm_path).sheet_names[0]
    llm = extract_sheet_vectors(llm_path, sheet)
    rows, results = [], {}
    for label, (humans, llm_vec) in {
        "Compliance": (human_c_list, llm["compliance"]),
        "Plausibility": (human_p_list, llm["plausibility"]),
    }.items():
        llm_vec, human_med = median(humans, llm_vec)
        for stat, func in metrics.items():
            value = func(human_med, llm_vec)
            rows.append({"Model": tag, "Metric": label, "Stat": stat, "Value": value})
            results[f"{label}_{stat}"] = value
    alias = MODEL_ALIASES.get(tag, tag)
    print(f"\n=== {alias} ===")
    for k, v in results.items():
        if k.startswith("Compliance_"):
            print(f"{k:30s}: {v:.3f}")
    return pd.DataFrame(rows), (tag, results)

def get_avg_mae_df(human_c_list,
                   model_names,
                   llm_pattern,
                   mae_by_article_use_fn=mae_by_article_use,
                   num_uses=NUM_USES,
                   items_per_use=ITEMS_PER_USE,
                   items_per_article=ITEMS_PER_ARTICLE,
                   art_labels=ART_LABELS) -> pd.DataFrame:
    maes = []
    for m in model_names:
        llm_path = Path(llm_pattern.format(model=m))
        sheet = pd.ExcelFile(llm_path).sheet_names[0]
        llm_vecs = extract_sheet_vectors(llm_path, sheet)
        maes.append(
            mae_by_article_use_fn(
                human_c_list, llm_vecs["compliance"],
                num_uses=num_uses,
                items_per_use=items_per_use,
                items_per_article=items_per_article,
                art_labels=art_labels
            )
        )
    return maes[0] if len(maes) == 1 else sum(maes) / len(maes)

def print_mae_rowcol_avgs(mae_df: pd.DataFrame, title="Compliance MAE"):
    col_means = mae_df.mean(axis=0).round(3)  # per use
    row_means = mae_df.mean(axis=1).round(3)  # per article
    uses_line = "  ".join(f"{k}={v:.3f}" for k, v in col_means.items())
    arts_line = "  ".join(f"{k}={v:.3f}" for k, v in row_means.items())
    print(f"{title} — per use: {uses_line}")
    print(f"{title} — per article: {arts_line}")

def summarise_human_plausibility(human_p_disagg: np.ndarray):
    flat_scores = human_p_disagg.ravel()
    flat_scores = flat_scores[~np.isnan(flat_scores)]
    mean_score = np.mean(flat_scores)
    median_score = np.median(flat_scores)
    print("\n=== Human Plausibility Scores ===")
    print(f"Mean plausibility:   {mean_score:.3f}")
    print(f"Median plausibility: {median_score:.3f}")
    return mean_score, median_score

# ---------- Plotting functions ----------------------------------
sns.set_theme(style="white")
plt.rcParams.update({
    "axes.edgecolor": "black",
    "axes.linewidth": 1.4,
    "figure.dpi": 150,
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

def _pareto_mask(df):
    sdf = df.reset_index(drop=False).sort_values(["Price", "Kappa"], ascending=[True, False])
    keep, best = [], -np.inf
    for _, row in sdf.iterrows():
        if row["Kappa"] >= best:
            keep.append(row["index"])
            best = row["Kappa"]
    return df.index.isin(keep)

def plot_compliance_kappa(plot_df, save_prefix=None):
    df = plot_df[
        (plot_df["Metric"] == "Compliance") &
        (plot_df["Stat"].str.contains("Cohen", case=False))
    ].copy()
    df["ModelAlias"] = df["Model"].map(MODEL_ALIASES).fillna(df["Model"])
    df = df.dropna(subset=["ModelAlias"])
    alias_order = [MODEL_ALIASES.get(m, m) for m in models
                   if MODEL_ALIASES.get(m, m) in df["ModelAlias"].unique()]
    linebreak_map = {
        "Gemini 2.5 Flash": "Gemini 2.5\nFlash",
        "Gemini 2.5 Pro": "Gemini 2.5\nPro",
    }
    df["ModelAliasPlot"] = df["ModelAlias"].astype(str).map(lambda x: linebreak_map.get(x, x))
    alias_order_plot = [linebreak_map.get(a, a) for a in alias_order]
    df["ModelAliasPlot"] = pd.Categorical(df["ModelAliasPlot"],
                                          categories=alias_order_plot,
                                          ordered=True)
    df = df.sort_values("ModelAliasPlot")
    orange = sns.color_palette("OrRd", 9)[7]
    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(25, 8))
    sns.barplot(data=df, x="ModelAliasPlot", y="Value",
                color=orange, edgecolor="black", linewidth=1.3, ax=ax)
    ax.set_title("Agreement Between LLMs and Median Human Annotations",
                 fontweight="bold", fontsize=21)
    ax.set_xlabel("Models", fontweight="bold", fontsize=19)
    ax.set_ylabel("Cohen’s κ", fontweight="bold", fontsize=19)
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=15, labelsize=16, width=1.4) 
    for label in ax.get_xticklabels():
        label.set_fontweight("bold")
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")
    for p in ax.patches:
        h = p.get_height()
        ax.annotate(f"{h:.2f}",
                    (p.get_x() + p.get_width()/2., h),
                    ha="center", va="bottom",
                    fontsize=19, fontweight="bold",
                    xytext=(0, 4), textcoords="offset points")

    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout(pad=4.5)
    if save_prefix:
        plt.savefig(f"plots/{save_prefix}_compliance_kappa.png",
                    dpi=200, bbox_inches="tight", facecolor="white")
    plt.show()

def mae_heatmap(human_c_list, model_names, llm_pattern,
                mae_by_article_use,
                num_uses=NUM_USES,
                items_per_use=ITEMS_PER_USE,
                items_per_article=ITEMS_PER_ARTICLE,
                art_labels=ART_LABELS,
                title_prefix="Compliance MAE",
                save_prefix=None):
    maes = []
    for m in model_names:
        llm_path = Path(llm_pattern.format(model=m))
        sheet = pd.ExcelFile(llm_path).sheet_names[0]
        llm = extract_sheet_vectors(llm_path, sheet)
        maes.append(mae_by_article_use(
            human_c_list, llm["compliance"],
            num_uses=num_uses,
            items_per_use=items_per_use,
            items_per_article=items_per_article,
            art_labels=art_labels
        ))
    mae_df = maes[0] if len(maes) == 1 else sum(maes) / len(maes)
    plt.figure(figsize=(12, 6))
    ax = sns.heatmap(mae_df, annot=True, fmt=".2f", cmap="OrRd", cbar=False,
                     annot_kws={"weight": "bold", "size": 19})
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=19, fontweight="bold", rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=19, fontweight="bold", rotation=0)
    title_suffix = " (average)" if len(maes) > 1 else f"({MODEL_ALIASES.get(model_names[0], model_names[0])})"
    ax.set_title(f"{title_prefix}{title_suffix}", fontweight="bold", fontsize=21)
    plt.tight_layout(pad=3.5)
    if save_prefix:
        plt.savefig(f"plots/{save_prefix}_mae_heatmap.png",
                    dpi=200, bbox_inches="tight", facecolor="white")
    plt.show()

def confusion_matrix(human_c_list, model_names, llm_pattern, median,
                     title_prefix="Compliance Confusion", save_prefix=None):
    cats, cms = [1, 2, 3, 4, 5], []
    for m in model_names:
        llm_path = Path(llm_pattern.format(model=m))
        sheet = pd.ExcelFile(llm_path).sheet_names[0]
        llm = extract_sheet_vectors(llm_path, sheet)
        llm_int, med_int = median(human_c_list, llm["compliance"])
        cm = pd.crosstab(pd.Categorical(med_int, cats, True),
                         pd.Categorical(llm_int, cats, True),
                         dropna=False)
        cms.append(cm.values)
    cm_avg = cms[0] if len(cms) == 1 else np.nanmean(cms, axis=0)
    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(cm_avg, annot=True,
                     fmt="d" if len(cms) == 1 else ".1f",
                     cmap="OrRd", cbar=False,
                     linewidths=0.5, linecolor="white",
                     annot_kws={"weight": "bold", "size": 26})
    ax.set_xticklabels(cats, fontweight="bold", fontsize=24)
    ax.set_yticklabels(cats, fontweight="bold", fontsize=24, rotation=0)
    ax.set_xlabel("LLM rating", fontweight="bold", fontsize=24)
    ax.set_ylabel("Human median rating", fontweight="bold", fontsize=24)
    title_suffix = " (average)" if len(cms) > 1 else f" ({MODEL_ALIASES.get(model_names[0], model_names[0])})"
    ax.set_title(f"{title_prefix}{title_suffix}", fontweight="bold", fontsize=26)
    plt.tight_layout(pad=3.5)
    if save_prefix:
        plt.savefig(f"plots/{save_prefix}_confusion_matrix.png",
                    dpi=200, bbox_inches="tight", facecolor="white")
    plt.show()

def plot_pareto(df, title="Price vs Compliance Agreement", save_path=None):
    fig, ax = plt.subplots(figsize=(15, 11))
    pareto_mask = _pareto_mask(df)
    orange = sns.color_palette("OrRd", 6)[3]
    red = sns.color_palette("OrRd", 6)[5]
    ax.scatter(df.loc[~pareto_mask, "Price"], df.loc[~pareto_mask, "Kappa"],
               s=280, color=orange, edgecolor="black", linewidth=1, alpha=0.9)
    ax.scatter(df.loc[pareto_mask, "Price"], df.loc[pareto_mask, "Kappa"],
               s=280, color=red, edgecolor="black", linewidth=2, alpha=1.0)
    for _, r in df.iterrows():
        if r["ModelAlias"] == "Grok 3 mini":
            ax.annotate(r["ModelAlias"], (r["Price"], r["Kappa"]),
                        xytext=(2, 12), textcoords="offset points", 
                        fontsize=19, fontweight="bold")
        elif r["ModelAlias"] == "Gemini 2.5 Flash":
            ax.annotate(r["ModelAlias"], (r["Price"], r["Kappa"]),
                        xytext=(10, -2), textcoords="offset points", 
                        fontsize=19, fontweight="bold")
        elif r["ModelAlias"] == "GPT-5":
            ax.annotate(r["ModelAlias"], (r["Price"], r["Kappa"]),
                        xytext=(10, -6), textcoords="offset points", 
                        fontsize=19, fontweight="bold")
        else:
            ax.annotate(r["ModelAlias"], (r["Price"], r["Kappa"]),
                        xytext=(5, 5), textcoords="offset points", 
                        fontsize=19, fontweight="bold")
    ax.set_xlabel("Cost per M output tokens (USD$)", fontweight="bold", fontsize=19)
    ax.set_ylabel("Cohen’s κ", fontweight="bold", fontsize=19)
    ax.set_title(title, fontweight="bold", fontsize=21)
    ax.tick_params(axis="both", labelsize=19, width=1.4)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.grid(True, alpha=0.3)
    xmin, xmax = df["Price"].min(), df["Price"].max()
    ax.set_xlim(xmin - (xmax - xmin) * 0.05 , xmax + (xmax - xmin) * 0.25)
    ymin, ymax = df["Kappa"].min(), df["Kappa"].max()
    ymargin = (ymax - ymin) * 0.2
    ax.set_ylim(ymin - ymargin, ymax + ymargin)
    plt.tight_layout(pad=3.5)
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.show()

# ---------- Main ------------------------------------------------------------------
def main():
    # --- Prints ---
    Path("plots").mkdir(exist_ok=True)
    human_c_disagg, human_p_disagg = load_humans_with_nans(human_file_disaggregated)
    print("\n=== Human inter-rater reliability (Krippendorff’s α) ===")
    print(f"Compliance:   α = {alpha(human_c_disagg):.3f}")
    print(f"Plausibility: α = {alpha(human_p_disagg):.3f}")
    human_c_list, human_p_list = load_humans(human_file)
    all_rows, all_details, normal_details = [], [], []
    for m in models:
        df, detail = evaluate_model(human_c_list, human_p_list, m)
        all_rows.append(df)
        all_details.append(detail)
        normal_details.append(detail)
    for suffix in ["empty", "harsh", "articleless"]:
        df, detail = evaluate_model(human_c_list, human_p_list, "4o", ablation_suffix=suffix)
        all_rows.append(df)
        all_details.append(detail)
    plaus_stats = {}
    for _, res in normal_details:
        for k, v in res.items():
            if k.startswith("Plausibility_"):
                plaus_stats.setdefault(k, []).append(v)
    print("\n=== Average Plausibility Stats (across models) ===")
    for stat, values in plaus_stats.items():
        arr = np.array(values, dtype=float)
        if np.isnan(arr).any():
            print(f"{stat:30s}: {np.nanmean(arr):.3f} (NaNs ignored)")
        else:
            print(f"{stat:30s}: {arr.mean():.3f}")
    print("\n=== Human compliance score counts by use case ===")
    score_counts = score_counts_by_usecase(human_file)
    print(score_counts)
    mae_avg_df = get_avg_mae_df(human_c_list, models, llm_pattern)
    print_mae_rowcol_avgs(mae_avg_df, title="Compliance MAE")
    summarise_human_plausibility(human_p_disagg)
    
    # --- Plots ---    
    plot_df = pd.concat(all_rows, ignore_index=True)
    df = plot_df[(plot_df["Metric"] == "Compliance") &
                (plot_df["Stat"].str.contains("Cohen", case=False))].copy()
    plot_compliance_kappa(plot_df, save_prefix="llm_vs_humans")
    df["ModelAlias"] = df["Model"].map(MODEL_ALIASES).fillna(df["Model"])
    df["Price"] = df["Model"].map(PRICES)
    df = df.dropna(subset=["Price", "Value"]).rename(columns={"Value": "Kappa"})
    plot_pareto(df, save_path="plots/llm_price_vs_kappa.png")
    mae_heatmap(human_c_list, models, llm_pattern, mae_by_article_use,
            save_prefix="all_models")
    confusion_matrix(human_c_list, models, llm_pattern, median,
                    save_prefix="all_models")
    mae_heatmap(human_c_list, ["pro"], llm_pattern, mae_by_article_use,
                save_prefix="gemini_pro")
    confusion_matrix(human_c_list, ["pro"], llm_pattern, median,
                    save_prefix="gemini_pro")
    return plot_df, all_details

if __name__ == "__main__":
    main()