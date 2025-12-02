from pathlib import Path
from typing import List, Tuple
import os
import re
import pandas as pd
import numpy as np
import google.generativeai as genai

# ---------- Client initialisation ----------------------------------------------
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
COMPLIANCE_TEXT_MARK = "Compliance [Text]"
ARTICLES = ["9", "10", "12", "14", "15"]

# ---------- LLM wrapper ----------------------------------------------
def chat_with_pro(prompt: str) -> str:
    model = genai.GenerativeModel("gemini-2.5-pro")
    response = model.generate_content(prompt)
    return response.text.strip()

# ---------- Excel helpers ----------------------------------------------
def _find_marker_positions(df: pd.DataFrame, marker: str) -> List[Tuple[int, int]]:
    mask = df.to_numpy() == marker
    rows, cols = np.where(mask)
    return list(zip(rows, cols))

def load_annotations_grouped_by_article(path: Path):
    grouped = {art: [] for art in ARTICLES}
    xls = pd.ExcelFile(path)
    for sheet in xls.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet, header=None)
        positions = _find_marker_positions(df, COMPLIANCE_TEXT_MARK)
        for tr, tc in positions:
            for r in range(tr + 1, df.shape[0]):
                text_val = df.iloc[r, tc]
                if not isinstance(text_val, str) or not text_val.strip():
                    continue
                art_cell = df.iloc[r, 0]
                if not isinstance(art_cell, str):
                    continue
                m = re.search(r"Art\s*(\d+)", art_cell)
                if not m:
                    continue
                art_num = m.group(1)
                if art_num not in grouped:
                    grouped[art_num] = []
                grouped[art_num].append(text_val.strip())
    return grouped

# ---------- Prompt ----------------------------------------------
def build_article_summary_prompt(article: str, text: str) -> str:
    return f"""
Write one clear sentence summarising the main documentation issues, weaknesses, or recurring problems apparent.

DATA (all reviewer compliance comments for this article, across all uses and iterations):
{text}
""".strip()

# ---------- Main ----------------------------------------------
def summarise_all():
    excel_path = Path("human_annotations.xlsx")
    grouped = load_annotations_grouped_by_article(excel_path)
    for art in ARTICLES:
        texts = grouped.get(art, [])
        print(f"\n=== Article {art} ===")
        if not texts:
            print("No items found, skipping.")
            continue
        full_text = "\n\n".join(texts)
        prompt = build_article_summary_prompt(art, full_text)
        summary_text = chat_with_pro(prompt)
        print(summary_text)

if __name__ == "__main__":
    summarise_all()
