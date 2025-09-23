"""
This script takes excerpts of technical documentation from "documentation". Each excerpt relates to an aspect of the EU AI Act from "articles.txt".
The script then prompts one or more LLMs to score the excerpts for compliance (with repsect to the selected articles) and plausibility with brief explanations.
Finally, it parses the responses and saves them into "llm_annotations".
Notably, before sending the responses to "llm_annotations", it permutes them in the same manner that annotator assignments were permuted to prevent bias. 
"""
import os
import re
from pathlib import Path
import itertools
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai
load_dotenv()

# ---------- Parameters ----------------------------------------------
models        = ["4o"] # Options: ["gpt5","o3","o3_mini","4o","sonnet","pro","flash","gemma","grok3_mini","grok4"]
articles      = [9, 10, 12, 14, 15] # Current setting: Include all articles of interest from articles.txt
intended_uses = [1, 2, 3, 4, 5, 6, 7, 8] # Current setting: Include the 8 use cases from intended_uses.txt that annotators graded
PROMPT_MODIFIER = ("Your scores for both compliance and plausibility should be well-calibrated and objective. They should be rigorous but fair.")
ABLATION = 0 # Options: 0 (no ablation: standard prompt modifier), 1 (empty prompt modifier), 2 (harsh prompt modifier), 3 (remove LLMs' access to text from the EU AI Act)

# ---------- Client initialisation ----------------------------------------------
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
grok_client = OpenAI(
    api_key=os.getenv("XAI_API_KEY"),
    base_url="https://api.x.ai/v1"
)

# ---------- LLM wrappers ----------------------------------------------
def chat_with_gpt_5(question: str) -> str:
    response = openai_client.chat.completions.create(
        model="gpt-5-2025-08-07",
        messages=[{"role": "user", "content": question}],
    )
    return response.choices[0].message.content.strip()

def chat_with_o3(question: str) -> str:
    response = openai_client.chat.completions.create(
        model="o3-2025-04-16",
        messages=[{"role": "user", "content": question}],
    )
    return response.choices[0].message.content.strip()

def chat_with_o3_mini(question: str) -> str:
    response = openai_client.chat.completions.create(
        model="o3-mini-2025-01-31",
        messages=[{"role": "user", "content": question}],
    )
    return response.choices[0].message.content.strip()

def chat_with_4o(question: str) -> str:
    response = openai_client.chat.completions.create(
        model="gpt-4o-2024-11-20",
        messages=[{"role": "user", "content": question}],
    )
    return response.choices[0].message.content.strip()

def chat_with_sonnet(question: str) -> str:
    response = anthropic_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": question}],
    )
    return response.content[0].text.strip()

def chat_with_pro(question: str) -> str:
    model = genai.GenerativeModel("gemini-2.5-pro")
    response = model.generate_content(question)
    return response.text.strip()

def chat_with_flash(question: str) -> str:
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(question)
    return response.text.strip()

def chat_with_gemma(question: str) -> str:
    model = genai.GenerativeModel("gemma-3-27b-it")
    response = model.generate_content(question)
    return response.text.strip()

def chat_with_grok3_mini(question: str) -> str:
    response = grok_client.chat.completions.create(
        model="grok-3-mini",
        messages=[{"role": "user", "content": question}],
    )
    return response.choices[0].message.content.strip()

def chat_with_grok4(question: str) -> str:
    response = grok_client.chat.completions.create(
        model="grok-4-0709",
        messages=[{"role": "user", "content": question}],
    )
    return response.choices[0].message.content.strip()

def chat_with_llm(question: str, model: str) -> str:
    if model == "gpt5":
        return chat_with_gpt_5(question)
    elif model == "o3":
        return chat_with_o3(question)
    elif model == "o3_mini":
        return chat_with_o3_mini(question)
    elif model == "4o":
        return chat_with_4o(question)
    elif model == "sonnet":
        return chat_with_sonnet(question)
    elif model == "pro":
        return chat_with_pro(question)
    elif model == "flash":
        return chat_with_flash(question)
    elif model == "gemma":
        return chat_with_gemma(question)
    elif model == "grok3_mini":
        return chat_with_grok3_mini(question)
    elif model == "grok4":
        return chat_with_grok4(question)
    else:
        raise ValueError("Model not in list.")

# ---------- Load and parse ----------------------------------------------
def extract_article(path: Path, art: int) -> str | None:
    txt = path.read_text(encoding="utf-8")
    pat = rf"(^\s*\*\*Article\s+{art}\b[\s\S]+?)(?=^\s*\*\*Article\s+\d+\b|\Z)"
    m = re.search(pat, txt, flags=re.MULTILINE)
    return m.group(1).strip() if m else None

def load_system_outline(use_num: int, system_num: int) -> str:
    path = Path("system_outlines") / f"Use{use_num}" / f"System{system_num}.txt"
    return path.read_text(encoding="utf-8")

def load_documentation(art: int, use_num: int, system_num: int, scenario_num) -> str:
    base = Path("documentation") / f"Article{art}" / f"Use{use_num}" / f"System{system_num}"
    if isinstance(scenario_num, int):
        path = base / f"Violation_{scenario_num}.txt"
    elif scenario_num == "compliant":
        path = base / "Compliance.txt"
    return path.read_text(encoding="utf-8")

def _clean_expl(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'^[^A-Za-z]+', '', text)
    m = re.search(r"\.(?=[^A-Za-z]*$)", text)
    if m:
        text = text[:m.end()]
    return text.strip()

def parse_llm_answer(ans: str) -> dict:
    out = {"Compliance [1-5]": None, "Compliance [Text]": "",
           "Plausibility [1-5]": None, "Plausibility [Text]": ""}
    mc = re.search(r"Compliance:\s*\[?([1-5])\]?", ans, flags=re.I)
    mp = re.search(r"Plausibility:\s*\[?([1-5])\]?", ans, flags=re.I)
    if mc:
        out["Compliance [1-5]"] = int(mc.group(1))
    if mp:
        out["Plausibility [1-5]"] = int(mp.group(1))
    mce = re.search(r"(?is)Compliance\s*:\s*.*?Explanation\s*:\s*(.+?)(?=(?:\s|\*)*Plausibility\s*:|$)",ans)
    mpe = re.search(r"(?is)Plausibility\s*:\s*.*?Explanation\s*:\s*(.+)",ans)
    if mce:
        out["Compliance [Text]"] = _clean_expl(mce.group(1))
    if mpe:
         out["Plausibility [Text]"] = _clean_expl(mpe.group(1))
    return out

# ---------- Prompt for LLM ----------------------------------------------
def build_prompt(article_text: str, documentation: str, system_outline: str, art: int, ablation: int) -> str:
    base_prompt = f"""
---
Format your final response EXACTLY as follows (do not include anything else):

Compliance: [Your score here from 1 to 5]
Explanation: [Your explanation here]

Plausibility: [Your score here from 1 to 5]
Explanation: [Your explanation here]

Ensure your explanations flow naturally from the required starting point.

---

For each excerpt, select a score from 1 to 5 in response to each of the following statements, and provide two explanations: one for each score.

Compliance: “This excerpt depicts compliance with the relevant Article of the European Union’s AI Act.”
Please provide a score from 1 to 5, using the following grading criteria. Then, include an explanation (100-750 characters) justifying why your score is neither higher nor lower.
[1] Very low probability of compliance
[2] Low probability of compliance
[3] Moderate probability of compliance
[4] High probability of compliance
[5] Very high probability of compliance

Plausibility: “This excerpt resembles credible and logically consistent technical documentation.”
Please provide a score from 1 to 5, using the following grading criteria. Then, include an explanation justifying why your score is neither higher nor lower.
[1] Very low probability of plausibility
[2] Low probability of plausibility
[3] Moderate probability of plausibility
[4] High probability of plausibility
[5] Very high probability of plausibility

Please start each of your explanations as follows:
“I believe there is a (insert the appropriate non-numerical quantifier for your score, given above) probability the AI system complies with Article {art}...”
“I believe there is a (insert the appropriate non-numerical quantifier for your score, given above) probability the excerpt is plausible…”

When evaluating for compliance, base your score strictly on the cited Article, not general AI governance principles. Ignore any self-assessment in the documentation. For plausibility, judge whether the excerpt is credible, logically consistent, and professional.

*****System Outline*****
{system_outline}

*****Extract of Article*****
{article_text}
""".strip()

    if ablation == 0:
        return base_prompt + f"\n\n*****Excerpt of Documentation*****\n{documentation} \n\n {PROMPT_MODIFIER}"
    elif ablation == 1:
        return base_prompt + f"\n\n*****Excerpt of Documentation*****\n{documentation}"
    elif ablation == 2:
        return base_prompt + f"\n\n*****Excerpt of Documentation*****\n{documentation} \n\n Your scores for both compliance and plausibility should be critical. They should be harsh but fair."
    elif ablation == 3:
        m = re.search(r"\d+", article_text)
        return base_prompt + f"\n\n*****Article***** Focus on Artcile\n {int(m.group())} \n\n {PROMPT_MODIFIER}"

# ---------- Permutation ----------------------------------------------
def load_rand_assign(path: Path) -> dict:
    """
    Load the permutation that jumbled annotators' assignments to prevent bias:

    (use, article, scenario_num) -> scenario_label (A/B/C)

    Where scenario_num is:
        1 = Compliance
        2 = Violation_1
        3 = Violation_2
    """
    mapping = {}
    use = art = None
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("Use"):
                m = re.match(r"Use\s+(\d+),\s*Article\s+(\d+)", line)
                if m:
                    use, art = map(int, m.groups())
            else:
                m = re.match(r"(\d+)\s*--\s*([ABC])", line)
                if m:
                    scen_num = int(m.group(1)) 
                    scen_label = f"Scenario {m.group(2)}"
                    mapping[(use, art, scen_num)] = scen_label
    return mapping

# ---------- Build sheet -----------------------------------------------------------
SUBHEAD = [
    "Compliance [1-5]",
    "Compliance [Text]",
    "Plausibility [1-5]",
    "Plausibility [Text]",
]

def empty_block():
    return {h: None if "1-5" in h else "" for h in SUBHEAD}

def make_sheet_df(articles, intended_uses, results_for_model):
    row0 = [""] + list(itertools.chain.from_iterable([[f"Use {u}"] * len(SUBHEAD) for u in intended_uses]))
    row1 = [""] + list(itertools.chain.from_iterable([SUBHEAD for _ in intended_uses]))

    data_rows = []
    for art in articles:
        for scen in ["Scenario A", "Scenario B", "Scenario C"]:
            label = f"Art {art} / {scen}"
            row = [label]
            for u in intended_uses:
                block = results_for_model.get((art, scen, u), empty_block())
                row.extend([block[h] for h in SUBHEAD])
            data_rows.append(row)

    df = pd.DataFrame([row0, row1] + data_rows)
    df.columns = [" "] + [f"c{i}" for i in range(1, df.shape[1])]
    return df

# ---------- Main routine ----------------------------------------------
if __name__ == "__main__":
    articles_txt = Path("articles.txt")
    article_texts = {art: extract_article(articles_txt, art) for art in articles}
    rand_assign = load_rand_assign(Path("rand_assign.txt"))

    for model in models:
        fails = 0
        answer = None
        try:
            out_path = Path(f"llm_annotations/{model}_annotations_new.xlsx")
            results = {}
            for art in articles:
                art_text = article_texts.get(art) or ""
                for use_num in intended_uses:
                    # This switches at Use 5 (as human annotation switched here too for logistical reasons)
                    sys_assign = 1 if use_num <= 5 else 2 
                    sys_outline = load_system_outline(use_num, sys_assign)
                    for sn in [1, 2, 3]:
                        # This is the main routine -- prompting the LLM and parsing its response
                        scen_label = rand_assign[(use_num, art, sn)]
                        doc = load_documentation(art, use_num, sys_assign, "compliant" if sn == 1 else sn - 1)
                        prompt = build_prompt(art_text, doc, sys_outline, art, ABLATION)
                        answer = chat_with_llm(prompt, model)
                        block = empty_block() | parse_llm_answer(answer)

                        # This checks whether the model can format its reponses correctly
                        if (block["Compliance [1-5]"] is None or
                            block["Plausibility [1-5]"] is None):
                            fails += 1
                            print(f"{model} did not properly format format for Art {art}, Use {use_num}, {scen_label} (fails={fails})")
                            if fails >= 5:
                                raise RuntimeError(f"{model} failed too many times to provide a correct response")
                            continue 

                        results[(art, scen_label, use_num)] = block

                print(f"{model} has compledted Article {art}")

            df = make_sheet_df(articles, intended_uses, results)
            with pd.ExcelWriter(out_path, engine="openpyxl") as xlw:
                df.to_excel(xlw, sheet_name=model[:31], index=False, header=False)
            print(f"Completed annotations by model {model}")

        except Exception as e:
            last = answer if answer is not None else "<no answer captured>"
            print(f"Skipping model {model} due to error: {e}\n\nLatest response by {model} was {last}")
            continue