"""
Step 2 of generating technical documentation excerpts.
The script extracts EU AI Act articles from "articles.txt" and pairs them with AI system outlines from "../system_outlines".
It then prompts gpt-4.1-mini to produce three structured examples of potential compliance violations.
Finally, it parses the compliance violation scenarios and saves them into organised text files in "../violations".
"""
import os
import re
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI       
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------- Parameters ----------------------------------------------
articles      = [9, 10, 12, 14, 15] # Current setting: Include all articles of interest from articles.txt
system_nums   = [1, 2] # Current setting: Include 2 of the 4 system outlines for each use case from "../system_outlines"
intended_uses = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] # Current setting: Include all 10 use cases from intended_uses.txt

# ---------- Client initialisation ----------------------------------------------
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------- LLM wrappers ----------------------------------------------
def chat_with_gpt(question: str) -> str:
    resp = openai_client.chat.completions.create(
        model="gpt-4.1-mini-2025-04-14",
        messages=[{"role": "user", "content": question}],
    )
    return resp.choices[0].message.content.strip()

# ---------- Load and parse ----------------------------------------------
def extract_article(path: str, art: int) -> str | None:
    txt = open(path, encoding="utf-8").read()
    pat = rf"(^\s*\*\*Article\s+{art}\b[\s\S]+?)(?=^\s*\*\*Article\s+\d+\b|\Z)"
    match = re.search(pat, txt, flags=re.MULTILINE)
    return match.group(1).strip() if match else None

def load_outline(use_num: int, system_num: int) -> str:
    outline_path = Path(f"system_outlines/gpt/Use{use_num}/System{system_num}.txt")
    return outline_path.read_text(encoding="utf-8")

def parse_scenarios(combined_text: str) -> list[str]:
    matches = list(re.finditer(r"^Example\s+[1-3]\.\s*", combined_text, flags=re.MULTILINE))
    scenarios = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(combined_text)
        block = combined_text[start:end].strip()
        scenarios.append(block)
    return scenarios

# ---------- Prompt for LLM ----------------------------------------------
def build_prompt(article_text: str, system_outline: str) -> str:
    return f"""
    *****Task*****
    You are acting as an EU AI Act compliance expert and machine-learning specialist.  
    You will be given an extract from the EU AI Act and a short description of an AI system.
    Your task is to generate three distinct ways that the AI system could violate the provided extract.

    You must not produce:
    - violations that reflect errors in the technical documentation rather than the system (e.g., lies or missing details).
    - violations without any justification, evidence, or granular details (e.g., "the model is biased" without elaboration).

    You must produce:
    - violations that are plausible and realistic
    - violations that are subtle and not blatant

    Start each example with: "Example 1.", "Example 2.", "Example 3."
    Each description should consist of four parts [a]-[d], each given 1-3 sentences. 
    There should be an empty line separating each part from the next and each example from the next. 

    Example X:  
    [a] **Quotation:**  
    <verbatim quote from the EU AI Act extract that will be violated>

    [b] **Guideline:**  
    <granular and realistic standards that experts would use to ensure compliance with the quotation>

    [c] **Violation:**  
    <precise account of a violation of guideline [b] that would imply non-compliance with extract [a]>

    [d] **Justification:**  
    <an explanation for why violation [c] breaches the quoted requirement and why it is realistic yet subtle>

    Return nothing else outside those three examples, formatted in the manner outlined above.

    *****Extract*****
    {article_text}

    *****System*****
    {system_outline}
    """

# ---------- Main routine ----------------------------------------------
if __name__ == "__main__":
    for art in articles:
        article_text = extract_article("articles.txt", art)
        for use_num in intended_uses:
            for system_num in system_nums:
                try:
                    outline_text = load_outline(use_num, system_num)
                except FileNotFoundError as e:
                    print("No system outline")
                    continue
                prompt = build_prompt(article_text, outline_text)
                combined_output = chat_with_gpt(prompt)
                scenario_texts = parse_scenarios(combined_output)
                for scenario_idx, scenario_content in enumerate(scenario_texts, start=1):
                    out_path = (
                        PROJECT_ROOT
                        / "violations_new"
                        / f"Article{art}"
                        / f"Use{use_num}"
                        / f"System{system_num}"
                        / f"Scenario{scenario_idx}.txt"
                    )
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_text(scenario_content, encoding="utf-8")