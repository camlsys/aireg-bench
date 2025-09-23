"""
Step 3 of generating technical documentation excerpts.
This script takes EU AI Act article extracts, AI system outlines, and compliance violation scenarios from "articles.txt", "../system_outlines", and "../violations".
It then prompts gpt-4.1-mini to generate technical documentation.
Finally, it saves the responses into organised text files in "../documentation".
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
scenario_nums = [1, 2] # Current setting: Include 2 of the 3 violation scenarios from "../violations"

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
def extract_article(path: Path, art: int) -> str | None:
    txt = path.read_text(encoding="utf-8")
    pat = rf"(^\s*\*\*Article\s+{art}\b[\s\S]+?)(?=^\s*\*\*Article\s+\d+\b|\Z)"
    match = re.search(pat, txt, flags=re.MULTILINE)
    return match.group(1).strip() if match else None

def load_outline(use_num: int, system_num: int) -> str:
    outline_path = Path(f"system_outlines/Use{use_num}/System{system_num}.txt")
    return outline_path.read_text(encoding="utf-8")

def load_violation(art: int, use_num: int, system_num: int, scenario_num: int) -> str:
    violation_path = (
        Path("violations")
        / f"Article{art}"
        / f"Use{use_num}"
        / f"System{system_num}"
        / f"Scenario{scenario_num}.txt"
    )
    return violation_path.read_text(encoding="utf-8")

def extract_violation(txt: str) -> str | None:
    pattern = re.compile(
        r"\*\*Violation:\*\*\s*"                   
        r"(.*?)"                                  
        r"(?=\s*\[\w+\]\s*\*\*Justification:\*\*|\Z)",  
        flags=re.DOTALL | re.IGNORECASE
    )
    match = pattern.search(txt)
    return match.group(1).strip() if match else None

# ---------- Prompt for LLM ----------------------------------------------
def build_prompt(article_text: str, system_outline: str, compliance_profile: str) -> str:
    return f"""
    *****Task*****
    Please write the documentation so that it aligns with the compliance profile (but show it, don't tell it).
    
    *****Compliance profile*****
    {compliance_profile}

    *****Details*****
    You are acting as a compliance expert and machine-learning specialist. You are tasked with contributing to the writing of technical documentation for an AI system. 
    You represent the provider of an AI system. You do not represent the system deployers and are not aware of their identities, though you need not explicitly reference this in your response.
    To inform your response, you will be given an extract from the EU AI Act, a short description of an AI system, and a compliance profile. 
    If the extract references provisions outside of the extract itself (whether from elsewhere in the EU AI Act or from external legislation) interpret them using the context in which they are referenced and your prior knowledge of the EU AI Act. 
    You will produce a section of technical documentation (intended to inform a compliance assessment against the provided extract of the EU AI Act) for the specified AI system and its compliance profile.

    Substance - You must (unless told otherwise by the compliance profile):
    - Present all necessary provider decisions (with associated evidence and rationale) to facilitate a sober and detail-oriented compliance assessment. 
    - Discuss realistic system components and modalities representative of those typically used in 2025, reflecting current industry standards. 
    - Discuss realistic compliance measures representative of those typically used in 2025, reflecting current industry standards. 
    - Ensure the documentation is consistent with the provided system description and compliance profile. 
    - Ensure the documentation is internally consistent (e.g., system attributes and compliance measures fit together without contradiction, describing a coherent and realistic set of technical and operational facts).
    - Ensure the system description contains a substantive, rather than a cursory, set of facts. To do so, you may need to fictionalise evidence, research, findings, and details to support your claims. 
    - Ensure any fictionalised numerical details and supporting evidence (e.g., dataset size, performance, benchmarks, adversarial testing, data processing) are realistic.
    - Ensure that any quoted numbers are consistent with each other and can be plausibly combined (e.g., a model trained on a large number of samples would require a large amount of compute). 
    - Address only the provided extract of the EU AI Act; do not address other articles or related regulations.
    - Ensure the system is not prohibited by Article 5 of the EU AI Act and is also not biometric. 

    Formatting - You must (unless told otherwise by the compliance profile):
    - Begin your response with: **Article X**, where X is the number of the article given in the extract.
    - Create subtitles for the different parts of your response that are appropriate for legal prose; avoid just repeating the provisions from the extract as subtitles.
    - Tailor paragraph length and detail; each bullet should be addressed fully, typically in 150-300 words.

    Style - You must (unless told otherwise by the compliance profile):
    - Produce a professional and realistic simulation of the structured prose an auditor may receive. 
    - Use technical but accessible language, briefly clarifying domain-specific terminology.
    - Soberly and concretely present technical and operational facts, focusing on "showing, not telling."
    - Both state what was done and why it was done that way.
    - Be granular and precise without being excessively elaborate.

    Negatives - You must (unless told otherwise by the compliance profile):
    - Not weigh in on the legal interpretation of the facts, such as asserting compliance or a lack thereof. 
    - Not include unrequested introductions, conclusions, or section summaries (i.e., your prose should start and end where the section naturally starts and ends). 
    - Not disclose the fictional nature of any evidence or findings in your response. 
    - Not reference these instructions or the system compliance profile in your response.
        
    You must (unless told otherwise by the compliance profile):
    - Return nothing else outside the requested documentation, formatted in the manner outlined above.

    *****Compliance profile*****
    {compliance_profile}

    *****Extract*****
    {article_text}
        
    *****System*****
    {system_outline}
    """

compliant_profile = "The AI system complies with every aspect of the EU AI Act." 

# ---------- Main routine ----------------------------------------------
if __name__ == "__main__":
    articles_txt = PROJECT_ROOT / "articles.txt"
    output_dir   = PROJECT_ROOT / "documentation_new"
    for art in articles:                                            
        article_text = extract_article(articles_txt, art)           
        for use_num in intended_uses: 
            for system_num in system_nums:                          
                system_outline = load_outline(use_num, system_num)
                for scenario_num in scenario_nums:
                    violation_example = load_violation(art, use_num, system_num, scenario_num) 
                    violation_example = extract_violation(violation_example)
                    prompt = build_prompt(article_text, system_outline, violation_example)
                    answer = chat_with_gpt(prompt)
                    out_path = (
                        output_dir
                        / f"Article{art}"
                        / f"Use{use_num}"
                        / f"System{system_num}"
                    )
                    out_path.mkdir(parents=True, exist_ok=True)
                    file_path = out_path / f"Violation_{scenario_num}.txt"
                    file_path.write_text(answer, encoding="utf-8")
                    print(f"Generated documentation for Article {art}, Use {use_num}, System {system_num}, Violation {scenario_num}")
                
                prompt = build_prompt(article_text, system_outline, compliant_profile)
                answer = chat_with_gpt(prompt)
                out_path = (
                    output_dir
                    / f"Article{art}"
                    / f"Use{use_num}"
                    / f"System{system_num}"
                )
                out_path.mkdir(parents=True, exist_ok=True)
                file_path = out_path / f"Compliance.txt"
                file_path.write_text(answer, encoding="utf-8")
                print(f"Generated documentation for Article {art}, Use {use_num}, System {system_num}, Compliance")
                
