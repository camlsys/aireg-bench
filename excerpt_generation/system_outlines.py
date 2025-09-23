"""
Step 1 of generating technical documentation excerpts.
The script reads a list of intended AI use cases from "intended_uses.txt", with each use case based on the wording of the EU AI Act.
It then prompts gpt-4.1-mini to generate four structured AI system descriptions for each use case. 
Finally, it parses the responses and saves them into organised text files in "../system_outlines".
"""
import os
import re
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------- Client initialisation ----------------------------------------------
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------- LLM wrappers ----------------------------------------------
def chat_with_gpt(question: str) -> str:
    resp = openai_client.chat.completions.create(
        model="gpt-4.1-mini-2025-04-14",
        messages=[{"role": "user", "content": question}],
    )
    return resp.choices[0].message.content.strip()

# ---------- Load, parse, and save ----------------------------------------------
def load_intended_uses(path: str = "intended_uses.txt") -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]

def parse_system_descriptions(text: str) -> list[str]:
    pattern = re.compile(
        r"System\s+\d+\.\s*"                  
        r"(.*?)"                              
        r"(?="                                
          r"\s*(?:\d+\.\s*System\s+\d+\.|"    
          r"System\s+\d+\.|"                  
          r"\Z"                               
        r"))",
        flags=re.DOTALL
    )
    blocks = []
    for raw in pattern.findall(text):
        para = " ".join(raw.strip().split())
        blocks.append(para)
    return blocks

def save_descriptions(prose_blocks: list[str], base_dir: Path, use_num: int) -> None:
    iu_folder = base_dir / f"Use{use_num}"
    iu_folder.mkdir(parents=True, exist_ok=True)
    if prose_blocks:
        for i, block in enumerate(prose_blocks, start=1):
            example_path = iu_folder / f"System{i}.txt"
            with open(example_path, "w", encoding="utf-8") as fout:
                fout.write(block)
    else:
        print("Error in parsing")

# ---------- Prompt for LLM ----------------------------------------------
def build_prompt(intended_use: str) -> str:
    return f"""
    Your task is to generate four distinct AI system descriptions for the provided intended use.
    Each AI system must employ only one or two domain-appropriate types of machine learning models or algorithms. You should pick the algorithm you feel is most appropriate for the use case in the contemporary era, but here are some examples of the types of algorithms that you might choose: MLP, CNN, Transformers (encoder-only, decoder-only, or encoder-decoder), SVM, RNN, Naive Bayes, GNN, Random Forest, KNN, GBDT, Linear Regression, transformer-based Large Language Model (LLM), transformer-based Vision Language Model (VLM), diffusion-based text-to-image generation model, or similar. 
    Transformers can be used in distinct ways, including for processing different data types such as tabular data, text, audio, API calls, and more.  
    Your choices of models or algorithms should reflect those likely to be deployed in 2025. You should focus on realism given the particular application as well as domain-appropriateness. 
    Systems must not employ biometric technologies or violate Article 5 of the EU AI Act.
    
    Start each description with: "System 1." "System 2." "System 3." "System 4."
    Each description should be a single continuous paragraph. 
    There should be an empty line separating each system description from the next.

    Intended use: {intended_use}

    For each system, provide a concise description consisting of the following four components, each in a single sentence:
    [a] System Name and Type: State the AI system's name and the machine learning models or algorithms it relies on. When naming these AI systems, ensure the names are diverse, realistic, and professional (while also fictional). System names should be formed by using multiple, separate words; never form system names by concatenating words (e.g., "EducationSmart") or using portmanteaus (e.g.,"EduBoost").
    [b] System use: Provide the system's intended usage and the specific sector or area it will serve.
    [c] Objective and Technological Capabilities: Describe the system's primary objective and outline the technological capabilities that enable the achievement of this objective.
    [d] Provider: Identify the natural or legal person, public authority, agency or other body that develops the system or that has the system developed and places it on the market or puts the system into service under its own name or trademark, whether for payment or free of charge. Where providers are companies, ensure their names are realistic and professional (while fictional); please do not simply concatenate words (e.g., "EducationSmart") or use portmanteaus (e.g.,"EduBoost")
    [e] The AI subject: Define the persons (or groups of persons) who are directly affected by the AI system, experiencing its outcomes and consequences.

    Return nothing else outside those four system descriptions, formatted in the manner outlined above. 
    """

# ---------- Main routine ----------------------------------------------
def system_variance(intended_use: str, use_num: int, base_dir: Path) -> None:
    prompt   = build_prompt(intended_use)
    response = chat_with_gpt(prompt)
    blocks   = parse_system_descriptions(response)
    save_descriptions(blocks, base_dir, use_num)

if __name__ == "__main__":
    base_folder = PROJECT_ROOT / "system_outlines_new"
    base_folder.mkdir(parents=True, exist_ok=True)
    intended_uses = load_intended_uses()
    for use_num, intended_use in enumerate(intended_uses, start=1):
        system_variance(intended_use, use_num, base_folder)