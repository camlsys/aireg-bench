## Background and Overview

Paper: [AIReg-Bench: Benchmarking Language Models That Assess AI Regulation Compliance](https://arxiv.org/abs/2510.01474) 

As governments around the world move to regulate AI, there is growing interest in using Large Language Models (LLMs) to assess whether or not an AI system complies with a given AI regulation (AIR). However, there is presently no way to benchmark the performance of LLMs at this task. 

To start to fill this void, we introduce the AIReg-Bench benchmark: a legal expert-annotated dataset designed to test how well LLMs can assess compliance with the EU AI Act (AIA). 

We created this dataset through a two-step process: 

1. By prompting an LLM with carefully-structured instructions, we generated 120 technical documentation excerpts, each depicting a fictional, albeit plausible, AI system. These reflect the type of technical documentation that an AI provider might produce to demonstrate their compliance with an AI regulation.  

Importantly, all of the AI systems represented in our dataset are high-risk under the AIA (Article 6; Annex III) that fall into neither the AIA's exemptions nor its prohibitions. Further, they are drafted from the perspective of an AI provider. 

2. Legal experts then reviewed and labelled each excerpt to indicate whether, and in what way, the AI system described therein violates specific Articles of the AIA. 

We hope the resulting dataset provides a starting point to understand the opportunities and limitations of LLM-based AIR compliance assessment tools and establishes a benchmark against which the performance of subsequent LLMs at this task can be compared.

In addition to the AIReg-Bench benchmark dataset (see AIReg-Bench Dataset below), this repository contains: 

- The code we used to prompt an LLM to generate the technical documentation excerpts in the dataset (see Dataset Generation below)
- The code we used to evaluate a set of SOTA LLMs' performance on the benchmark (see Evaluation and LLM Annotations below)
- Other utilities (see Other Files and Folders below)

## Files
The dataset, LLM annotations, and evaluation sections are the most relevant for using the benchmark in practice.

### AIReg-Bench Dataset
- **documentation | folder** - This contains 300 .txt files. Each represents a unique excerpt of (AI-generated) AI system technical documentation. This documentation is organised within the folder in accordance with [1] the articles of the EU AI Act it pertains to (e.g., Article 9), [2] the intended use of the AI system (e.g., recruitment), [3] the details of the AI system being used (e.g., transformer for summarising text), and [4] the compliance profile of the system (e.g., neglecting to consider bias).
- **human_annotation.xlsx** - This contains three human-graded annotations for 120 of the excerpts of technical documentation, including scores for compliance and plausibility with brief explanations.
- **llm_annotations | folder** - This contains LLM-graded annotations for 120 of the excerpts of technical documentation, including scores for compliance and plausibility with brief explanations.

### LLM Annotations
- **LLM_annotation.py**: This script takes excerpts of technical documentation from "documentation | folder", which relates to an article of the EU AI Act from "articles.txt". It then prompts one or more LLMs to score the excerpts for compliance (with the respective articles) and plausibility with brief explanations. Finally, it parses the responses and saves them into "llm_annotations.xlsx". Notably, before sending the responses to "llm_annotations.xlsx", it permutes them in the same manner that annotator assignments were permuted to prevent bias. 

### Evaluation
- **eval | folder** - This folder contains any evaluations of the human annotations or LLM annotations.
  - **stats.py** - This script evaluates compliance and plausibility ratings between humans (from "human_annotations.xlsx") and multiple LLMs (from "{model}_annotations.xlsx"). It computes several metrics related to agreement and accuracy between the LLMs and the median human rating. It also calculates descriptive statistics and inter-rater reliability for the human data.

### Dataset Generation
- **excerpt_generation | folder**: This folder contains three scripts — “system_outlines.py”, “violations.py”, and “documentation.py” — which are run in sequence to generate the benchmark dataset. Originally, they wrote to the “system_outlines | folder”, “violations | folder”, and “documentation | folder”, but now produce outputs in” system_outlines_new  | folder”, “violations_new  | folder”, and “documentation_new | folder” to avoid overwriting the original dataset that has since been annotated.
  - **system_outlines.py**: Step 1 of dataset generation. This script reads intended AI use cases from “intended_uses.txt” (based on the EU AI Act) and prompts gpt-4.1-mini to generate 4 descriptions of different AI systems for each use case. It saves the results into subfolders under “system_outlines | folder”. These outlines serve as the basis for later violation and documentation generation.
  - **violations.py**: Step 2 of dataset generation. This script reads AIA articles from “articles.txt” and pairs them with the outlines from “system_outlines | folder”. For each combination of system outline and article, the LLM generates three example scenarios of potential non-compliance. It saves the outputs into subfolders under “violations | folder”.
  - **documentation.py**: Step 3 of dataset generation. The script combines system outlines from “system_outlines | folder” with violation scenarios from “violations | folder” (and relevant article extracts) to generate technical documentation excerpts. Each excerpt reflects the AI system while along with the specified compliance issues (or lack thereof - if no violations are specified). The outputs are written to “documentation | folder”, forming the core of the benchmark dataset.

### Other Files and Folders
- **intended_uses.txt**: This contains a set of 10 intended uses of AI systems. These intended uses are all drawn from the list of high-risk AI (HRAI) systems intended uses in Annex III of the AIA. This document is used during our dataset generation process to produce a distribution of technical documentations reflecting a diversity of HRAI intended uses. 
- **articles.txt**: This contains those AIA articles that dictate requirements for HRAI systems and are the focus of this benchmark dataset (specifically, Articles 9, 10, 12, 14, and 15). This list is used during our generation process to ideate non-compliant AI systems that are diverse in terms of how they violate the AIA and, due to the nuanced nature of their non-compliance, also relatively challenging for LLMs to detect. 
- **requirements.txt**: This contains all the packages that should be required to run this codebase: numpy, pandas, matplotlib, scikit-learn, krippendorff, python-dotenv, openai, anthropic, google-generativeai, openpyxl.
- **rand_assign.txt**: This documents the way in which technical documentation excerpts were permitted during the annotation phase, to prevent bias from annotators noticing a repeating pattern (e.g., every third excerpt is compliant). 
- **system_outlines | folder**: This contains the AI system descriptions corresponding to the intended uses, which was used during the original benchmark generation process. 
- **violations | folder**: This contains the violation scenarios corresponding to the AI system outlines and articles within the EU AI Act, which was used during the original benchmark generation process. 
- **plots | folder**: This contains some of the main plots from AIReg-Bench. 

## Requirements
Create a .env folder with the api keys of the models that you wish to evaluate.
Run pip install requirements.txt for convenient installation of the relevant packages. 
You will need Python 3.10 or above.

## Citation

Please use this Bibtex to cite this work:

@misc{marino2025airegbenchbenchmarkinglanguagemodels,
      title={AIReg-Bench: Benchmarking Language Models That Assess AI Regulation Compliance}, 
      author={Bill Marino and Rosco Hunter and Zubair Jamali and Marinos Emmanouil Kalpakos and Mudra Kashyap and Isaiah Hinton and Alexa Hanson and Maahum Nazir and Christoph Schnabl and Felix Steffek and Hongkai Wen and Nicholas D. Lane},
      year={2025},
      eprint={2510.01474},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2510.01474}, 
}