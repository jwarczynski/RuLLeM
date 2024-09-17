# Leveraging Large Language Models for Building Interpretable Rule-Based Data-to-Text Systems

This repository contains the code and resources for the paper titled **"Leveraging Large Language Models for Building Interpretable Rule-Based Data-to-Text Systems"** by Jędrzej Warczyński, Mateusz Lango, and Ondřej Dušek. 

## Paper Abstract

In this work, we introduce a straightforward approach that employs a large language model to automatically implement a fully interpretable rule-based data-to-text system in pure Python. Our experimental evaluation on the WebNLG dataset demonstrates that this system produces text of superior quality compared to zero-shot LLMs, as measured by the BLEU and BLEURT metrics. Moreover, it generates fewer hallucinations than the fine-tuned BART baseline. Additionally, the proposed method generates text significantly faster than neural approaches, requiring only a single CPU for processing.

## Repository Contents

This repository contains:

- Python implementation of the rule-based data-to-text system.
- Scripts for experimental evaluation, including BLEU and BLEURT metrics.
- Scripts for dataset augmenatation with graph clustering algorithm.

## Link to the Paper

For a detailed explanation of our approach and findings, please refer to our paper: [Leveraging Large Language Models for Building Interpretable Rule-Based Data-to-Text Systems](https://aclanthology.org/2024.inlg-main.48/).

## Acknowledgements
This work was co-funded by the European Union (ERC, NG-NLG, 101039303).

<img width="421" alt="LOGO_ERC-FLAG_FP" src="https://github.com/user-attachments/assets/cd7b961b-6560-4138-8070-a62c3045e385">
