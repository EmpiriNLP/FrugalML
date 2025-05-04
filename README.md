# Frugal ML: Towards a Cost-Effective Domain Adaptation for NLP

This is a project from **EmpiriNLP** team for the `COMP0087: Statistical Natural Language Processing` module at University College London (UCL). In this work, multiple parameter-efficient fine-tuning techniques such as LoRA, Prefix-Tuning, are compared among themselves along their performance and computational costs.

Source code for experiments are stored in different branches, whereas the final report is kept in main.

# Abstract
This study investigates cost-effective fine-tuning strategies for large language models (LLMs) within the financial question answering (QA) domain. A set of parameter-efficient fine-tuning (PEFT) approaches- prefix tuning, low-rank adaptation (LoRA), adapters, and programmatic prompts- were evaluated with the FinQA dataset as the benchmark and computation cost measured in floating point operations (FLOs). Of the evaluated methods, prefix tuning achieved the best trade-off between performance and efficiency, while programmatic prompting demonstrated marked accuracy improvements without incurring training costs. 
This project represents preliminary steps towards a cost-conscious and accessible system of delivering financial knowledge.