# CMPE 252 Final Project: PEFT Adaptation Study

This repository contains my CMPE 252 final project work on parameter-efficient fine-tuning for domain-specific question answering and comparison across adaptation strategies.

## Project Focus

The work evaluates how lightweight tuning methods affect instruction-tuned language models in medical and legal settings. The current project materials cover:

- baseline evaluation without task-specific tuning
- LoRA fine-tuning for medical and legal QA
- cross-domain transfer analysis
- rank ablation and training-size ablation
- comparison against full fine-tuning
- PEFT variants including AdaLoRA, IA3, and QLoRA

## Included Work

- experiment scripts in `notebooks/`
- summarized findings in `results/experiment_results.md`
- report source in `report/`
- planning and methodology notes in `00_Project_Overview.md` through `06_Experiment_Plan.md`
- proposal document in `CMPE-252_Ravikumar_FinalProjectProposal.pdf`

## Distinctive Elements In This Repository

Unlike a minimal results-only project layout, this repository keeps the full working trail together:

- planning notes and experiment design documents
- implementation scripts for each experiment track
- report assets and references
- recorded baseline and fine-tuning observations

## Current Status

The repository is structured as an evolving final-project workspace rather than only a polished artifact drop. It includes both the technical implementation path and the supporting academic documentation for the project.
