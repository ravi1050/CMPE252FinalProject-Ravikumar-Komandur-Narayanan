# CMPE-252 Final Project: Domain Adaptation of Open-Source LLMs Using LoRA

**Student:** Ravikumar
**Course:** CMPE-252 (SJSU)
**Deadline:** March 17, 2026

---

## Project Summary

Fine-tune four open-source LLMs using LoRA (Low-Rank Adaptation) on legal and medical datasets, then evaluate whether domain-specific fine-tuning improves performance over base models.

## Hypothesis

1. LoRA fine-tuning on domain-specific data significantly improves task performance compared to base models
2. Cross-domain transfer (legal model on medical data and vice versa) performs worse than in-domain fine-tuning
3. Smaller models with LoRA can match or approach larger model baseline performance in specialized domains

## Models

| Model | Parameters | Developer |
|-------|-----------|-----------|
| Phi-3-Mini-4k-instruct | 3.8B | Microsoft |
| LLaMA-3.2 | 1B / 3B | Meta |
| Qwen-2.5 | TBD | Alibaba |
| Mistral | 7B | Mistral AI |

## Datasets

**Medical:**
- PubMedQA (qiaojin/PubMedQA) - 1000 labeled samples
- MedQuAD - Medical question answering

**Legal:**
- Caselaw Access Project - Legal case data
- LegalQAEval - Legal question answering

## 10 Experiments

| # | Experiment | Purpose |
|---|-----------|---------|
| 6.1 | Baseline (no fine-tuning) | Measure base model performance on both domains |
| 6.2 | In-domain LoRA (medical) | Fine-tune on medical, test on medical |
| 6.3 | In-domain LoRA (legal) | Fine-tune on legal, test on legal |
| 6.4 | Cross-domain (medical→legal) | Fine-tune on medical, test on legal |
| 6.5 | Cross-domain (legal→medical) | Fine-tune on legal, test on medical |
| 6.6 | LoRA vs Full fine-tuning | Compare parameter-efficient vs full update |
| 6.7 | Hallucination analysis | Manual review of 50-100 outputs for factual errors |
| 6.8 | Ablation study | Vary LoRA rank (4, 8, 16, 32) and see impact |
| 6.9 | AdaLoRA and IA3 | Compare alternative PEFT methods |
| 6.10 | QLoRA | 4-bit quantized LoRA comparison |

## Evaluation Metrics

- ROUGE-1, ROUGE-2, ROUGE-L (text overlap)
- BLEU (n-gram precision)
- Accuracy (for classification tasks like PubMedQA yes/no/maybe)
- Manual hallucination review (50-100 samples)

## Compute

- Google Colab Pro (free for SJSU students)
- NVIDIA A100-SXM4-40GB (confirmed working)
- 100 compute units/month

## Timeline

| Week | Dates | Work |
|------|-------|------|
| 1 | Feb 16 | Data collection, preprocessing, training pipeline |
| 2 | Feb 23 | Run experiments 6.1-6.5 (baseline + in-domain + cross-domain) |
| 3 | Mar 2 | Run experiments 6.6-6.10 (ablation, QLoRA, IA3, etc.) |
| 4 | Mar 9 | Analysis, charts, final paper writing |
| 5 | Mar 17 | Final submission |

## Project Structure

```
AI_MS_Project/
├── 00_Project_Overview.md          ← This file
├── 01_Environment_Setup.md         ← Libraries, GPU, Hugging Face setup
├── 02_Libraries_Explained.md       ← What each library does and why
├── 03_Models_Explained.md          ← Details on each model
├── 04_Datasets_Explained.md        ← Details on each dataset
├── 05_LoRA_Explained.md            ← How LoRA fine-tuning works
├── 06_Experiment_Plan.md           ← Detailed experiment design
├── CMPE-252_Ravikumar_FinalProjectProposal.pdf  ← Original proposal
├── notebooks/                      ← Colab notebooks (when we build them)
└── results/                        ← Experiment results (when we run them)
```
