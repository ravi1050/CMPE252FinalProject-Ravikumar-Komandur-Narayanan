# Experiment Results Summary

## Executive Summary

This document consolidates the completed experimental findings for the CMPE 252 final project. At present, the strongest completed result is the Phi-3 medical QA comparison between the untuned baseline model and an in-domain LoRA fine-tuned variant on PubMedQA.

The current evidence supports the main project hypothesis: parameter-efficient fine-tuning can materially improve task performance on a specialized domain without updating the full model.

## Completed Comparison

### Phi-3 Medical QA: Baseline vs LoRA

| Model | Dataset | Setting | ROUGE-1 | ROUGE-2 | ROUGE-L | Accuracy |
|---|---|---|---:|---:|---:|---:|
| Phi-3-Mini-4k-instruct | PubMedQA | Baseline | 0.2265 | 0.0822 | 0.1557 | 59.50% |
| Phi-3-Mini-4k-instruct | PubMedQA | LoRA fine-tuned | 0.3594 | 0.1578 | 0.2848 | 75.50% |

### Improvement Summary

| Metric | Absolute Change | Relative Improvement |
|---|---:|---:|
| ROUGE-1 | +0.1329 | +59% |
| ROUGE-2 | +0.0756 | +92% |
| ROUGE-L | +0.1291 | +83% |
| Accuracy | +16.00 points | +27% |

## Experimental Context

### Baseline Evaluation

| Item | Value |
|---|---|
| Date | February 2026 |
| Model | Phi-3-Mini-4k-instruct |
| Quantization | 4-bit |
| Dataset | PubMedQA (200 test samples, indices 800-1000) |
| GPU | NVIDIA A100-SXM4-40GB |
| Training | None |

### LoRA Fine-Tuning Setup

| Item | Value |
|---|---|
| Date | February 2026 |
| Model | Phi-3-Mini-4k-instruct |
| Dataset | PubMedQA |
| Training split | 800 samples |
| Evaluation split | 200 samples |
| GPU | NVIDIA H100 80GB HBM3 |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Trainable parameters | 12,582,912 |
| Total parameters | 3,833,662,464 |
| Epochs | 3 |
| Effective batch size | 16 |
| Learning rate | `2e-4` |
| Optimizer | `adamw_8bit` |
| Precision | `bfloat16` |
| Training time | 112 seconds |
| Final training loss | 1.3751 |

## Training Loss Progression

| Step | Loss |
|---:|---:|
| 10 | 1.5931 |
| 20 | 1.4216 |
| 30 | 1.4134 |
| 40 | 1.3747 |
| 50 | 1.3879 |
| 60 | 1.3710 |
| 70 | 1.3424 |
| 80 | 1.3333 |
| 90 | 1.3526 |
| 100 | 1.3643 |
| 110 | 1.3106 |
| 120 | 1.3620 |
| 130 | 1.3447 |
| 140 | 1.3542 |
| 150 | 1.3006 |

## Interpretation

### Main Finding

The completed Phi-3 medical experiment supports the claim that LoRA fine-tuning improves domain-specific QA performance substantially while updating only a small fraction of the model parameters.

### Why This Matters

- only 0.33% of the model parameters were trained
- training completed in under two minutes on H100
- all reported evaluation metrics improved
- the largest relative gains appeared in ROUGE-2 and ROUGE-L, indicating better overlap with reference answers
- classification accuracy improved from 59.50% to 75.50%

## Compute Efficiency Snapshot

| Task | A100 (40GB) | H100 (80GB) |
|---|---|---|
| Baseline evaluation (200 samples) | ~15 min | ~3.5 min |
| LoRA training (800 samples, 3 epochs) | ~15-20 min estimated | 112 seconds |
| Fine-tuned evaluation (200 samples) | ~15 min estimated | ~3.5 min |

## Results Coverage Map

Only completed results are listed as measured values below. Planned items are intentionally left without fabricated metrics.

| Experiment | Status | Measured Results Available |
|---|---|---|
| 6.1 Baseline evaluation | Complete for Phi-3 medical | Yes |
| 6.2 In-domain LoRA, medical | Complete for Phi-3 medical | Yes |
| 6.2 In-domain LoRA, legal | In progress | No |
| 6.3 Cross-domain transfer | Planned | No |
| 6.4 LoRA rank ablation | Planned | No |
| 6.5 Training-size ablation | Planned | No |
| 6.6 LoRA vs full fine-tuning | Planned | No |
| 6.8 AdaLoRA | Script prepared | No |
| 6.9 IA3 | Script prepared | No |
| 6.10 QLoRA | Script prepared | No |

## Next Results To Add

The highest-value additions to this document are:

1. legal-domain LoRA results
2. QLoRA versus standard LoRA comparison
3. AdaLoRA and IA3 comparison on the same model and dataset

Once those runs are complete, this file can be expanded into a consolidated final comparison table across methods, domains, and models.
