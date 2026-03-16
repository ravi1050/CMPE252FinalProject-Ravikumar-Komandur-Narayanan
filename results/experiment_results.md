# Experiment Results

## Experiment 6.1: Baseline Evaluation (No Fine-Tuning)

**Date:** February 2026
**Model:** Phi-3-Mini-4k-instruct (3.8B parameters, 4-bit quantized)
**Dataset:** PubMedQA (pqa_labeled) - 200 test samples (indices 800-1000)
**GPU:** NVIDIA A100-SXM4-40GB
**Training:** None (base model as downloaded from Hugging Face)

### Results

| Metric | Score |
|--------|-------|
| ROUGE-1 | 0.2265 |
| ROUGE-2 | 0.0822 |
| ROUGE-L | 0.1557 |
| Accuracy (yes/no/maybe) | 59.50% (119/200) |

### Observations
- Model correctly answered the majority of questions (above 33% random baseline)
- ROUGE scores are low, indicating the model's explanations differ significantly in wording from reference answers
- Model has general medical knowledge but lacks domain-specific depth
- This establishes the baseline for comparison with fine-tuned models

---

## Experiment 6.2: In-Domain LoRA Fine-Tuning (Medical)

**Date:** February 2026
**Model:** Phi-3-Mini-4k-instruct (3.8B parameters, 4-bit quantized)
**Dataset:** PubMedQA (pqa_labeled)
- Training: 800 samples (indices 0-800)
- Evaluation: 200 samples (indices 800-1000)
**GPU:** NVIDIA H100 80GB HBM3

### LoRA Configuration

| Parameter | Value |
|-----------|-------|
| Rank (r) | 16 |
| Alpha | 32 |
| Dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj |
| Trainable parameters | 12,582,912 (0.33% of total) |
| Total parameters | 3,833,662,464 |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 3 |
| Batch size | 4 |
| Gradient accumulation steps | 4 |
| Effective batch size | 16 |
| Learning rate | 2e-4 |
| Optimizer | adamw_8bit |
| Precision | bfloat16 |
| Total steps | 150 |
| Training time | 112 seconds |
| Final training loss | 1.3751 |

### Training Loss Progression

| Step | Loss |
|------|------|
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

Loss decreased from 1.59 to 1.30 over training, indicating the model learned from the data.

### Results: Baseline vs Fine-Tuned

| Metric | Baseline | Fine-tuned | Change | % Improvement |
|--------|----------|------------|--------|---------------|
| ROUGE-1 | 0.2265 | 0.3594 | +0.1329 | +59% |
| ROUGE-2 | 0.0822 | 0.1578 | +0.0756 | +92% |
| ROUGE-L | 0.1557 | 0.2848 | +0.1291 | +83% |
| Accuracy | 59.50% | 75.50% | +16.00% | +27% |

### Hypothesis Testing

**Hypothesis 1:** "LoRA fine-tuning on domain-specific data significantly improves task performance compared to base models."

**Result: SUPPORTED**

All metrics showed significant improvement after LoRA fine-tuning:
- ROUGE-1 improved by 59%
- ROUGE-2 nearly doubled (+92%)
- ROUGE-L improved by 83%
- Accuracy improved by 16 percentage points (59.5% → 75.5%)

This was achieved by training only 0.33% of the model's parameters for 112 seconds on 800 samples.

### Key Findings
1. LoRA fine-tuning with only 12.5M trainable parameters (out of 3.8B) produced significant improvements
2. Training time was minimal (112 seconds on H100)
3. GPU memory usage remained low (2.4 GB) due to 4-bit quantization
4. The model's medical QA accuracy jumped from near-random (59.5%) to meaningful (75.5%)
5. ROUGE scores indicate the fine-tuned model generates explanations more aligned with expert reference answers

### Adapter Storage
- Saved to: Google Drive (`/content/drive/MyDrive/phi3-medical-lora`)
- Adapter size: ~10-50 MB (vs 2.3 GB for full model)

---

## Remaining Experiments

| # | Experiment | Status |
|---|-----------|--------|
| 6.1 | Baseline evaluation | ✅ Complete (Phi-3 medical) |
| 6.2 | In-domain LoRA (medical) | ✅ Complete (Phi-3) - Need LLaMA, Qwen, Mistral |
| 6.2 | In-domain LoRA (legal) | Pending |
| 6.3 | Cross-domain transfer | Pending |
| 6.4 | LoRA rank ablation | Pending |
| 6.5 | Training data size ablation | Pending |
| 6.6 | LoRA vs Full fine-tuning | Pending |
| 6.7 | Hallucination analysis | Pending |
| 6.8 | AdaLoRA | Pending |
| 6.9 | IA3 | Pending |
| 6.10 | QLoRA vs standard LoRA | Pending |

---

## GPU Comparison

| Task | A100 (40GB) | H100 (80GB) |
|------|-------------|-------------|
| Baseline eval (200 samples) | ~15 min | ~3.5 min |
| LoRA training (800 samples, 3 epochs) | ~15-20 min (est.) | 112 seconds |
| Fine-tuned eval (200 samples) | ~15 min (est.) | ~3.5 min |

H100 is approximately 4-5x faster than A100 for our workload.
