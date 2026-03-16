# Detailed Experiment Plan

## Experiment Execution Order

We run experiments in dependency order - later experiments depend on results from earlier ones.

---

## Phase 1: Baselines (Sunday Session 1)

### Experiment 6.1 - Baseline (No Fine-Tuning)

**Purpose:** Measure how each model performs on medical and legal tasks WITHOUT any fine-tuning. This is our control group.

**What we do:**
```
For each model (Phi-3, LLaMA, Qwen, Mistral):
    For each domain (medical, legal):
        1. Load base model (no LoRA)
        2. Run test questions through model
        3. Measure ROUGE and accuracy
        4. Save results
```

**Expected result:** Decent but generic answers. Models know general medical/legal concepts but lack depth.

**This gives us 8 baseline scores** (4 models × 2 domains) to compare everything against.

---

## Phase 2: Core LoRA Experiments (Sunday Session 1-2)

### Experiment 6.2 - In-Domain LoRA (Medical)

**Purpose:** Fine-tune each model on medical data, test on medical data. Does domain-specific training help?

**What we do:**
```
For each model:
    1. Load base model
    2. Apply LoRA config (r=16, alpha=32)
    3. Fine-tune on PubMedQA training set
    4. Evaluate on PubMedQA test set
    5. Compare to Experiment 6.1 baseline
```

**Expected result:** Significantly better medical answers than baseline. This validates Hypothesis 1.

### Experiment 6.3 - In-Domain LoRA (Legal)

**Same as 6.2 but for legal domain.** Fine-tune on legal data, test on legal data.

### Experiment 6.4 - Cross-Domain (Medical → Legal)

**Purpose:** Take the medical-finetuned model, test it on legal questions. Does medical training help or hurt legal performance?

**What we do:**
```
For each model:
    1. Load the medical-finetuned model from Experiment 6.2
    2. Evaluate on legal test set
    3. Compare to baseline (6.1) and legal-finetuned (6.3)
```

**Expected result:** Worse than legal-finetuned, possibly worse than baseline. This validates Hypothesis 2.

### Experiment 6.5 - Cross-Domain (Legal → Medical)

**Same as 6.4 but reversed.** Legal model tested on medical questions.

---

## Phase 3: Advanced Experiments (Week 2-3)

### Experiment 6.6 - LoRA vs Full Fine-Tuning

**Purpose:** Compare LoRA (efficient) vs full fine-tuning (expensive). How much quality do we lose with LoRA?

**What we do:**
```
Pick 1-2 smaller models (Phi-3-Mini, LLaMA-3.2-3B):
    1. Full fine-tune on medical data (all parameters updated)
    2. Evaluate on medical test set
    3. Compare to LoRA results from Experiment 6.2
```

**Why only 1-2 models:** Full fine-tuning of Mistral 7B won't fit on A100. Smaller models will.

**Expected result:** Full fine-tuning slightly better, but LoRA surprisingly close. Shows LoRA is worth the efficiency tradeoff.

### Experiment 6.7 - Hallucination Analysis

**Purpose:** Do fine-tuned models hallucinate less than base models on domain questions?

**What we do:**
```
1. Select 50-100 generated answers from baseline AND fine-tuned models
2. Manually review each for:
   - Factual correctness
   - Made-up citations
   - Contradictions with source material
   - Confident but wrong statements
3. Score: correct / partially correct / hallucinated
4. Compare hallucination rates: baseline vs fine-tuned
```

**This is manual work** - no automation. Budget 2-3 hours for review.

### Experiment 6.8 - Ablation Study (LoRA Rank)

**Purpose:** How does LoRA rank affect performance? Find the sweet spot.

**What we do:**
```
Pick 1 model (Phi-3-Mini) on 1 domain (medical):
    For each rank r = [4, 8, 16, 32]:
        1. Fine-tune with LoRA at that rank
        2. Evaluate on test set
        3. Record: ROUGE score, training time, VRAM used, trainable params
```

**Expected result:** Performance improves from r=4 to r=16, then plateaus or slightly improves at r=32. Diminishing returns.

**This produces a nice chart** for the paper: rank vs performance curve.

### Experiment 6.9 - AdaLoRA and IA3

**Purpose:** Compare alternative PEFT methods to standard LoRA.

**What we do:**
```
Pick 1 model, 1 domain:
    1. Fine-tune with AdaLoRA (adaptive rank)
    2. Fine-tune with IA3 (learned scaling)
    3. Compare to standard LoRA (Experiment 6.2)
```

**AdaLoRA config:**
```python
from peft import AdaLoraConfig
config = AdaLoraConfig(
    init_r=16, target_r=8,  # Start at rank 16, prune to 8
    task_type=TaskType.CAUSAL_LM
)
```

**IA3 config:**
```python
from peft import IA3Config
config = IA3Config(
    task_type=TaskType.CAUSAL_LM
)
```

### Experiment 6.10 - QLoRA

**Purpose:** Can 4-bit quantization + LoRA match standard LoRA quality?

**What we do:**
```
For each model:
    1. Load model in 4-bit (BitsAndBytesConfig)
    2. Apply LoRA
    3. Fine-tune on medical data
    4. Compare to standard LoRA (6.2) and baseline (6.1)
    5. Record VRAM savings
```

**Expected result:** Slightly lower quality than standard LoRA, but dramatically less VRAM. Important practical finding.

---

## Results Table Template

After all experiments, we'll fill this table:

| Experiment | Model | Domain | ROUGE-1 | ROUGE-2 | ROUGE-L | Accuracy | VRAM | Time |
|-----------|-------|--------|---------|---------|---------|----------|------|------|
| 6.1 Baseline | Phi-3 | Medical | | | | | | |
| 6.1 Baseline | Phi-3 | Legal | | | | | | |
| 6.2 LoRA | Phi-3 | Medical | | | | | | |
| 6.3 LoRA | Phi-3 | Legal | | | | | | |
| ... | ... | ... | | | | | | |

---

## Compute Budget Estimate

| Experiment | Runs | Time per run | Total time |
|-----------|------|-------------|-----------|
| 6.1 Baseline | 8 (4 models × 2 domains) | 10 min | ~1.5 hrs |
| 6.2 Medical LoRA | 4 models | 45 min | ~3 hrs |
| 6.3 Legal LoRA | 4 models | 45 min | ~3 hrs |
| 6.4 Cross-domain | 4 models | 10 min (eval only) | ~40 min |
| 6.5 Cross-domain | 4 models | 10 min (eval only) | ~40 min |
| 6.6 Full fine-tune | 2 models | 2 hrs | ~4 hrs |
| 6.7 Hallucination | Manual | - | ~3 hrs |
| 6.8 Ablation | 4 ranks × 1 model | 45 min | ~3 hrs |
| 6.9 AdaLoRA + IA3 | 2 methods × 1 model | 45 min | ~1.5 hrs |
| 6.10 QLoRA | 4 models | 30 min | ~2 hrs |
| **Total** | | | **~22 hrs GPU + 3 hrs manual** |

Colab Pro A100 sessions can run ~8-12 hours before timeout. We'll need 2-3 separate sessions for all training.

---

## Sunday Goal

By end of Sunday, we should have:
- [x] Environment verified (DONE)
- [ ] All datasets downloaded and preprocessed
- [ ] Training pipeline code working
- [ ] Experiment 6.1 (baselines) complete
- [ ] Experiment 6.2 and 6.3 started (at least 1-2 models each)
