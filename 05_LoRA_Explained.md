# LoRA Explained (Low-Rank Adaptation)

## The Problem

A 3.8B parameter model has 3,800,000,000 weights to update during training.

**Full fine-tuning:**
- Update all 3.8B parameters
- Needs ~25-30 GB VRAM just for the model + gradients + optimizer states
- Slow, expensive, risk of "catastrophic forgetting" (model forgets general knowledge)

**We need a smarter approach.**

---

## The LoRA Idea (Simple Explanation)

Instead of updating the entire model, LoRA adds tiny adapter matrices to specific layers and only trains those.

**Analogy:** Imagine you have an expert doctor (the base model). Instead of sending them back to medical school (full fine-tuning), you give them a specialized handbook for cardiology (LoRA adapter). They keep all their general knowledge and gain new specialized knowledge.

---

## How It Works (Technical but Simple)

In a transformer model, attention layers have large weight matrices (e.g., 4096 x 4096 = 16 million parameters each).

**Full fine-tuning** updates this entire 4096 x 4096 matrix.

**LoRA** decomposes the update into two small matrices:

```
Original weight matrix W: 4096 x 4096 (16M parameters) → FROZEN, not updated

LoRA adds:
  Matrix A: 4096 x 16  (65K parameters) → TRAINED
  Matrix B: 16 x 4096  (65K parameters) → TRAINED

Total LoRA parameters: 130K vs 16M = less than 1% of original
```

The "16" here is the **rank** (r) - the key hyperparameter. Higher rank = more capacity but more parameters.

**During inference:**
```
Output = W·x + (B·A)·x
         ↑       ↑
      original  LoRA adapter
      (frozen)  (trained)
```

---

## LoRA Configuration Explained

```python
from peft import LoraConfig, TaskType

config = LoraConfig(
    r=16,                              # Rank: size of adapter matrices
    lora_alpha=32,                     # Scaling factor (usually 2x rank)
    target_modules=["q_proj", "v_proj"],  # Which layers to adapt
    lora_dropout=0.05,                 # Dropout for regularization
    task_type=TaskType.CAUSAL_LM       # Text generation task
)
```

### r (rank) - THE key parameter
- **r=4:** Very small adapter, fewer trainable params, may underfit
- **r=8:** Small, good for simple tasks
- **r=16:** Medium, good default for most tasks
- **r=32:** Large, closer to full fine-tuning, may overfit on small datasets
- **Experiment 6.8 (ablation)** tests r=4, 8, 16, 32 to find the sweet spot

### lora_alpha (scaling)
- Controls how much the LoRA update affects the output
- Rule of thumb: set to 2x rank (r=16 → alpha=32)
- Higher alpha = LoRA has more influence on output

### target_modules
- Which layers inside the model get LoRA adapters
- `q_proj` and `v_proj` are attention layers (Query and Value projections)
- Can also target `k_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- More modules = more trainable parameters = better results but more VRAM

### lora_dropout
- Randomly drops some LoRA connections during training
- Prevents overfitting (especially with small datasets like our 1000-sample PubMedQA)

### task_type
- `CAUSAL_LM` = text generation (predict next word)
- This is what we use for all our models

---

## Trainable Parameters Comparison

For Phi-3-Mini (3.8B parameters):

| Method | Trainable params | % of model | VRAM needed |
|--------|-----------------|------------|-------------|
| Full fine-tuning | 3,800,000,000 | 100% | ~30 GB |
| LoRA (r=16) | ~4,000,000 | ~0.1% | ~12 GB |
| LoRA (r=4) | ~1,000,000 | ~0.03% | ~10 GB |
| QLoRA (r=16) | ~4,000,000 | ~0.1% | ~6 GB |

---

## LoRA vs QLoRA vs AdaLoRA vs IA3

### LoRA (our main method)
- Adds low-rank matrices to attention layers
- Base model in float16 (16-bit)
- Good balance of quality and efficiency

### QLoRA (Experiment 6.10)
- Same as LoRA BUT base model is quantized to 4-bit
- 4-bit base model + 16-bit LoRA adapters
- Uses ~50% less VRAM than regular LoRA
- Slightly lower quality due to quantization

```python
# QLoRA setup
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    quantization_config=bnb_config
)
```

### AdaLoRA (Experiment 6.9)
- Adaptive LoRA - automatically adjusts rank per layer
- Some layers get higher rank (more important), some get lower
- Potentially better than fixed-rank LoRA
- Slightly more complex to configure

### IA3 (Experiment 6.9)
- Infused Adapter by Inhibiting and Amplifying Inner Activations
- Even simpler than LoRA - just learns scaling vectors
- Fewer trainable parameters than LoRA
- Faster training, but may not capture as much domain knowledge

---

## The Training Loop (What Actually Happens)

```
For each batch of training data:
    1. Feed input through model (forward pass)
       - Base model weights are FROZEN
       - LoRA adapter weights are active

    2. Compute loss (how wrong was the prediction?)
       - Compare model output to expected answer
       - Cross-entropy loss for language modeling

    3. Compute gradients (which direction to update?)
       - Only compute gradients for LoRA parameters
       - Base model gradients are NOT computed (saves VRAM and time)

    4. Update LoRA weights (backward pass)
       - Optimizer adjusts LoRA matrices A and B
       - Base model stays exactly the same

Repeat for N epochs (full passes through training data)
```

---

## After Training: Saving and Using the Adapter

```python
# Save only the LoRA adapter (tiny file, ~10-50 MB)
model.save_pretrained("phi3-medical-lora")

# Load base model + adapter for inference
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
model = PeftModel.from_pretrained(base_model, "phi3-medical-lora")

# Generate answer
inputs = tokenizer("What is the treatment for...", return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0]))
```

The saved adapter is only 10-50 MB vs the full model at ~7+ GB. This is another advantage of LoRA - you can have many domain-specific adapters sharing one base model.
