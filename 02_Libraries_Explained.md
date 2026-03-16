# Libraries Explained

## The Stack

```
Your Code (training script)
    ↓
transformers (model loading, tokenization, training loop)
    ↓
peft (LoRA adapter injection)
    ↓
accelerate (GPU memory management)
    ↓
bitsandbytes (quantization for QLoRA)
    ↓
PyTorch (tensor operations on GPU)
    ↓
CUDA (NVIDIA GPU driver)
    ↓
A100 GPU (hardware)
```

---

## transformers (by Hugging Face)

**What:** The main library for working with LLMs. Think of it as the interface to thousands of pre-trained models.

**Key classes we'll use:**

```python
from transformers import AutoModelForCausalLM   # Load a model
from transformers import AutoTokenizer           # Load a tokenizer
from transformers import TrainingArguments       # Configure training
from transformers import Trainer                 # Run training
```

- `AutoModelForCausalLM` - Loads a language model that generates text (causal = predicts next word)
- `AutoTokenizer` - Loads the model's tokenizer (text → numbers → text)
- `TrainingArguments` - All training settings (learning rate, batch size, epochs, etc.)
- `Trainer` - Orchestrates the training loop (feed data, compute loss, update weights, repeat)

**Why "Auto"?** The `Auto` classes automatically detect the model architecture. Whether it's Phi-3, LLaMA, or Mistral, the same code works.

---

## datasets (by Hugging Face)

**What:** Library to load, process, and manage datasets efficiently.

```python
from datasets import load_dataset

dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
```

**Key features:**
- Downloads datasets from Hugging Face Hub (thousands available)
- Memory-efficient (uses Apache Arrow format, doesn't load everything into RAM)
- Built-in `map()` function to preprocess data in parallel
- Train/test split functionality

---

## peft (Parameter-Efficient Fine-Tuning)

**What:** The core library for our project. Implements LoRA, QLoRA, AdaLoRA, IA3.

```python
from peft import LoraConfig, get_peft_model, TaskType

config = LoraConfig(
    r=16,                          # LoRA rank (how many parameters to add)
    lora_alpha=32,                 # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Which layers to adapt
    lora_dropout=0.05,             # Dropout for regularization
    task_type=TaskType.CAUSAL_LM   # Text generation task
)

model = get_peft_model(base_model, config)
```

**Why LoRA instead of full fine-tuning?**
- Full fine-tuning updates ALL parameters (3.8 billion for Phi-3-Mini) → needs huge VRAM
- LoRA adds small adapter matrices to specific layers → updates only ~0.1% of parameters
- Result is nearly as good as full fine-tuning but 10-100x more efficient

**Methods we'll compare:**
| Method | What it does |
|--------|-------------|
| LoRA | Adds low-rank matrices to attention layers |
| QLoRA | LoRA + 4-bit quantized base model (less VRAM) |
| AdaLoRA | Adaptive LoRA - automatically adjusts rank per layer |
| IA3 | Learned rescaling vectors (even fewer parameters than LoRA) |

---

## accelerate (by Hugging Face)

**What:** Handles the complexity of running on GPUs.

We mostly don't interact with it directly. It works behind the scenes when we use `Trainer`. It handles:
- Moving model and data to GPU
- Mixed precision training (float16 instead of float32 = 2x less VRAM)
- Gradient accumulation (simulate larger batch sizes)

---

## bitsandbytes

**What:** Enables quantization - compressing model weights to use less memory.

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,              # Load model in 4-bit precision
    bnb_4bit_quant_type="nf4",      # Quantization type (NormalFloat4)
    bnb_4bit_compute_dtype=torch.float16  # Compute in float16
)
```

**Normal model:** Each parameter = 16 bits (float16) → 3.8B params × 2 bytes = ~7.6 GB
**4-bit model:** Each parameter = 4 bits → 3.8B params × 0.5 bytes = ~1.9 GB

This is how QLoRA works - load the base model in 4-bit, train LoRA adapters in 16-bit.

---

## evaluate

**What:** Framework for computing evaluation metrics.

```python
import evaluate

rouge = evaluate.load("rouge")
results = rouge.compute(predictions=generated_texts, references=reference_texts)
# Returns: rouge1, rouge2, rougeL scores
```

---

## rouge_score

**What:** The actual ROUGE metric computation. Used by `evaluate` under the hood.

**ROUGE measures text overlap:**
- ROUGE-1: Unigram overlap (individual word matching)
- ROUGE-2: Bigram overlap (two-word phrase matching)
- ROUGE-L: Longest common subsequence

**Example:**
```
Reference: "The patient has a fever and headache"
Generated: "The patient presents with fever and mild headache"

ROUGE-1: high (most individual words match)
ROUGE-2: moderate (some bigrams match: "the patient", "fever and")
ROUGE-L: high (long common subsequence exists)
```

Higher ROUGE = generated text is more similar to reference answer.
