# Environment Setup Guide

## Google Colab Pro Setup

1. Sign in at `colab.google.com`
2. Colab Pro activated (free for SJSU students)
3. Runtime → Change runtime type → **A100 GPU**

## Verified Configuration

```
PyTorch version: 2.9.0+cu128
CUDA available: True
GPU: NVIDIA A100-SXM4-40GB
VRAM: 40 GB
```

## Step 1: Install Libraries

Run this first in every new Colab notebook:

```python
!pip install transformers datasets peft accelerate bitsandbytes evaluate rouge_score
```

**What each install does:**

| Library | Purpose | Why we need it |
|---------|---------|---------------|
| `transformers` | Load and use AI models from Hugging Face | This is how we load Phi-3, LLaMA, Qwen, Mistral |
| `datasets` | Load datasets from Hugging Face Hub | This is how we load PubMedQA, legal datasets |
| `peft` | Parameter-Efficient Fine-Tuning (LoRA, QLoRA, IA3, AdaLoRA) | Core of our project - fine-tune models efficiently |
| `accelerate` | GPU memory management and distributed training | Handles moving data between CPU and GPU efficiently |
| `bitsandbytes` | 4-bit and 8-bit quantization | Needed for QLoRA (Experiment 6.10) - fits bigger models in less VRAM |
| `evaluate` | Compute evaluation metrics | Framework to run ROUGE, BLEU scores |
| `rouge_score` | ROUGE metric implementation | Measures text overlap between generated and reference answers |

## Step 2: Verify GPU

```python
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

**What this does:**
- `torch` is PyTorch - the deep learning framework that runs computations on GPU
- `CUDA` is NVIDIA's GPU computing platform - if CUDA is available, PyTorch can use the GPU
- We check GPU name and VRAM to confirm we have A100 with 40GB

**If CUDA shows False:** You forgot to change runtime type to GPU. Go to Runtime → Change runtime type → A100 GPU.

## Step 3: Verify Model Loading

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
print(f"Phi-3-Mini tokenizer loaded. Vocab size: {tokenizer.vocab_size}")
```

**What this does:**
- A **tokenizer** converts human text into numbers the model understands
- Example: "Hello world" → [15043, 3186]
- Each model has its own tokenizer (different vocabulary)
- `AutoTokenizer.from_pretrained()` downloads the tokenizer from Hugging Face Hub
- We test with Phi-3-Mini to confirm Hugging Face access works

**The HF_TOKEN warning is safe to ignore.** All our models and datasets are public. The token only matters for private/gated models.

## Step 4: Verify Dataset Loading

```python
from datasets import load_dataset

dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
print(f"PubMedQA loaded. Size: {len(dataset)} samples")
print(f"Sample: {dataset[0]['question'][:100]}...")
```

**What this does:**
- `load_dataset()` downloads a dataset from Hugging Face Hub
- `"qiaojin/PubMedQA"` is the dataset name on Hugging Face
- `"pqa_labeled"` is the subset (labeled version with yes/no/maybe answers)
- `split="train"` loads the training split
- PubMedQA has 1000 labeled medical question-answer pairs

## All Verified ✓

Once all 4 steps run without errors, the environment is ready for training.
