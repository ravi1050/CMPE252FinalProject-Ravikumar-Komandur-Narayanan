# Models Explained

## What is a Large Language Model (LLM)?

An LLM is a neural network trained on massive text data to predict the next word. Given "The patient has a", it predicts "fever" or "headache" based on patterns learned from billions of text examples.

**Parameters** = the learned weights inside the model. More parameters generally means more capability but more compute/memory needed.

---

## Our Four Models

### 1. Phi-3-Mini-4k-instruct (Microsoft)

**Hugging Face:** `microsoft/Phi-3-mini-4k-instruct`
**Parameters:** 3.8 billion
**Context window:** 4,096 tokens (~3,000 words)

**Why this model:**
- Small but surprisingly capable - Microsoft trained it on high-quality curated data
- "Instruct" version = already fine-tuned to follow instructions (Q&A format)
- 3.8B parameters fits comfortably on A100 even for full fine-tuning
- Good baseline to see how a small model responds to domain adaptation

**VRAM needed:**
- Full precision: ~7.6 GB
- LoRA fine-tuning: ~10-12 GB
- Full fine-tuning: ~25-30 GB

---

### 2. LLaMA-3.2 (Meta)

**Hugging Face:** `meta-llama/Llama-3.2-3B-Instruct` (3B version)
**Parameters:** 1B or 3B (we'll use 3B)
**Context window:** 128K tokens

**Why this model:**
- Meta's latest small model, very recent (2024)
- 3B is similar size to Phi-3-Mini - good for direct comparison
- Strong instruction-following capability
- Large community and lots of fine-tuning examples available

**Note:** LLaMA models require accepting Meta's license on Hugging Face. You'll need to:
1. Create a Hugging Face account
2. Go to the model page and accept the license
3. Create an access token at huggingface.co/settings/tokens
4. Set the token in Colab

---

### 3. Qwen-2.5 (Alibaba)

**Hugging Face:** `Qwen/Qwen2.5-3B-Instruct` (3B version)
**Parameters:** 0.5B / 1.5B / 3B / 7B / 14B / 32B / 72B (we'll use 3B)
**Context window:** 32K tokens

**Why this model:**
- Strong multilingual model from Alibaba
- Competitive with LLaMA at same size
- Available in many sizes - good for comparing model scale effects
- No gated access - freely downloadable

---

### 4. Mistral (Mistral AI)

**Hugging Face:** `mistralai/Mistral-7B-Instruct-v0.3`
**Parameters:** 7 billion
**Context window:** 32K tokens

**Why this model:**
- Larger model (7B) - tests whether bigger model benefits more from LoRA
- Known for strong reasoning capabilities
- Uses Sliding Window Attention - efficient for long contexts
- This is our "big" model to compare against the ~3B models

**VRAM needed:**
- Full precision: ~14 GB
- LoRA fine-tuning: ~18-20 GB
- QLoRA (4-bit): ~6-8 GB
- Full fine-tuning: would need ~50+ GB (won't fit on A100 - this is why LoRA matters)

---

## Why These Four?

| Model | Size | Purpose in our study |
|-------|------|---------------------|
| Phi-3-Mini | 3.8B | Small model, high-quality training data |
| LLaMA-3.2 | 3B | Small model, latest Meta architecture |
| Qwen-2.5 | 3B | Small model, different pretraining (multilingual focus) |
| Mistral | 7B | Larger model, tests if size + LoRA beats small + LoRA |

Three ~3B models let us compare architectures at the same scale.
One 7B model tests if more parameters help domain adaptation.

---

## What Does "Instruct" Mean?

Base models just predict next words. They might respond to a question with more questions or random text.

"Instruct" models have been additionally trained (instruction-tuned) to:
- Follow instructions
- Answer questions directly
- Format responses appropriately

We use instruct versions because our task is question-answering. The model already knows HOW to answer - we're teaching it domain-specific KNOWLEDGE through LoRA.

---

## What Happens During Fine-Tuning?

```
Before LoRA (base model):
Q: "What is the standard treatment for atrial fibrillation?"
A: Generic, possibly vague answer

After LoRA on medical data:
Q: "What is the standard treatment for atrial fibrillation?"
A: More specific, medically accurate answer citing rate control,
   rhythm control, anticoagulation
```

The model's general language ability stays intact. LoRA adds domain-specific knowledge on top.
