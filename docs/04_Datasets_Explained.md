# Datasets Explained

## Overview

We need datasets from two domains to test our hypothesis about domain-specific adaptation:

| Domain | Training Dataset | Evaluation Dataset |
|--------|-----------------|-------------------|
| Medical | PubMedQA | PubMedQA (test split) |
| Legal | Caselaw / Legal text | LegalQAEval |

---

## Medical Datasets

### PubMedQA

**Hugging Face:** `qiaojin/PubMedQA`
**Size:** 1,000 labeled samples (pqa_labeled)
**Task:** Given a biomedical research question + abstract context, answer yes/no/maybe

**Sample:**
```json
{
  "question": "Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?",
  "context": "Programmed cell death (PCD) is the regulated death of cells within an organism...",
  "long_answer": "These findings suggest a role for mitochondria in PCD...",
  "final_decision": "yes"
}
```

**How we'll use it:**
- Fine-tuning: Train the model on question + context → generate long_answer
- Evaluation: Compare generated answer to reference long_answer using ROUGE
- Also evaluate classification: does the model predict yes/no/maybe correctly?

**Subsets:**
- `pqa_labeled` - 1,000 expert-labeled samples (we use this)
- `pqa_unlabeled` - 61,249 unlabeled samples (could use for additional training)
- `pqa_artificial` - 211,269 auto-labeled samples (noisy but large)

### MedQuAD (Medical Question Answering Dataset)

**Source:** NIH / NLM
**Size:** ~47,000 question-answer pairs
**Task:** Medical question answering across multiple topics

**Topics include:** diseases, drugs, genetic conditions, clinical trials

**How we'll use it:** Additional training data to supplement PubMedQA for fine-tuning.

---

## Legal Datasets

### Caselaw Access Project

**Source:** Harvard Law School
**Content:** Full text of US court decisions

**How we'll use it:**
- Extract Q&A pairs from legal case summaries
- Or use as raw legal text for continued pretraining
- Provides domain-specific legal language patterns

### LegalQAEval

**Task:** Legal question answering
**How we'll use it:** Evaluation dataset - test how well legal-finetuned models answer legal questions

---

## Data Preprocessing (What We'll Do on Sunday)

### Step 1: Load and Explore

```python
from datasets import load_dataset

# Medical
pubmedqa = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
print(pubmedqa[0])  # See the structure

# Check fields
print(pubmedqa.features)  # What columns exist
print(pubmedqa.num_rows)  # How many samples
```

### Step 2: Format for Training

Models expect a specific input format. We need to convert our datasets into prompt-response pairs:

```python
def format_medical_qa(example):
    prompt = f"""You are a medical expert. Answer the following question based on the context.

Question: {example['question']}
Context: {' '.join(example['context']['contexts'])}

Answer:"""

    response = example['long_answer']

    return {"prompt": prompt, "response": response}
```

### Step 3: Tokenize

Convert text to numbers (tokens) that the model understands:

```python
def tokenize(example):
    full_text = example['prompt'] + " " + example['response']
    return tokenizer(full_text, truncation=True, max_length=512)
```

### Step 4: Split

```python
# 80% train, 10% validation, 10% test
dataset = dataset.train_test_split(test_size=0.2)
test_valid = dataset['test'].train_test_split(test_size=0.5)
```

---

## Dataset Size Considerations

| Dataset | Samples | Training time estimate (A100) |
|---------|---------|------------------------------|
| PubMedQA labeled (1000) | 800 train / 100 val / 100 test | ~30-45 min per model |
| PubMedQA artificial (211K) | Much larger | ~3-5 hours per model |
| MedQuAD (47K) | ~37K train | ~2-3 hours per model |

For our project, starting with PubMedQA labeled (1000 samples) is practical. If results are weak, we can add more data from the larger subsets.

---

## Key Question for Each Dataset

Before training, always check:
1. **What's the input?** (question, context, both?)
2. **What's the expected output?** (free text answer, yes/no, classification?)
3. **How do we measure success?** (ROUGE for text, accuracy for classification)
4. **How much data is available?** (affects training time and potential overfitting)
