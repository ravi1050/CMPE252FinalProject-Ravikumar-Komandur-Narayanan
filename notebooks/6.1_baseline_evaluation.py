# ============================================================
# EXPERIMENT 6.1: Baseline Evaluation (No Fine-Tuning)
# ============================================================
# Purpose: Measure how Phi-3-Mini performs on medical QA
#          WITHOUT any training. This is our control group.
#
# What we do:
#   1. Load model as-is from Hugging Face
#   2. Load PubMedQA dataset (1000 medical Q&A samples)
#   3. Ask the model 200 test questions
#   4. Measure ROUGE scores and yes/no/maybe accuracy
#
# Results:
#   ROUGE-1: 0.2265
#   ROUGE-2: 0.0822
#   ROUGE-L: 0.1557
#   Accuracy: 59.50% (119/200)
# ============================================================

# --- CELL 1: Install libraries ---
!pip install unsloth transformers datasets peft accelerate bitsandbytes evaluate rouge_score trl -q

# --- CELL 2: Verify GPU and load Unsloth ---
from unsloth import FastLanguageModel
import torch

print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print("Unsloth loaded successfully")

# --- CELL 3: Load Phi-3-Mini model ---
# Downloads Phi-3-Mini from Hugging Face and loads onto GPU
# load_in_4bit=True compresses the model to use less memory
# max_seq_length=2048 means model can process ~1500 words at once

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="microsoft/Phi-3-mini-4k-instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)

print(f"Model loaded: {model.config._name_or_path}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"GPU memory used: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

# --- CELL 4: Load PubMedQA dataset ---
from datasets import load_dataset

dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")

print(f"Dataset size: {len(dataset)} samples")
print(f"Fields: {list(dataset.features.keys())}")

# --- CELL 5: Create prompt formatting function ---
def format_pubmedqa_prompt(sample):
    """Convert a PubMedQA sample into a prompt for the model"""
    contexts = " ".join(sample['context']['contexts'])

    prompt = (
        "You are a medical expert. Based on the context below, "
        "answer the question.\n\n"
        f"Context: {contexts}\n\n"
        f"Question: {sample['question']}\n\n"
        "Answer with 'yes', 'no', or 'maybe', then explain your reasoning."
    )
    return prompt

# --- CELL 6: Run baseline evaluation on 200 test samples ---
import evaluate
from tqdm import tqdm

rouge = evaluate.load("rouge")

eval_dataset = dataset.select(range(800, 1000))
print(f"Evaluation set size: {len(eval_dataset)} samples")

predictions = []
references = []
correct_decisions = 0
total_decisions = 0

FastLanguageModel.for_inference(model)

for i, sample in enumerate(tqdm(eval_dataset, desc="Running baseline")):
    prompt = format_pubmedqa_prompt(sample)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1792).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=True,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    model_answer = response[len(prompt):].strip()

    predictions.append(model_answer)
    references.append(sample['long_answer'])

    answer_lower = model_answer.lower()
    expected = sample['final_decision'].lower()
    if expected in answer_lower[:50]:
        correct_decisions += 1
    total_decisions += 1

# --- CELL 7: Display baseline results ---
rouge_results = rouge.compute(predictions=predictions, references=references)

print("\n" + "=" * 50)
print("BASELINE RESULTS: Phi-3-Mini on PubMedQA (no fine-tuning)")
print("=" * 50)
print(f"ROUGE-1: {rouge_results['rouge1']:.4f}")
print(f"ROUGE-2: {rouge_results['rouge2']:.4f}")
print(f"ROUGE-L: {rouge_results['rougeL']:.4f}")
print(f"Accuracy (yes/no/maybe): {correct_decisions}/{total_decisions} = {correct_decisions/total_decisions:.2%}")
print("=" * 50)

baseline_results = {
    "model": "Phi-3-Mini",
    "dataset": "PubMedQA",
    "fine_tuned": False,
    "rouge1": rouge_results['rouge1'],
    "rouge2": rouge_results['rouge2'],
    "rougeL": rouge_results['rougeL'],
    "accuracy": correct_decisions / total_decisions,
    "num_samples": total_decisions,
}
