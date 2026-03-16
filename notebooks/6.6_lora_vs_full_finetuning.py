# ============================================================
# EXPERIMENT 6.6: LoRA vs Full Fine-Tuning
# ============================================================
# Purpose: Compare LoRA (0.33% params) vs full fine-tuning (100%)
#          Does full fine-tuning significantly beat LoRA?
#
# NOTE: Full fine-tuning of 3.8B model needs ~25-30GB VRAM
#       H100 (80GB) can handle this. A100 (40GB) also works.
# ============================================================

from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import evaluate
from tqdm import tqdm
import torch

# Load model WITHOUT 4-bit quantization for full fine-tuning
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="microsoft/Phi-3-mini-4k-instruct",
    max_seq_length=2048,
    load_in_4bit=False,  # Full precision for full fine-tuning
)

print(f"Model loaded in full precision")
print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

# NO LoRA - train all parameters
# Just set all parameters to trainable
for param in model.parameters():
    param.requires_grad = True

total_params = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params: {total_params:,}")
print(f"Trainable: {trainable:,} (100%)")

# Load and format dataset
dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
train_dataset = dataset.select(range(0, 800))
eval_dataset = dataset.select(range(800, 1000))

def format_training_sample(sample):
    contexts = " ".join(sample['context']['contexts'])
    text = (
        "You are a medical expert. Based on the context below, "
        "answer the question.\n\n"
        f"Context: {contexts}\n\n"
        f"Question: {sample['question']}\n\n"
        "Answer with 'yes', 'no', or 'maybe', then explain your reasoning.\n\n"
        f"{sample['final_decision']}. {sample['long_answer']}"
        f"{tokenizer.eos_token}"
    )
    return {"text": text}

def format_pubmedqa_prompt(sample):
    contexts = " ".join(sample['context']['contexts'])
    prompt = (
        "You are a medical expert. Based on the context below, "
        "answer the question.\n\n"
        f"Context: {contexts}\n\n"
        f"Question: {sample['question']}\n\n"
        "Answer with 'yes', 'no', or 'maybe', then explain your reasoning."
    )
    return prompt

formatted_train = train_dataset.map(format_training_sample)

# Train - lower learning rate for full fine-tuning, smaller batch
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=formatted_train,
    args=TrainingArguments(
        output_dir="/content/drive/MyDrive/phi3-medical-full-ft",
        num_train_epochs=3,
        per_device_train_batch_size=2,          # Smaller batch - more VRAM needed
        gradient_accumulation_steps=8,          # Effective batch size still 16
        learning_rate=5e-5,                     # Lower LR for full fine-tuning
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        warmup_steps=10,
        weight_decay=0.01,
        optim="adamw_torch",                   # Standard optimizer for full FT
        seed=42,
        report_to="none",
    ),
    dataset_text_field="text",
    max_seq_length=2048,
    packing=False,
)

print("Starting FULL fine-tuning (all parameters)...")
trainer_stats = trainer.train()

print(f"\nFull fine-tuning complete!")
print(f"Training time: {trainer_stats.metrics['train_runtime']:.0f} seconds")
print(f"Final loss: {trainer_stats.metrics['train_loss']:.4f}")
print(f"GPU memory peak: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")

# Evaluate
rouge = evaluate.load("rouge")
predictions = []
references = []
correct = 0
total = 0

model.eval()

for i, sample in enumerate(tqdm(eval_dataset, desc="Full FT evaluation")):
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

    if sample['final_decision'].lower() in model_answer.lower()[:50]:
        correct += 1
    total += 1

rouge_results = rouge.compute(predictions=predictions, references=references)

# Compare with LoRA results from Experiment 6.2
lora_results = {"rouge1": 0.3594, "rouge2": 0.1578, "rougeL": 0.2848, "accuracy": 0.7550}

print(f"\n{'='*60}")
print(f"COMPARISON: LoRA vs Full Fine-Tuning (Phi-3-Mini, PubMedQA)")
print(f"{'='*60}")
print(f"{'Metric':<25} {'LoRA (0.33%)':<15} {'Full FT (100%)':<15}")
print(f"{'-'*60}")
print(f"{'ROUGE-1':<25} {lora_results['rouge1']:<15.4f} {rouge_results['rouge1']:<15.4f}")
print(f"{'ROUGE-2':<25} {lora_results['rouge2']:<15.4f} {rouge_results['rouge2']:<15.4f}")
print(f"{'ROUGE-L':<25} {lora_results['rougeL']:<15.4f} {rouge_results['rougeL']:<15.4f}")
print(f"{'Accuracy':<25} {lora_results['accuracy']:<15.2%} {correct/total:<15.2%}")
print(f"{'Training time':<25} {'112s':<15} {trainer_stats.metrics['train_runtime']:.0f}s")
print(f"{'GPU memory':<25} {'2.4 GB':<15} {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
print(f"{'='*60}")
