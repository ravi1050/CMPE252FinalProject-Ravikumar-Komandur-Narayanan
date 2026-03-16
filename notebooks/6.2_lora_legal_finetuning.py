# ============================================================
# EXPERIMENT 6.2b: In-Domain LoRA Fine-Tuning (Legal)
# ============================================================
# Purpose: Fine-tune Phi-3-Mini on legal data using LoRA,
#          then test on legal data.
#
# NOTE: This requires reloading a fresh Phi-3 model (no medical LoRA).
#       Must disconnect and restart runtime, or load base model fresh.
# ============================================================

# --- CELL 20: Reload fresh Phi-3 model (no LoRA) ---
# We need a clean model to fine-tune on legal data
# Cannot apply a second LoRA on top of the medical one

# Option A: If starting fresh session, run Cells 0-5 first then continue here
# Option B: If continuing session, reload the base model:

from unsloth import FastLanguageModel
import torch

model_legal, tokenizer_legal = FastLanguageModel.from_pretrained(
    model_name="microsoft/Phi-3-mini-4k-instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)

print(f"Fresh Phi-3 loaded for legal fine-tuning")
print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

# --- CELL 21: Apply LoRA for legal training ---
model_legal = FastLanguageModel.get_peft_model(
    model_legal,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

model_legal.print_trainable_parameters()

# --- CELL 22: Prepare legal training data ---
# NOTE: Update format_legal_training_sample based on actual dataset structure

from datasets import load_dataset

legal_dataset = load_dataset("isaacus/LegalQAEval", split="test")

# Split legal data: 80% train, 20% eval
legal_split = legal_dataset.train_test_split(test_size=0.2, seed=42)
legal_train = legal_split['train']
legal_eval = legal_split['test']

def format_legal_training_sample(sample):
    """Format a legal sample as prompt + expected response for training"""
    # TODO: Update field names based on actual dataset structure
    text = (
        "You are a legal expert. Based on the context below, "
        "answer the question.\n\n"
        f"Context: {sample.get('context', sample.get('passage', ''))}\n\n"
        f"Question: {sample.get('question', sample.get('query', ''))}\n\n"
        "Provide a clear and accurate legal answer.\n\n"
        f"{sample.get('answer', sample.get('response', ''))}"
        f"{tokenizer_legal.eos_token}"
    )
    return {"text": text}

formatted_legal_train = legal_train.map(format_legal_training_sample)

print(f"Legal training samples: {len(formatted_legal_train)}")
print(f"Legal eval samples: {len(legal_eval)}")
print(f"\n--- Sample ---")
print(formatted_legal_train[0]['text'][:500] + "...")

# --- CELL 23: Train on legal data ---
from trl import SFTTrainer
from transformers import TrainingArguments

legal_trainer = SFTTrainer(
    model=model_legal,
    tokenizer=tokenizer_legal,
    train_dataset=formatted_legal_train,
    args=TrainingArguments(
        output_dir="/content/drive/MyDrive/phi3-legal-lora",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        warmup_steps=10,
        weight_decay=0.01,
        optim="adamw_8bit",
        seed=42,
        report_to="none",
    ),
    dataset_text_field="text",
    max_seq_length=2048,
    packing=False,
)

print("Starting legal training...")
legal_trainer_stats = legal_trainer.train()

print(f"\nLegal training complete!")
print(f"Training time: {legal_trainer_stats.metrics['train_runtime']:.0f} seconds")
print(f"Final loss: {legal_trainer_stats.metrics['train_loss']:.4f}")

# --- CELL 24: Save legal LoRA adapter ---
model_legal.save_pretrained("/content/drive/MyDrive/phi3-legal-lora")
tokenizer_legal.save_pretrained("/content/drive/MyDrive/phi3-legal-lora")
print("Legal LoRA adapter saved to Google Drive!")

# --- CELL 25: Evaluate legal-finetuned model on legal data ---
import evaluate
from tqdm import tqdm

rouge = evaluate.load("rouge")

def format_legal_prompt(sample):
    """Convert a legal QA sample into a prompt for the model"""
    prompt = (
        "You are a legal expert. Based on the context below, "
        "answer the question.\n\n"
        f"Context: {sample.get('context', sample.get('passage', ''))}\n\n"
        f"Question: {sample.get('question', sample.get('query', ''))}\n\n"
        "Provide a clear and accurate legal answer."
    )
    return prompt

legal_ft_predictions = []
legal_ft_references = []

FastLanguageModel.for_inference(model_legal)

for i, sample in enumerate(tqdm(legal_eval, desc="Legal fine-tuned evaluation")):
    prompt = format_legal_prompt(sample)
    inputs = tokenizer_legal(prompt, return_tensors="pt", truncation=True, max_length=1792).to("cuda")

    with torch.no_grad():
        outputs = model_legal.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=True,
        )

    response = tokenizer_legal.decode(outputs[0], skip_special_tokens=True)
    model_answer = response[len(prompt):].strip()

    legal_ft_predictions.append(model_answer)
    legal_ft_references.append(sample.get('answer', sample.get('response', '')))

    if (i + 1) % 50 == 0:
        print(f"  Processed {i+1}/{len(legal_eval)}")

# --- CELL 26: Display legal fine-tuning results ---
legal_ft_rouge = rouge.compute(predictions=legal_ft_predictions, references=legal_ft_references)

print("\n" + "=" * 60)
print("RESULTS: Legal-finetuned Phi-3 on Legal QA")
print("=" * 60)
print(f"ROUGE-1: {legal_ft_rouge['rouge1']:.4f}")
print(f"ROUGE-2: {legal_ft_rouge['rouge2']:.4f}")
print(f"ROUGE-L: {legal_ft_rouge['rougeL']:.4f}")
print("=" * 60)

# --- CELL 27: Cross-domain - Legal model on medical data ---
# Test if legal training hurts medical performance

med_cross_predictions = []
med_cross_references = []

dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
med_eval = dataset.select(range(800, 1000))

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

for i, sample in enumerate(tqdm(med_eval, desc="Legal model on medical data")):
    prompt = format_pubmedqa_prompt(sample)
    inputs = tokenizer_legal(prompt, return_tensors="pt", truncation=True, max_length=1792).to("cuda")

    with torch.no_grad():
        outputs = model_legal.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=True,
        )

    response = tokenizer_legal.decode(outputs[0], skip_special_tokens=True)
    model_answer = response[len(prompt):].strip()

    med_cross_predictions.append(model_answer)
    med_cross_references.append(sample['long_answer'])

med_cross_rouge = rouge.compute(predictions=med_cross_predictions, references=med_cross_references)

print("\n" + "=" * 60)
print("CROSS-DOMAIN: Legal-trained Phi-3 on Medical QA")
print("=" * 60)
print(f"ROUGE-1: {med_cross_rouge['rouge1']:.4f}")
print(f"ROUGE-2: {med_cross_rouge['rouge2']:.4f}")
print(f"ROUGE-L: {med_cross_rouge['rougeL']:.4f}")
print("=" * 60)
