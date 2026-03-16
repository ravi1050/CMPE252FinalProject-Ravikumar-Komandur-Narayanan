# ============================================================
# EXPERIMENT 6.5: Training Data Size Ablation
# ============================================================
# Purpose: Test how training data size affects performance
#          Sizes tested: 10% (80), 50% (400), 100% (800)
#
# NOTE: Change TRAIN_PERCENTAGE each run, restart runtime between runs
# ============================================================

from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import evaluate
from tqdm import tqdm
import torch

TRAIN_PERCENTAGE = 10  # CHANGE THIS: 10, 50, 100

# Calculate sample count
train_sizes = {10: 80, 50: 400, 100: 800}
train_count = train_sizes[TRAIN_PERCENTAGE]

# Load fresh model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="microsoft/Phi-3-mini-4k-instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Apply LoRA (r=16, same as Experiment 6.2)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# Load and format dataset
dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
train_dataset = dataset.select(range(0, train_count))
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

print(f"\n=== TRAINING SIZE = {TRAIN_PERCENTAGE}% ({train_count} samples) ===")

# Train
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=formatted_train,
    args=TrainingArguments(
        output_dir=f"/content/drive/MyDrive/phi3-medical-lora-{TRAIN_PERCENTAGE}pct",
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

trainer_stats = trainer.train()

print(f"Training time: {trainer_stats.metrics['train_runtime']:.0f} seconds")
print(f"Final loss: {trainer_stats.metrics['train_loss']:.4f}")

# Evaluate
rouge = evaluate.load("rouge")
predictions = []
references = []
correct = 0
total = 0

FastLanguageModel.for_inference(model)

for i, sample in enumerate(tqdm(eval_dataset, desc=f"Eval {TRAIN_PERCENTAGE}% data")):
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

print(f"\n{'='*60}")
print(f"RESULTS: {TRAIN_PERCENTAGE}% Training Data ({train_count} samples)")
print(f"{'='*60}")
print(f"Training time: {trainer_stats.metrics['train_runtime']:.0f}s")
print(f"Final loss: {trainer_stats.metrics['train_loss']:.4f}")
print(f"ROUGE-1: {rouge_results['rouge1']:.4f}")
print(f"ROUGE-2: {rouge_results['rouge2']:.4f}")
print(f"ROUGE-L: {rouge_results['rougeL']:.4f}")
print(f"Accuracy: {correct}/{total} = {correct/total:.2%}")
print(f"{'='*60}")
