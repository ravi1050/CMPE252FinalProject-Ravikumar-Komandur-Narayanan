# ============================================================
# EXPERIMENT 6.8: AdaLoRA (Adaptive LoRA)
# ============================================================
# Purpose: Compare AdaLoRA (adaptive rank) vs standard LoRA (fixed rank)
#          AdaLoRA automatically adjusts rank per layer
# ============================================================

from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from peft import AdaLoraConfig, get_peft_model, TaskType
import evaluate
from tqdm import tqdm
import torch

# Load fresh model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="microsoft/Phi-3-mini-4k-instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Apply AdaLoRA instead of standard LoRA
# AdaLoRA starts with init_r and prunes to target_r based on importance
adalora_config = AdaLoraConfig(
    init_r=16,               # Starting rank
    target_r=8,              # Target rank after pruning
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, adalora_config)
model.print_trainable_parameters()

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

# Train
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=formatted_train,
    args=TrainingArguments(
        output_dir="/content/drive/MyDrive/phi3-medical-adalora",
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

print("Starting AdaLoRA training...")
trainer_stats = trainer.train()

print(f"\nAdaLoRA training complete!")
print(f"Training time: {trainer_stats.metrics['train_runtime']:.0f} seconds")
print(f"Final loss: {trainer_stats.metrics['train_loss']:.4f}")

# Evaluate
rouge = evaluate.load("rouge")
predictions = []
references = []
correct = 0
total = 0

model.eval()

for i, sample in enumerate(tqdm(eval_dataset, desc="AdaLoRA evaluation")):
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

lora_results = {"rouge1": 0.3594, "rouge2": 0.1578, "rougeL": 0.2848, "accuracy": 0.7550}

print(f"\n{'='*60}")
print(f"COMPARISON: Standard LoRA vs AdaLoRA (Phi-3-Mini, PubMedQA)")
print(f"{'='*60}")
print(f"{'Metric':<25} {'LoRA (r=16)':<15} {'AdaLoRA':<15}")
print(f"{'-'*60}")
print(f"{'ROUGE-1':<25} {lora_results['rouge1']:<15.4f} {rouge_results['rouge1']:<15.4f}")
print(f"{'ROUGE-2':<25} {lora_results['rouge2']:<15.4f} {rouge_results['rouge2']:<15.4f}")
print(f"{'ROUGE-L':<25} {lora_results['rougeL']:<15.4f} {rouge_results['rougeL']:<15.4f}")
print(f"{'Accuracy':<25} {lora_results['accuracy']:<15.2%} {correct/total:<15.2%}")
print(f"{'='*60}")
