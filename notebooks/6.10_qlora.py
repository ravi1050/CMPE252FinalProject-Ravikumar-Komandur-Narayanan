# ============================================================
# EXPERIMENT 6.10: QLoRA (Quantized LoRA)
# ============================================================
# Purpose: Compare QLoRA (4-bit base + LoRA) vs standard LoRA (16-bit base + LoRA)
#          Note: Our Experiment 6.2 already used 4-bit loading (QLoRA)
#          For a fair comparison, we need standard LoRA with 16-bit base model
# ============================================================

from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import evaluate
from tqdm import tqdm
import torch

# --- PART A: Standard LoRA (16-bit base model) ---
# Load model in 16-bit (NOT 4-bit) for standard LoRA comparison

model_16bit, tokenizer_16bit = FastLanguageModel.from_pretrained(
    model_name="microsoft/Phi-3-mini-4k-instruct",
    max_seq_length=2048,
    load_in_4bit=False,  # 16-bit precision
)

print(f"16-bit model loaded. GPU memory: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

# Apply LoRA on 16-bit model
model_16bit = FastLanguageModel.get_peft_model(
    model_16bit,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

model_16bit.print_trainable_parameters()

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
        f"{tokenizer_16bit.eos_token}"
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

# Train standard LoRA (16-bit)
trainer = SFTTrainer(
    model=model_16bit,
    tokenizer=tokenizer_16bit,
    train_dataset=formatted_train,
    args=TrainingArguments(
        output_dir="/content/drive/MyDrive/phi3-medical-lora-16bit",
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

print("Starting Standard LoRA (16-bit) training...")
trainer_stats_16bit = trainer.train()

print(f"16-bit LoRA training time: {trainer_stats_16bit.metrics['train_runtime']:.0f}s")
print(f"16-bit LoRA final loss: {trainer_stats_16bit.metrics['train_loss']:.4f}")
print(f"16-bit LoRA GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")

# Evaluate standard LoRA
rouge = evaluate.load("rouge")
predictions_16bit = []
references_16bit = []
correct_16bit = 0
total_16bit = 0

FastLanguageModel.for_inference(model_16bit)

for i, sample in enumerate(tqdm(eval_dataset, desc="16-bit LoRA eval")):
    prompt = format_pubmedqa_prompt(sample)
    inputs = tokenizer_16bit(prompt, return_tensors="pt", truncation=True, max_length=1792).to("cuda")

    with torch.no_grad():
        outputs = model_16bit.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=True,
        )

    response = tokenizer_16bit.decode(outputs[0], skip_special_tokens=True)
    model_answer = response[len(prompt):].strip()

    predictions_16bit.append(model_answer)
    references_16bit.append(sample['long_answer'])

    if sample['final_decision'].lower() in model_answer.lower()[:50]:
        correct_16bit += 1
    total_16bit += 1

rouge_16bit = rouge.compute(predictions=predictions_16bit, references=references_16bit)

# QLoRA results from Experiment 6.2 (4-bit base + LoRA)
qlora_results = {"rouge1": 0.3594, "rouge2": 0.1578, "rougeL": 0.2848, "accuracy": 0.7550,
                 "time": "112s", "memory": "2.4 GB"}

print(f"\n{'='*60}")
print(f"COMPARISON: QLoRA (4-bit) vs Standard LoRA (16-bit)")
print(f"{'='*60}")
print(f"{'Metric':<25} {'QLoRA (4-bit)':<15} {'LoRA (16-bit)':<15}")
print(f"{'-'*60}")
print(f"{'ROUGE-1':<25} {qlora_results['rouge1']:<15.4f} {rouge_16bit['rouge1']:<15.4f}")
print(f"{'ROUGE-2':<25} {qlora_results['rouge2']:<15.4f} {rouge_16bit['rouge2']:<15.4f}")
print(f"{'ROUGE-L':<25} {qlora_results['rougeL']:<15.4f} {rouge_16bit['rougeL']:<15.4f}")
print(f"{'Accuracy':<25} {qlora_results['accuracy']:<15.2%} {correct_16bit/total_16bit:<15.2%}")
print(f"{'Training time':<25} {qlora_results['time']:<15} {trainer_stats_16bit.metrics['train_runtime']:.0f}s")
print(f"{'GPU memory':<25} {qlora_results['memory']:<15} {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
print(f"{'='*60}")
