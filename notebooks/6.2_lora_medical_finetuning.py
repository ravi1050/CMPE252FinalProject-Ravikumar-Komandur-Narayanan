# ============================================================
# EXPERIMENT 6.2: In-Domain LoRA Fine-Tuning (Medical)
# ============================================================
# Purpose: Fine-tune Phi-3-Mini on medical data using LoRA,
#          then test on medical data. Does domain training help?
#
# Prerequisites: Run Experiment 6.1 first (model + dataset loaded)
#
# What we do:
#   1. Apply LoRA adapters to the loaded model
#   2. Format 800 training samples as prompt + answer pairs
#   3. Train for 3 epochs (~150 steps)
#   4. Evaluate on same 200 test questions as baseline
#   5. Compare: did LoRA improve performance?
# ============================================================

# --- CELL 10: Apply LoRA adapters to Phi-3-Mini ---
# This adds trainable adapter matrices to the attention layers
# Original 3.8B parameters are FROZEN - only adapters train
# r=16 means rank 16 adapters (~4M trainable parameters)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,                    # Rank: size of adapter (our default)
    lora_alpha=32,           # Scaling factor (2x rank)
    lora_dropout=0.05,       # Dropout to prevent overfitting
    target_modules=[         # Which layers get LoRA adapters
        "q_proj",            # Query projection (attention)
        "k_proj",            # Key projection (attention)
        "v_proj",            # Value projection (attention)
        "o_proj",            # Output projection (attention)
    ],
    bias="none",
    use_gradient_checkpointing="unsloth",  # Saves VRAM
    random_state=42,         # For reproducibility
)

# Show trainable vs total parameters
model.print_trainable_parameters()

# --- CELL 11: Prepare training data ---
# Format the first 800 samples into prompt + response pairs
# This is what the model will learn from
# EOS token tells the model where the answer ends

train_dataset = dataset.select(range(0, 800))

def format_training_sample(sample):
    """Format a sample as prompt + expected response for training"""
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

# Apply formatting to all 800 training samples
formatted_train = train_dataset.map(format_training_sample)

print(f"Training samples: {len(formatted_train)}")
print(f"\n--- Sample training text (truncated) ---")
print(formatted_train[0]['text'][:500] + "...")

# --- CELL 12: Configure and run training ---
# This is where the actual LoRA fine-tuning happens
# The model learns from the 800 medical Q&A pairs
# num_train_epochs=3 means 3 full passes through the data
# per_device_train_batch_size=4 means process 4 samples at a time

from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=formatted_train,
    args=TrainingArguments(
        output_dir="./phi3-medical-lora",      # Where to save checkpoints
        num_train_epochs=3,                     # 3 passes through training data
        per_device_train_batch_size=4,          # 4 samples per batch
        gradient_accumulation_steps=4,          # Effective batch size = 4 x 4 = 16
        learning_rate=2e-4,                     # How fast to learn (standard for LoRA)
        fp16=not torch.cuda.is_bf16_supported(),  # Use float16 if bf16 not supported
        bf16=torch.cuda.is_bf16_supported(),      # Use bfloat16 if supported (A100 yes)
        logging_steps=10,                       # Print loss every 10 steps
        warmup_steps=10,                        # Gradually increase learning rate
        weight_decay=0.01,                      # Regularization to prevent overfitting
        optim="adamw_8bit",                     # Memory-efficient optimizer
        seed=42,                                # Reproducibility
        report_to="none",                       # Don't log to wandb etc
    ),
    dataset_text_field="text",                  # Which field contains our formatted text
    max_seq_length=2048,                        # Max tokens per sample
    packing=False,                              # Don't pack multiple samples together
)

print("Training configuration ready. Starting training...")
print(f"Total training samples: {len(formatted_train)}")
print(f"Batch size: 4 x 4 gradient accumulation = 16 effective")
print(f"Epochs: 3")
print(f"Estimated steps: {(len(formatted_train) // 16) * 3}")

# START TRAINING
trainer_stats = trainer.train()

print(f"\nTraining complete!")
print(f"Training time: {trainer_stats.metrics['train_runtime']:.0f} seconds")
print(f"Final loss: {trainer_stats.metrics['train_loss']:.4f}")
print(f"GPU memory used: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

# --- CELL 13: Save the LoRA adapter ---
# Save ONLY the LoRA adapter (small file ~10-50MB)
# The base model is not saved again - just the adapter

model.save_pretrained("phi3-medical-lora")
tokenizer.save_pretrained("phi3-medical-lora")
print("LoRA adapter saved to phi3-medical-lora/")

# --- CELL 14: Evaluate fine-tuned model on test set ---
# Same 200 questions as baseline, but now with LoRA-trained model

ft_predictions = []
ft_references = []
ft_correct_decisions = 0
ft_total_decisions = 0

FastLanguageModel.for_inference(model)

for i, sample in enumerate(tqdm(eval_dataset, desc="Running fine-tuned evaluation")):
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

    ft_predictions.append(model_answer)
    ft_references.append(sample['long_answer'])

    answer_lower = model_answer.lower()
    expected = sample['final_decision'].lower()
    if expected in answer_lower[:50]:
        ft_correct_decisions += 1
    ft_total_decisions += 1

    if (i + 1) % 50 == 0:
        print(f"  Processed {i+1}/{len(eval_dataset)}")

# --- CELL 15: Compare baseline vs fine-tuned results ---
ft_rouge_results = rouge.compute(predictions=ft_predictions, references=ft_references)

print("\n" + "=" * 60)
print("COMPARISON: Baseline vs LoRA Fine-tuned (Phi-3-Mini, PubMedQA)")
print("=" * 60)
print(f"{'Metric':<25} {'Baseline':<15} {'Fine-tuned':<15} {'Change':<15}")
print("-" * 60)
print(f"{'ROUGE-1':<25} {baseline_results['rouge1']:<15.4f} {ft_rouge_results['rouge1']:<15.4f} {ft_rouge_results['rouge1'] - baseline_results['rouge1']:+.4f}")
print(f"{'ROUGE-2':<25} {baseline_results['rouge2']:<15.4f} {ft_rouge_results['rouge2']:<15.4f} {ft_rouge_results['rouge2'] - baseline_results['rouge2']:+.4f}")
print(f"{'ROUGE-L':<25} {baseline_results['rougeL']:<15.4f} {ft_rouge_results['rougeL']:<15.4f} {ft_rouge_results['rougeL'] - baseline_results['rougeL']:+.4f}")

ft_accuracy = ft_correct_decisions / ft_total_decisions
print(f"{'Accuracy':<25} {baseline_results['accuracy']:<15.2%} {ft_accuracy:<15.2%} {ft_accuracy - baseline_results['accuracy']:+.2%}")
print("=" * 60)

if ft_rouge_results['rouge1'] > baseline_results['rouge1']:
    print("\nLoRA fine-tuning IMPROVED performance on medical QA.")
    print("Hypothesis 1 is SUPPORTED for Phi-3-Mini on PubMedQA.")
else:
    print("\nLoRA fine-tuning did NOT improve ROUGE scores.")
    print("Hypothesis 1 is NOT supported for this configuration.")
