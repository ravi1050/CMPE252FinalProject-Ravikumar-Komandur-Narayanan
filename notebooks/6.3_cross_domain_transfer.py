# ============================================================
# EXPERIMENT 6.3: Cross-Domain Transfer
# ============================================================
# Purpose: Test if medical-trained model performs worse on legal
#          questions (and vice versa). Tests Hypothesis 2.
#
# Prerequisites: Experiments 6.1 and 6.2 completed.
#                Medical-finetuned Phi-3 still in memory.
# ============================================================

# --- CELL 16: Load legal dataset ---
# LegalQAEval from Hugging Face - legal question answering

legal_dataset = load_dataset("isaacus/LegalQAEval", split="test")

print(f"Legal dataset size: {len(legal_dataset)} samples")
print(f"Fields: {list(legal_dataset.features.keys())}")
print(f"\n--- Sample ---")
print(legal_dataset[0])

# --- CELL 17: Format legal prompts ---
# Adjust prompt format based on the legal dataset structure
# NOTE: Update this after seeing the dataset fields in Cell 16

def format_legal_prompt(sample):
    """Convert a legal QA sample into a prompt for the model"""
    # TODO: Update field names based on actual dataset structure
    prompt = (
        "You are a legal expert. Based on the context below, "
        "answer the question.\n\n"
        f"Context: {sample.get('context', sample.get('passage', ''))}\n\n"
        f"Question: {sample.get('question', sample.get('query', ''))}\n\n"
        "Provide a clear and accurate legal answer."
    )
    return prompt

# Test on one sample
print(format_legal_prompt(legal_dataset[0]))

# --- CELL 18: Cross-domain test - Medical model on legal data ---
# The medical-finetuned Phi-3 answers legal questions
# We expect WORSE performance than baseline on legal

legal_predictions = []
legal_references = []

FastLanguageModel.for_inference(model)

# Use up to 200 samples for evaluation
legal_eval_size = min(200, len(legal_dataset))
legal_eval = legal_dataset.select(range(legal_eval_size))

for i, sample in enumerate(tqdm(legal_eval, desc="Medical model on legal data")):
    prompt = format_legal_prompt(sample)
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

    legal_predictions.append(model_answer)
    # TODO: Update reference field name based on dataset structure
    legal_references.append(sample.get('answer', sample.get('response', '')))

    if (i + 1) % 50 == 0:
        print(f"  Processed {i+1}/{legal_eval_size}")

# --- CELL 19: Calculate cross-domain results ---
cross_domain_rouge = rouge.compute(predictions=legal_predictions, references=legal_references)

print("\n" + "=" * 60)
print("CROSS-DOMAIN: Medical-trained Phi-3 on Legal QA")
print("=" * 60)
print(f"ROUGE-1: {cross_domain_rouge['rouge1']:.4f}")
print(f"ROUGE-2: {cross_domain_rouge['rouge2']:.4f}")
print(f"ROUGE-L: {cross_domain_rouge['rougeL']:.4f}")
print("=" * 60)

cross_domain_results_med_to_legal = {
    "model": "Phi-3-Mini",
    "trained_on": "Medical (PubMedQA)",
    "tested_on": "Legal (LegalQAEval)",
    "rouge1": cross_domain_rouge['rouge1'],
    "rouge2": cross_domain_rouge['rouge2'],
    "rougeL": cross_domain_rouge['rougeL'],
}
