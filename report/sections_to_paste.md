# Future Work

Several directions can extend the findings of this project. First, extending experiments to additional open-source architectures such as LLaMA-3.2, Qwen-2.5, and Mistral at different parameter scales would allow cross-architecture comparisons and test whether larger models benefit more from LoRA adaptation. Second, systematically varying the LoRA rank (r = 4, 8, 16, 32) would help identify the optimal trade-off between adapter capacity and overfitting risk on small datasets. Third, comparing LoRA against alternative parameter-efficient methods such as AdaLoRA and IA3 would determine whether adaptive rank allocation or learned scaling vectors offer advantages for domain adaptation. Fourth, investigating how performance scales with training set size using the larger PubMedQA artificial subset (211K samples) and the full MedQuAD corpus (47K samples) would clarify whether the current gains are limited by data volume. Finally, conducting manual expert review of model outputs to assess factual accuracy and hallucination rates would address a key limitation of relying solely on automated metrics.


# References

[1] Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention Is All You Need. Advances in Neural Information Processing Systems (NeurIPS).

[2] Lee, J., Yoon, W., Kim, S., et al. (2020). BioBERT: A Pre-trained Biomedical Language Representation Model for Biomedical Text Mining. Bioinformatics, 36(4), 1234-1240.

[3] Gu, Y., Tinn, R., Cheng, H., et al. (2021). Domain-Specific Language Model Pretraining for Biomedical NLP. ACM Transactions on Computing for Healthcare, 3(1), 1-23.

[4] Alsentzer, E., Murphy, J., Boag, W., et al. (2019). Publicly Available Clinical BERT Embeddings. arXiv:1904.03323.

[5] Chalkidis, I., Fergadiotis, M., Malakasiotis, P., et al. (2020). LEGAL-BERT: The Muppets Straight Out of Law School. Findings of EMNLP 2020, 2898-2904.

[6] Gururangan, S., Marasovic, A., Swayamdipta, S., et al. (2020). Don't Stop Pretraining: Adapt Language Models to Domains and Tasks. Proceedings of ACL, 8342-8360.

[7] Jin, Q., Dhingra, B., Liu, Z., Cohen, W., & Lu, X. (2019). PubMedQA: A Dataset for Biomedical Research Question Answering. Proceedings of EMNLP-IJCNLP, 2567-2577.

[8] Abacha, A.B. & Demner-Fushman, D. (2019). A Question-Entailment Approach to Question Answering. BMC Bioinformatics, 20(511).

[9] Hu, E.J., Shen, Y., Wallis, P., et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. Proceedings of ICLR.

[10] Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient Finetuning of Quantized Language Models. Advances in NeurIPS, 36.

[11] Zhang, Q., Chen, M., Bukharin, A., et al. (2023). AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning. Proceedings of ICLR.

[12] Liu, H., Tam, D., Muqeeth, M., et al. (2022). Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning. Advances in NeurIPS, 35.

[13] Abdin, M., Jacobs, S.A., et al. (2024). Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone. arXiv:2404.14219.

[14] Touvron, H., Lavril, T., Izacard, G., et al. (2023). LLaMA: Open and Efficient Foundation Language Models. arXiv:2302.13971.

[15] Jiang, A.Q., Sablayrolles, A., Mensch, A., et al. (2023). Mistral 7B. arXiv:2310.06825.

[16] Qwen Team. (2024). Qwen2.5 Technical Report. arXiv:2412.15115.

[17] Lin, C.Y. (2004). ROUGE: A Package for Automatic Evaluation of Summaries. Text Summarization Branches Out, 74-81.

[18] Harvard Law School Library Innovation Lab. (2018). Caselaw Access Project. https://case.law/


# Contributions

This project was a collaborative effort with shared responsibilities across all phases. Nathan Howland and Gunanidhi Ramakrishnan led the experiment implementation, including model fine-tuning, evaluation pipeline development, and results generation across all datasets and configurations. Ravikumar Komandur Narayanan contributed to the project proposal, literature review, experiment design, and final report preparation. All members participated in interpreting results, discussing findings, and reviewing the final deliverables.


# Results Summary (paste after "Results" heading, before screenshots)

Legal fine-tuning produced the most dramatic in-domain improvement (+371% F1 on LegalQAEval), suggesting the base model had minimal legal knowledge to begin with. However, cross-domain performance degraded sharply, confirming that domain-specific adaptation comes at the cost of generalization to other domains.

Fine-tuning on MedQUAD produced a strong in-domain F1 gain of +57%. Interestingly, legal QA F1 also improved by +45%, suggesting some transfer of general reasoning ability from clinical QA data. However, PubMedQA F1 dropped by 26%, indicating that MedQUAD's clinical style does not fully generalize to biomedical research questions.

LoRA fine-tuning on PubMedQA improved in-domain F1 by +22%, confirming that domain-specific adaptation helps the model internalize biomedical research reasoning patterns, even with only 0.33% of parameters being trained.

QLoRA achieved nearly identical in-domain performance to standard LoRA (F1 of 0.60 vs 0.61) while using significantly less GPU memory, validating its use as a practical pathway for domain adaptation on resource-constrained hardware.
