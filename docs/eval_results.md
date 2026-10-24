# Evaluation results

Document: `sample_contract.pdf` · embeddings: `sentence-transformers` · top_k=5 · min_relevance=0.2

- Retrieval hit rate @5: **10/10** (100%), MRR 0.85
- Correct abstentions (unanswerable questions): **5/5**

| Question | Result | Top score | Rank |
|---|---|---|---|
| How many days notice is needed to terminate for convenience? | ✅ | 0.68 | 1 |
| How long does the provider have to remedy a material breach? | ✅ | 0.43 | 2 |
| Which law governs the agreement? | ✅ | 0.35 | 1 |
| What is the monthly fee? | ✅ | 0.48 | 1 |
| Within how many days must invoices be paid? | ✅ | 0.68 | 1 |
| What is the limit of the provider's liability? | ✅ | 0.62 | 1 |
| How long must confidential information be kept secret? | ✅ | 0.22 | 1 |
| How long is the initial term of the agreement? | ✅ | 0.46 | 2 |
| How much notice is needed to stop the automatic renewal? | ✅ | 0.38 | 2 |
| Who are the parties to the agreement? | ✅ | 0.32 | 1 |
| What is the CEO's favourite colour? | ✅ | 0.12 | - |
| Give me a recipe for chocolate cake | ✅ | 0.05 | - |
| What is the capital of France? | ✅ | -0.03 | - |
| How do I reset my router password? | ✅ | 0.06 | - |
| Who won the football world cup? | ✅ | 0.03 | - |
