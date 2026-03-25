# Design Decisions

## Why multi-stage pipeline?

Real-world systems separate:
- retrieval
- ranking
- reranking

This improves:
- scalability
- flexibility

---

## Why simulation?

Problem:
- insufficient real interaction data

Solution:
- build realistic simulator

---

## Why Redis?

- fast feature access
- production standard
- separates offline and online pipelines

---

## Why Logistic Regression?

- simple baseline
- interpretable
- fast

---

## Why ranking metrics?

AUC is insufficient for ranking tasks.

Used:
- Precision@K
- Recall@K
- NDCG@K

---

## Why counterfactual evaluation?

Allows:
- offline testing of new policies
- avoids costly online experiments

---