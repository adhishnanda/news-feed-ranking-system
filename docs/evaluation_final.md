# Evaluation Strategy

## 1. Ranking Metrics

### Precision@K
Measures relevance in top-K results.

### Recall@K
Measures coverage of relevant items.

### NDCG@K
Measures ranking quality with position importance.

---

## 2. Feed Quality Metrics

### Category Diversity@K
Encourages varied content categories.

### Source Diversity@K
Prevents dominance of single sources.

### Freshness@K
Ensures recent content is prioritized.

---

## 3. Counterfactual Evaluation

Goal:
Estimate performance of a new policy using logged data.

### IPS
Weights outcomes using importance sampling.

### SNIPS
Stabilized version of IPS.

### Clipped IPS
Controls variance.

---

## Key Insight

Counterfactual evaluation allows:
- offline policy comparison
- reduced need for live experiments

---

## Limitation

Because logs are simulated:
- estimates are approximations
- not perfect causal inference

---