# V2 Upgrade Plan

## What V1 already had
- basic end-to-end ranking pipeline
- Logistic Regression and LightGBM
- basic user/item/context features
- offline evaluation with AUC and accuracy

## What V2 adds
- point-in-time correct historical feature generation
- user and item feature tables with timestamps
- as-of joins to reduce temporal leakage
- time-based train/validation/test split
- temporal validity checks
- production-style stubs for:
  - online feature store
  - candidate generation
  - counterfactual evaluation

## Why this matters
A recommender system should train only on features that would have been available at ranking time. V2 improves realism by enforcing temporal correctness and separating offline training data construction from future online serving design.