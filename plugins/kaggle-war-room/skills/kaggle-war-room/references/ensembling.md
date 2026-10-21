# Ensembling guide

## 0. Prerequisites
- Every model has OOF predictions on the **same folds** (`preds/<exp_id>_oof.npy`) and test
  predictions (`preds/<exp_id>_test.npy`) aligned to the same id order.
- Without shared folds, OOF-based weights are invalid.

## 1. Choose members by diversity
1. `pred_correlation.py preds/*_oof.npy --labels oof_labels.npy` → correlation matrix +
   each model's individual score.
2. Start from the best single model. Greedily add the candidate with the **lowest
   correlation** to the current blend among those whose individual CV is within a reasonable
   margin of the best (e.g. within 2–3× `min_gain`).
3. Stop when the blended CV gain from adding a model is below `min_gain`.
4. Correlation > ~0.98 → the models are near-duplicates; adding them mostly adds inference
   time. Diversity sources, best first: different architecture family > different input
   resolution / crop > different loss / training data > different seed.

## 2. Combining methods
- **Mean of probabilities / logits** — default for classification.
- **Rank average** — when models are calibrated differently or the metric is rank-based (AUC).
- **Weighted average** — fit weights on OOF (e.g. scipy minimize / coordinate search with
  non-negative weights summing to 1). Protect against overfitting: fit weights on folds
  k−1 and evaluate on fold k (nested), or keep weights coarse (steps of 0.05–0.1).
- **Geometric mean** — useful for probabilities when one model is confident and others are not.
- **Stacking** (a meta-model on OOF) — only with enough data and strict nested CV. Usually
  not worth it for CV tasks with small data.
- **Segmentation**: average per-pixel probabilities before thresholding; tune threshold on OOF.
- **Detection**: Weighted Boxes Fusion (WBF) usually beats NMS for merging models; tune
  IoU threshold and skip-box threshold on OOF.

## 3. Validate the blend
- Blended CV beats best single by more than fold-std? If not, the blend is noise.
- Blended LB consistent with blended CV (same offset as single models)?
- Total inference (all members × TTA) fits `inference_time_limit_hours` with margin.

## 4. Anti-patterns
- Weights tuned by LB submissions.
- Blending many public notebooks' outputs (highly correlated + LB-tuned).
- Adding a model only because its LB is high.
- Different fold splits across members, then OOF weight fitting.
