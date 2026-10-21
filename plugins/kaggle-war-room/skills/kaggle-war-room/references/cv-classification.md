# Computer vision — classification

Ordered roughly by typical impact. Test cheaply first (1 fold, low resolution, few epochs).

## Validation
- StratifiedGroupKFold on the true unit of independence (patient, camera, video, site).
- Detect near-duplicates (perceptual hash) and keep them in the same fold.
- Multi-label: iterative stratification. Imbalanced: report per-class metric too.

## Data & input
- **Resolution** is often the largest lever; try 1.5–2× after the backbone is fixed.
- Aspect ratio: pad vs. crop vs. resize — check which the metric/objects tolerate.
- Domain-specific preprocessing (windowing for CT, color normalization for pathology,
  cropping to the region of interest).
- Clean label noise: inspect high-loss OOF samples.
- External / pseudo-labelled data if `rules.md` allows (pseudo-label with OOF-safe teachers only).

## Models
- Strong default families: ConvNeXt, EfficientNet(V2), NFNet, ViT/EVA/DINOv2-style
  transformers, Swin. Use pretrained weights allowed by rules.
- Mix families in the final blend for diversity.
- Larger backbone at low resolution vs. smaller at high resolution — compare on equal compute.

## Training
- AdamW + cosine schedule with warmup; layer-wise LR decay for transformers.
- Mixed precision, gradient accumulation for large inputs.
- EMA of weights often helps a little and stabilizes CV.
- Loss aligned to metric: BCE/CE baseline; focal or class-weighted for imbalance; for AUC,
  BCE is usually fine; for QWK, regression + optimized thresholds.
- Label smoothing for noisy labels.

## Augmentation
- Flips/rotations (if orientation-invariant), random resized crop, color jitter, blur.
- Mixup / CutMix for larger datasets; careful with small or fine-grained data.
- Keep test-time preprocessing identical to train (except randomness).

## Inference
- TTA: flips (cheap, usually safe); multi-scale (costly). Measure gain vs. time.
- Average fold models; temperature/threshold tuning on OOF only.
- Knowledge distillation into a faster model when inference time is the bottleneck.
