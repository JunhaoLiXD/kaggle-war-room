# Computer vision — segmentation

## Validation
- Group folds by image source (patient, scan, slide, scene). Tiles from one image stay together.
- Compute the exact competition metric on OOF (Dice per image vs. global, empty-mask
  handling, Hausdorff, etc.). Global vs per-image Dice can rank models differently.

## Data & input
- Resolution and tiling: full image vs. tiles with overlap; small structures need high res.
- Sampling: oversample tiles with positive pixels; keep some empty tiles to learn negatives.
- 2.5D for volumetric data: stack neighbouring slices as channels.
- Normalization per modality (CT windowing, per-slide stain normalization).

## Models
- U-Net / U-Net++ / FPN / DeepLabV3+ decoders with strong encoders (EfficientNet,
  ConvNeXt, ResNeSt, Mix Transformer / SegFormer).
- 3D U-Net or 2.5D for volumes; compare on equal compute.
- Auxiliary classification head ("is there any mask?") to suppress false positives on empty images.

## Training
- Loss: BCE + Dice (or Lovász / Focal-Tversky for imbalance). Align with the metric.
- Deep supervision helps on small datasets.
- Freeze-then-unfreeze encoder for small data; EMA; mixed precision.

## Augmentation
- Geometric (flips, rotate90, shift-scale-rotate, elastic for medical), applied to image
  and mask identically.
- Intensity (brightness, contrast, gamma, blur, noise). CutMix variants with mask mixing.

## Post-processing (tune on OOF only)
- Probability threshold; minimum component size; remove masks when the classifier head says empty.
- Morphological ops (fill holes, open/close) when shapes are known to be compact.
- For tiled inference: overlap + Gaussian/linear weighting at tile borders.

## Inference
- TTA: flips (and rotate90 if invariant); average probabilities, then threshold.
- Blend: average probabilities across models, re-tune threshold on blended OOF.
- RLE encoding: verify column-major vs row-major and 1-indexing against sample_submission.
