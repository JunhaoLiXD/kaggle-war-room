# Computer vision — detection

## Validation
- Group folds by source image / video / scene; stratify by object count or class if possible.
- Compute the exact metric (mAP@0.5, mAP@0.5:0.95, F-beta at IoU thresholds, per-image vs
  global) on OOF with the official implementation if provided.
- Images with no objects: check how they are scored.

## Data & input
- Box format conversions are a common bug: xywh vs xyxy, normalized vs pixel, origin.
  Visualize a few ground-truth boxes after every pipeline change.
- Resolution: small objects need higher input size or tiling (SAHI-style sliced inference).
- Remove or fix bad annotations (zero-area, out-of-bounds boxes).

## Models
- One-stage: YOLO family (check license vs `rules.md`), RetinaNet, FCOS.
- Two-stage: Faster/Cascade R-CNN (mmdetection / detectron2).
- Transformers: DETR variants (DINO, RT-DETR, Co-DETR).
- Different families make good blend partners.

## Training
- Strong augmentation: mosaic, mixup, random affine, HSV, flips; turn mosaic off for the last epochs.
- Anchor / input-size tuning to object size distribution.
- Multi-scale training; EMA; longer schedules often keep helping.
- Pseudo-labelling on test (only if rules allow) with high-confidence boxes from OOF-safe models.

## Post-processing (tune on OOF only)
- Score threshold, NMS IoU threshold, max detections per image.
- Soft-NMS for crowded scenes.
- Class-wise thresholds only with enough validation data.

## Inference & ensembling
- TTA: horizontal flip, multi-scale; merge with WBF.
- Blend models with **Weighted Boxes Fusion** (tune iou_thr, skip_box_thr, weights on OOF).
- Measure inference time at full hidden-test size; tiling + TTA + several models adds up fast.
