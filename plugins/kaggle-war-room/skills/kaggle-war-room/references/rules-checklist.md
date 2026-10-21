# Rules checklist (Phase 0)

Read **Overview, Evaluation, Rules, Data** pages in full (via the official `kaggle` CLI/API
where possible, otherwise ask the user to paste the page text). Fill every item in
`rules.md`. Write "not stated" rather than guessing; list unclear items for the user.

## Must extract
1. **Metric** — exact formula, averaging (macro/micro/per-image), thresholds, IoU levels,
   how empty masks/boxes are scored, any per-class weighting.
2. **Direction** — higher or lower is better. Copy into `settings.yaml` `metric.higher_is_better`.
3. **Submission format** — columns, header, id ordering, data types, encodings (RLE
   column-major vs row-major, 1-indexed?), box format (xywh / xyxy, normalized?), file name.
4. **Daily submission limit** and **number of final picks**. Copy into `settings.yaml`.
5. **Deadlines** — entry, team merger, final submission (UTC). Copy final deadline into
   `settings.yaml` `competition.deadline`.
6. **Public / private split** — size and how it was split (random / time / site / patient).
   This decides how much to trust the public LB.
7. **External data** — allowed? Must it be public and free? Must it be posted in the
   official external-data thread before a date?
8. **Pretrained models** — allowed? License constraints (e.g. commercial use), must be
   publicly available?
9. **Code competition** — yes/no; inference time limit (CPU and GPU separately); internet
   off?; hardware; hidden test size vs public test (inference time scales with it).
   Copy the limit into `settings.yaml` `compute.inference_time_limit_hours`.
10. **Teams** — max size; **no private code sharing between teams**; merger rules.
11. **Public code reuse** — license of public notebooks (e.g. Apache 2.0) and attribution needs.
12. **Hand labelling** — of test data, forbidden? Of training data, allowed?
13. **Winner obligations** — code license, documentation, reproducibility requirements.
14. **Data license** — usage restrictions on competition data.

## Also capture
- Host clarifications in pinned discussion threads (date + URL).
- Anything unusual: multi-stage competitions, rerun on new data, time-series leakage rules,
  restrictions on specific libraries.

## After writing
Set status line to "DRAFT — awaiting user confirmation" and **ask the user to confirm**.
After confirmation, change it to "CONFIRMED <date>".
