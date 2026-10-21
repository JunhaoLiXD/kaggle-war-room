# Anti-shake-up guide

Shake-up = final private ranking differs from the public LB. It punishes teams that fitted
the public LB instead of the underlying distribution.

## 1. How much to trust the public LB
- Public LB noise scales with **1/√(public test size)**. With a few hundred samples, LB
  differences of the order of `min_gain` are noise.
- Estimate noise: std of your CV fold scores × √(fold size / public test size) gives a rough
  LB noise level. Treat LB differences below ~2× that as ties.
- If public/private are split by time, site or patient, the public LB can be systematically
  different from private. Trust CV built to mimic that split.
- Track `LB − CV` for every submission in `experiments.csv`. A **stable offset** means CV is
  trustworthy. A **drifting or sign-flipping offset** means one of them is broken — find out
  which before continuing.

## 2. Signs you are overfitting the public LB
- Choosing between variants that differ by less than the LB noise level.
- Many submissions of near-identical models, keeping the best LB.
- Thresholds, class priors, or blend weights tuned by submitting.
- LB improving while CV is flat or falling (several times in a row).
- Copying top public notebooks whose LB gain has no CV support (they are often LB-tuned).
- A public-notebook "consensus" blend that everyone uses — high shake-up risk, low upside.

## 3. Healthy practice
- Decide by CV; use LB as a sanity check and to calibrate the CV↔LB offset.
- Use submissions to test hypotheses ("does higher resolution transfer to LB?"), not to
  hill-climb.
- Keep fold variance in view: a gain inside one fold-std is not proven.
- Prefer robust gains: more data, better augmentation, ensembles of diverse models, TTA,
  multiple seeds. These rarely shake.
- Avoid fragile gains: per-class thresholds tuned on small val sets, LB-tuned post-processing.

## 4. Final submission hedging (two picks)
- **Primary**: best CV among candidates whose LB agrees with CV (within noise). Usually a
  diverse, CV-weighted blend.
- **Hedge**: low correlation with primary (e.g. OOF corr noticeably lower — check
  `pred_correlation.py`), stable CV across folds, plausible LB. Examples: a blend without the
  public-notebook components, a different backbone family, a different post-processing.
- Do **not** pick two near-copies. Do **not** pick the hedge purely on public LB.
- If the best LB submission is not in the primary/hedge pair, write down why in
  `decisions.md` (e.g. "LB gain within noise, CV worse").
- The user makes the final choice.

## 5. Checks before final selection
- Full inference pipeline run end-to-end within the time limit with ≥20% margin.
- `check_submission.py` passes on both picks.
- Both picks are reproducible from saved code + weights.
