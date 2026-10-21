# Running on Kaggle (kernels, artifact datasets, code competitions)

Use Kaggle kernels when local GPUs are missing or too small, when the data only lives on
Kaggle, and always for the final scoring run of a code competition.

## Producer / consumer pattern
```
code/train.py ──push──▶ [train kernel, GPU, internet as rules allow]
                            │ outputs: weights, OOF preds, logs
                            ▼
              wait --download → kernels/train/output/vN/
                            │ upload-dataset (private) → owner/<comp>-weights
                            ▼
code/infer.py ──push──▶ [infer kernel, internet OFF, dataset_sources = weights]
                            │ writes submission.csv
                            ▼
        submit.py submit submission.csv --kernel owner/<infer-slug> --yes
```
- Keep training and inference in separate kernels: inference must run within the time limit,
  offline, on the hidden test set.
- Weights move between kernels as **private datasets**, versioned (`upload-dataset` again creates
  a new version). Record which dataset version each submission used in `decisions.md`.

## Commands
```
kaggle_run.py init train --code code/train.py --gpu
kaggle_run.py push train --hours 3            # dry run: quota, sources, cost
kaggle_run.py push train --hours 3 --yes      # after approval if above the GPU threshold
kaggle_run.py wait train --download           # checks for a few minutes; exit 2 = still running, re-run later
kaggle_run.py upload-dataset kernels/train/output/v1 --slug <comp>-weights --title "<comp> weights"   # dry run
kaggle_run.py upload-dataset ... --yes        # after the user's OK
kaggle_run.py init infer --code code/infer.py --gpu --dataset <owner>/<comp>-weights
```
Log every run in `experiments.csv` with real GPU hours (from the run, not the estimate).

## Reproducing a public notebook (fork)
```
kaggle_run.py fork owner/notebook-slug                 # private copy, inputs + Docker image kept, internet off
kaggle_run.py fork owner/notebook-slug --add-dataset me/<comp>-folds   # with your shared folds.csv
kaggle_run.py push notebook-slug --hours 2 --reviewed  # dry run; refused until you pass --reviewed
```
- Nothing is downloaded: competition data, datasets, models and kernel outputs stay mounted on
  Kaggle under the same `/kaggle/input/...` paths, and the original Docker image is kept, so a
  score gap is not an environment difference.
- Read `FORK.md`: provenance, what the fork changed, and an automatic scan for network calls,
  shell commands, obfuscated code and text addressed to AI agents.
- If the original needed internet (pip installs, downloads), the fork fails offline - replace
  those with dataset/model sources, or re-fork with `--keep-internet` only for exploration (code
  competitions score offline).
- Only download to your machine when you have a local GPU and must iterate on one part; even
  then download the competition data, not the models.

## Before pushing
- Estimate GPU hours and pass `--hours`; compare with `submit.py status` (weekly quota left).
- Smoke-test locally on a tiny subset first (1 fold, few steps) — failed Kaggle runs waste quota.
- Write logs to a file in the output folder (`/kaggle/working/train.log`) and print a short
  summary at the end, so `wait --download` gives a small, readable log.

## Code-competition checklist (most common failure causes)
- **Paths**: read from `/kaggle/input/...`, write `submission.csv` to `/kaggle/working/`;
  print discovered paths at start. No hard-coded dataset version folders.
- **Internet off**: no `pip install` from the internet, no model downloads; ship wheels and
  weights as datasets.
- **Hidden test differs**: don't assume file count, image sizes or class set from the public
  test; handle empty/edge inputs.
- **Time**: measure inference on the public test and scale by hidden/public size; keep ≥20%
  margin including TTA and every ensemble member.
- **Memory**: batch the test set, free models between ensemble members, avoid holding all
  predictions for large masks.
- **Determinism / state**: the scoring run starts clean — no reliance on earlier cell state
  or files from a previous run.
- **Format**: run `check_submission.py` on the kernel's own output before submitting.

If the scored submission fails: `kaggle_run.py output infer`, read the log tail, classify the
failure with the list above, fix, re-run, and only then resubmit (each attempt costs a slot).

## Sharing rules
Kernels and datasets stay **private**. Making anything public (code, weights, predictions) needs
a rules check (`rules.md`: sharing, licenses) and the user's explicit OK — private sharing
outside the team is forbidden in most competitions.
