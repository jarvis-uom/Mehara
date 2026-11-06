# Retraining run 3: evaluation rules, fixed before the run (3 Nov 2026)

**Status: fixed** (agreed by the team on 3 Nov 2026), before any run 3 code ran on Kaggle. Nothing below may change after the results are seen.

It is written after run 2's results (`RESULTS-run2.md`) and the independent review. Run 1's and run 2's rules stay as they were, for those runs.

**Question.** Can the radar U-Net keep run 2's storage gain without losing its skill on expert flood labels? And, against a control that sees the same photo-labelled scenes, does the "lake high enough" rule help beyond photo labels?

## Design

- **Reservoirs and folds:** the same 103 reservoirs and the same five neighbour folds as run 2 (run 2's `folds.csv`), with the same cleaned storage sheet.
- **Labels, version 2.1:** version 2 plus the dry-bed fix. Plants on a low-water bed, away from open water, become land also when the photo shows no open water at all (`labels.py:113`). The labels are made once on a CPU and saved, and both GPU parts read the same saved labels.
- **Starting weights:** run 6 (Bolivia 66.0), with seed s starting from run 6's seed s, and a new last layer for the three classes.
- **Flood chips in every batch:** a quarter of each batch (4 of 16 crops) comes from Sen1Floods11's hand-labelled training split. Expert water counts as water of either kind, and the rest as land. Bolivia and the test split are never used for training.
- **Two modes, each with 3 seeds averaged:**
  - **"rule":** labels version 2.1 on clear days, cloudy days and radar passes, as in run 2. Photo labels are moved to radar passes within 5 days and 1% of capacity, and the rule alone labels the rest. The Sri Lankan part of each batch is 40% clear days, 10% cloudy days and 50% radar passes.
  - **"photo"** (the fair control, no storage at all): photo labels on clear days and on every radar pass within 5 days of a clear photo, with no storage check. Its Sri Lankan part is 40% clear days and 60% of those radar passes.
- **Sampling:** pick a reservoir at random first, then a chip of the chosen kind from it, so that the big reservoirs don't dominate.
- **Training** is otherwise as run 2: 10 epochs of 300 batches of 16 crops, lr 3e-4, the last epoch kept.
- **Comparison methods:** run 2 (r2), run 6 (r6, 3 seeds), the tuned VV threshold and Otsu. They are not trained again; their maps are taken from run 2's saved areas (same chips, same folds). Run 2's retrained model is compared the same way.
- **Running:**
  - labels on a CPU (about 1 hour);
  - GPU part A, folds 1 to 3 (about 5 hours);
  - GPU part B, folds 4 and 5 (about 4 hours).

  Each part writes its tables after every fold. A part that fails is run again unchanged (same seeds). The parts are merged and scored on the laptop with the same scoring code as run 2.
- **No downloads during the run:** the Sen1Floods11 chips, Bolivia's among them, come in a private Kaggle dataset.

## Primary measure

The storage error on the later half of each held-out reservoir's passes, read with a power law fitted on its first half (% of capacity), as in runs 1 and 2. Paired Wilcoxon tests over reservoirs.

## The new model wins only if all five hold

1. **It beats the radar U-Nets:** its storage error is lower than run 2's and run 6's, each with p < 0.05 and a median gain of at least 1 point of capacity.
2. **It beats the radar thresholds:** lower than the tuned threshold's and Otsu's, each with p < 0.05 and a median gain of at least 1 point.
3. **It beats the seasonal baseline:** p < 0.05 and a median gain of at least 1 point.
4. **Less over-reading at low water:** on held-out clear days under 20% of capacity, take |map's water share − photo's share| on pixels where the photo is sure. Its per-reservoir median gap is lower than run 6's (p < 0.05).
5. **No loss on expert labels:** its water IoU on Sen1Floods11's Bolivia chips is at most 2 points below run 6's, the model it starts from (66.0, so at least 64.0; mean over folds).

**Separate claim, "the storage rule helps":** made only if the new model's storage error is lower than the fair control's (p < 0.05, median gain of at least 0.5 point). If it is not, any gain is credited to training on Sri Lankan chips, not to the rule.

## Reported, but not part of the win

- the comparison with run 2's retrained model (does the flood mix cost storage skill?);
- Spearman, and the error near full, both signed and as a size;
- the errors in million m3 as well as % of capacity;
- results by size, the label coverage, and water IoU on Sen1Floods11's test split;
- the "last clear photo" baseline, scored afterwards with `scripts/score_last_clear.py`.

## If it does not win

Report it as it is. In that case:
- run 2 stays the best storage reader;
- run 6 stays the best general water mapper;
- the write-up says which one a user should pick for which job.

## Known limits

- The training dates overlap the test dates. A date-split check (train before 2024, test after) is still to do.
- The labels inherit the photo's mistakes.
- Near full, the area hardly grows, so storage there needs SWOT's height (Module B).
