# Run 5's final model: plan (fixed 5 Nov 2026, before it is trained)

**Why.** Run 5 met all four rules of `PROTOCOL-5.md` with five fold models, each trained on four fifths of the 103 reservoirs and tested on the rest (`RESULTS-run5.md`). For use, one model is trained the same way on all 103 reservoirs. The five-fold results stay the evidence for the recipe. The final model is not a new test, and nothing in `RESULTS-run5.md` changes.

**What.**
- Run 5's code, unchanged: the private dataset `is4990-module-a-run5-code` (version 2), run as `run5_part.py --folds 6`.
- Fold index 5 holds no reservoir out (the 103 reservoirs have fold indexes 0 to 4), so `retrain5.py` trains on all 103 with everything else as in run 5:
  - the same labels, inputs and batch mix (10 reservoir, 4 Sen1Floods11 and 2 S1S2-Water crops in every batch of 16);
  - 10 epochs of 300 batches, learning rate 3e-4, the last epoch kept;
  - 3 seeds (500, 501 and 502), started from run 6's seeds 0, 1 and 2.
- The model is the 3-seed ensemble (water where the mean p(land) is below 0.5), as each fold's model was. Its weights are `fold5_photo5_seed0.pt` to `seed2.pt`.
- A local CPU dry run of exactly this command passed before the notebook was pushed.

**Where.** The private Kaggle notebook `is4990-module-a-run5-final`, run by a teammate from their own account, as run 5's parts were. With two T4s, training uses one GPU (about 35 to 45 minutes). Scoring S1S2-Water's 20 scenes is then shared between both GPUs (about 10 minutes).

**Checks.** These look for errors, not a new test, and are fixed now. `scripts/check_final5.py` runs them.
1. The run ends with no problems: 3 weights, and a log showing 103 reservoirs trained and none held out.
2. Sen1Floods11 test split IoU between **63.39 and 68.04**: the five folds' range (65.39 to 66.04), widened by 2 points each side.
3. S1S2-Water pooled IoU on the 15 test scenes between **80.85 and 87.50**: the five folds' range (82.85 to 85.50), widened by 2 points each side.

If a check fails, the cause is found before the model is used. The recipe is not tuned. Reported but not checked: Bolivia, precision and recall, and the 5 Asian scenes.

**Not checked:** Dynamic World and storage at the 103 reservoirs. The final model trains on all of them, so those scores would be in-sample.

**How it is used.**
- **New radar passes, and reservoirs or tanks outside the 103:** the final model.
- **Anything tested against storage records at the 103 reservoirs** (for example, Module B's tests): run 5's out-of-fold areas (`runs/2026-11-05-retrain-5/part*/fold*/areas_passes/`, 14,831 passes). There, each reservoir's areas come from a model that never trained on it.

## Result (5 Nov 2026, after the run): passes its checks

Run by a teammate on Kaggle (Tesla T4), 41 minutes in all; `runs/2026-11-05-run5-final/` (weights git-ignored), checked by `scripts/check_final5.py` (`final_check.csv`).

| Check | Final model | Allowed | Run 5 five-fold mean | Holds? |
|---|---|---|---|---|
| 1. No problems; 3 weights; 103 reservoirs trained, none held out | as stated | | | yes |
| 2. Sen1Floods11 test split IoU | 64.79 | 63.39 to 68.04 | 65.67 | yes |
| 3. S1S2-Water pooled IoU (15 test scenes; all 20 scored) | 87.13 | 80.85 to 87.50 | 84.63 | yes |

Reported: Bolivia 42.97 (five-fold mean 41.91); S1S2-Water precision 90.92 and recall 95.44 (88.41 and 95.21); median scene IoU 89.03 (86.33); the 5 Asian scenes 95.11 (94.28).

- Both values lie a little outside the five folds' own ranges (Sen1Floods11 0.6 below the lowest fold, S1S2-Water 1.6 above the highest), inside the allowed bands. A model trained on a quarter more reservoirs can differ this much; no error was found.
- These are error checks, not tests. In particular, the final model's 64.79 is not a rule 4 result: PROTOCOL-5's rules were met by the five fold models, and they stay the evidence.
- The weights (`fold5_photo5_seed0.pt` to `seed2.pt`) are added to the private dataset `is4990-module-a-run5-weights` beside the 15 fold models.
