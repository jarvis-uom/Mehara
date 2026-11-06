# Protocol for retraining run 5 (FIXED 5 Nov 2026, before any run 5 training)

**Status:** fixed on 5 Nov 2026 after an outside review. The team leader approved it. No run 5 training had started. Nothing below changes, except by a recorded amendment made before any score is seen.

**Aim.** Keep most of run 4's cut in false water on dry land (S1S2-Water pooled IoU 85.6, against 65.3 for run 3's photo-label control), while passing the "no loss" rules that run 4 missed by small margins (`RESULTS-run4.md`):
- median reservoir IoU against Dynamic World: 89.12, where 89.35 was needed;
- storage non-inferiority: p = 0.09;
- the Sen1Floods11 test split: 65.55, where 65.6 was needed.

**The comparison** is again with **run 3's photo-label control**, with the same bars as PROTOCOL-4. Run 4 is reported beside it.

**This choice was made after seeing run 4's test scores.** Run 5 is therefore not an independent test: its change was picked because run 4 lost a little at home. To limit tuning on the test sets:
- run 5 is **the last run in this series**. If it fails, run 3 stays Module A's model, and runs 4 and 5 are reported as they are;
- it makes **one change** from run 4, chosen in advance (below), with no further tuning;
- the results report this history.

**How the result is reported.**
- **Run 4 is the confirmatory test** of the false-water fix: its rule 1 held under a protocol fixed before any data was prepared.
- **Run 5 is a secondary, exploratory test** of the trade-off between the gain abroad and the small losses at home. If it meets all four rules, it becomes Module A's model, and the report says that the evidence for it is weaker than a fresh test would give.
- Bolivia and the 5 Asian S1S2-Water scenes did not drive the choice of the change, and they are reported as checks. They are not untouched: both were reported for run 4.

## The one change from run 4: the batch mix

| | Our reservoirs | Sen1Floods11 | S1S2-Water |
|---|---|---|---|
| Run 3 control (batch of 16) | 12 | 4 | 0 |
| Run 4 | 8 | 4 | 4 |
| **Run 5** | **10** | **4** | **2** |

- **Batch positions:** with `i % 8`, positions 0 and 4 are Sen1Floods11, position 2 is S1S2-Water, and the other 5 are our reservoirs. That gives 4 + 2 + 10 in every batch of 16, since the loader does not shuffle.
- **Why this mix:**
  - our reservoirs' share returns most of the way to the control's (10 of 16 against 12);
  - the flood share stays at a quarter;
  - S1S2-Water's share is halved. Run 4 passed rule 1 with a wide margin (85.6, where 68.3 was needed), so some of that gain can be given back.

**Everything else is exactly as in run 4** (PROTOCOL-4 items 1, 2, 4 and 5):
- photo labels plus never-water land;
- the same 29 S1S2-Water training scenes and 1,856 patches;
- dry beds counting twice;
- the fixed crop positions;
- the start (run 6), the 103 reservoirs and five folds, 3 seeds, 10 epochs of 300 batches, learning rate 3e-4, the last epoch kept;
- the same datasets (`is4990-module-a-run4-inputs`, `is4990-s1s2water-train`), and run 4's code with only the batch positions changed.

## Win rules (all four must hold)

These are PROTOCOL-4's rules and bars. The tests are one-sided paired Wilcoxon tests at p < 0.05, on unrounded values.
- **Within run 5**, all four rules must hold, so the chance of a false "win" is at most 0.05, with no correction (an intersection-union test).
- **Across runs 4 and 5**, the same test data are tested twice. This repeated testing is stated in the report. The bars and alpha are not changed for it.

1. **Less false water abroad.** On S1S2-Water's 15 test scenes, all of:
   - pooled IoU at least **68.3**;
   - median scene IoU at least **73.6**;
   - better than the control at **8 or more of the 15 scenes** (run 5 and the control are each the mean of 5 folds);
   - p < 0.05.
2. **No loss on our reservoirs, pixels.** Against Dynamic World, **on the 931 chips that run 4 used** (the Dynamic World chips on run 3's chip lists, all 103 reservoirs; this fixes in advance the deviation found in run 4). Run 3's chip lists keep only clear chips with a storage record within 2 days, which is why 73 of the photo check's 1,004 chips are not among them. The control's values on the 931 chips are 91.75, 90.35 and 85.98. Run 5 needs:
   - pooled IoU at least **91.25**;
   - median reservoir IoU at least **89.35**;
   - low-water IoU at least **84.98**.
3. **No loss on our reservoirs, storage** (later half, 103 reservoirs, every radar pass). With d = run 5's error minus the control's for each reservoir, all of:
   - the group median error is at most **8.78%** (the control's 8.28% plus 0.5);
   - the median d is at most **+0.25** points of capacity;
   - non-inferiority within 0.5 points: one-sided Wilcoxon on (the control's error + 0.5 − run 5's error) > 0, p < 0.05.
4. **No loss on floods in general.** On the Sen1Floods11 test split (90 chips), mean IoU of the 5 folds at least **65.6**.

**Reported, not rules:**
- the Hodges-Lehmann estimate and 95% confidence interval of the paired differences for rule 1 (15 scenes) and rule 3 (103 reservoirs), beside the p-values;
- run 4 beside run 5 on every rule, to show what halving S1S2-Water's share did;
- Bolivia (15 chips);
- the 5 Asian S1S2-Water scenes;
- precision and recall;
- the GPU each fold ran on.

## How it runs

- **Code:** the private dataset `is4990-module-a-run5-code` holds four files:
  - `retrain5.py` (from `kernel5/`) is run 4's `retrain4.py` with only the batch positions, the run's name and the weights' name changed (shown by a diff), plus a check that the batch is a multiple of 8;
  - `run5_part.py` runs one part (below). It installs any missing package once, before the folds start, so that two folds starting together never install at the same moment;
  - `score_s1s2_run5.py` scores a part's folds on S1S2-Water;
  - `score_s1s2.py` is the S1S2-Water check's own code, unchanged.

  A CPU dry run passed on 5 Nov, and every batch of 16 holds exactly 10 + 4 + 2.
- **Data:** each notebook attaches these eight datasets, run 4's with no change plus the code:
  - private: `is4990-module-a-run4-inputs`, `is4990-s1s2water-train`, `is4990-sen1floods11-radar`, `is4990-module-a-dynamic-world`, `is4990-s1s2water-gee` (the 20 scored scenes, for rule 1) and `is4990-module-a-run5-code`;
  - public: `is4990-module-a-first-check-chips` and `is4990-module-a-pass-chips`.
- **GPU:** Kaggle, in three parts (folds 1 and 2, folds 3 and 4, and fold 5), each run as a private notebook by a teammate from their own Kaggle account (shared with them as a collaborator).
  - With Kaggle's two T4s, two folds train at once, one on each GPU; with one GPU they train in turn.
  - Each part then scores its own folds on S1S2-Water's 20 scenes. A fold's scores do not depend on the other folds.
  - About 1 to 2 hours per part, 4 to 5 GPU hours in all.
- **Putting the parts together on the laptop:** each part's output is downloaded to its own folder (`part1/`, `part2/`, `part3/`), never copied over another. Each part writes its own `s1s2/scenes/<scene>.csv` with only its folds, so the script reads all three folders and joins the rows for each scene. Before scoring, it stops unless:
  - each of the 20 scenes has exactly 5 run 5 rows, one for each fold;
  - each fold folder has its 3 weights, its Sen1Floods11 scores, its pass areas and its Dynamic World counts.
- **Scoring:** as run 4's (`scripts/score_run4.py`, adapted for run 5's folders and name, with the Hodges-Lehmann estimates added).
