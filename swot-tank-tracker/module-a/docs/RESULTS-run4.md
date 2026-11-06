# Retraining run 4: results (5 Nov 2026)

**Rules:** fixed before any training data was prepared, in `PROTOCOL-4.md` (commit 1c4207d). The check is `scripts/score_run4.py`. Its outputs are in `runs/2026-11-04-retrain-4/`, and `protocol4_check.csv` holds the verdict. Rule 1's scene scores are in `runs/2026-11-05-run4-s1s2/`.

**Aim:** cut the false water on dark, dry land, the main weakness S1S2-Water found in run 3 and its photo-label control (precision 60 to 66%), without losing what we had gained on our reservoirs.

**What changed from run 3's photo-label control** (all together, so a win would show that the package works, not which part did it):
- photo labels plus never-water land (`land_out`);
- 29 S1S2-Water training scenes (1,856 patches of 512 × 512);
- a batch mix of 8 reservoir, 4 Sen1Floods11 and 4 S1S2-Water crops;
- dry beds counted twice;
- the crop positions fixed.

**Where it ran** (one fold per session, 3 seeds each; every GPU was a Tesla T4):

| Fold | Where | Minutes | Note |
|---|---|---|---|
| 1 | Colab, main account | 41 | seed 0 was loaded from an earlier, cut-off session |
| 2 | Kaggle | 44 | |
| 3 | Kaggle | 40 | |
| 4 | Colab, second account | 20 | seeds 0 and 1 were loaded from an earlier, cut-off session |
| 5 | Colab, second account | 40 | |
| Rule 1 scoring | Colab, second account | 41 | no training |

The minutes are those of the last session only. The same code and seeds run on every machine (run 3's fold 4 gave the same results on Kaggle and on Colab). A seed loaded after a cut-off session is that seed's finished weights file, so no seed was trained twice.

## Verdict by PROTOCOL-4: not a win

| Rule | Run 4 | Control | Bar | Holds? |
|---|---|---|---|---|
| 1a. S1S2-Water pooled IoU | **85.60** | 65.27 | 68.3 | yes |
| 1b. median scene IoU | **87.88** | 73.60 | 73.6 | yes |
| 1c. better at 8 or more of 15 scenes | 13 of 15 (median gain +6.3) | | 8 | yes |
| 1d. one-sided Wilcoxon | p = 0.0008 | | 0.05 | yes |
| 2a. Dynamic World pooled IoU | 91.34 | 91.75 | 91.25 | yes |
| 2b. median reservoir IoU | 89.12 | 90.35 | 89.35 | **no, by 0.2** |
| 2c. low-water IoU | 86.11 | 85.98 | 84.98 | yes |
| 3a. median storage error (% of capacity) | 8.69 | 8.28 | 8.78 | yes |
| 3b. median paired difference | +0.16 | | +0.25 | yes |
| 3c. non-inferior within 0.5 (one-sided Wilcoxon) | p = 0.091 | | 0.05 | **no** |
| 4. Sen1Floods11 test split | 65.55 | 67.6 | 65.6 | **no, by 0.05** |

Rule 1, the aim of the run, holds by a wide margin. Three of the "no loss" checks fail, each by a small amount. Under PROTOCOL-4 all four rules must hold, so **run 4 does not replace the control, and run 3 stays Module A's model.**

## Rule 1: S1S2-Water (15 expert-labelled test scenes)

| | Run 4 | Control | Run 3 |
|---|---|---|---|
| Pooled IoU (mean of 5 folds) | **85.6** (folds 84.6 to 86.9) | 65.3 (63.7 to 66.6) | 59.6 |
| Median scene IoU | **87.9** | 73.6 | 71.1 |
| Precision | **89.3** | 66.1 | 60.1 |
| Recall | 95.4 | 98.2 | 98.5 |
| 5 Asian extra scenes, pooled IoU | 94.9 | 93.5 | 92.3 |

The run removed most of the false water: precision rose from 66 to 89 for a loss of 3 points of recall.

**Per scene** (IoU, mean of 5 folds):

| Scene | Land cover | Run 4 | Control |
|---|---|---|---|
| 23 | Shrubland | 33.4 | 2.3 |
| 25 | Tree cover, broadleaved, deciduous | 78.4 | 75.2 |
| 29 | Herbaceous cover | 74.3 | 68.0 |
| 33 | Tree cover, broadleaved, deciduous | 89.3 | 88.4 |
| 35 | Tree cover, needleleaved, evergreen | 94.1 | 61.5 |
| 47 | Cropland, rainfed | 86.3 | 86.8 |
| 53 | Herbaceous cover | 91.5 | 67.8 |
| 57 | Water bodies | 97.1 | 96.1 |
| 75 | Cropland, rainfed | 91.1 | 72.3 |
| 77 | Tree cover, needleleaved, deciduous | 76.5 | 74.2 |
| 78 | Grassland | 56.6 | 15.3 |
| 80 | Shrub or herbaceous cover, flooded | 90.2 | 87.3 |
| 82 | Tree cover, broadleaved, evergreen | 92.5 | 98.5 |
| 88 | Bare areas | 57.5 | 39.0 |
| 89 | Water bodies | 87.9 | 73.6 |

The largest gains (18 to 41 points, at scenes 23, 35, 53, 75, 78 and 88) are mostly at the scenes where the control scored lowest. Run 4 is worse at two scenes, 47 (by 0.5) and 82 (by 6). Scenes 23, 78 and 88 are still the weakest. Scenes 78 and 82 vary most between folds (40 to 68 and 84 to 97).

**Caution in reading rule 1.** Run 4 trained on S1S2-Water's training split, and the control did not.
- The test scenes are not in the training set: no scene is shared, and the nearest training scene is at least 223 km from every test scene.
- But run 4 has learned that dataset's labelling convention and its kind of scene, so part of the gain is that familiarity.
- The 5 Asian scenes were held out of training on purpose. There the gain is smaller (94.9 against 93.5), but the control already scored high on them.

## Rule 2: Dynamic World on our reservoirs

| Group | Chips | Run 4 IoU | Control IoU | Run 4 median reservoir | Control median reservoir |
|---|---|---|---|---|---|
| All, 103 reservoirs | 931 | 91.3 | 91.8 | 89.1 | 90.4 |
| Low water (under 20% full), 30 reservoirs | 63 | 86.1 | 86.0 | 83.3 | 83.3 |

- Precision 99.3 against 99.4; recall 91.9 against 92.3.
- Run 4 is better at 17% of the 103 reservoirs (median gain -0.65).
- On our tanks, then, run 4 gives slightly less water than the control at most reservoirs. There was almost no false water at home to remove (precision about 99%), so the change shows only as a small loss of recall.

**Deviation (found after scoring).**
- Run 4's notebook scored only the chips on run 3's chip lists: 931 of the photo check's 1,004 chips, all at the same 103 reservoirs. The 73 missing chips are clear days without a storage record within 2 days.
- The protocol's control numbers come from all 1,004 chips. A paired comparison needs the same chips, so the control was scored again on the 931.
- The control's pooled IoU hardly moves (91.72 on 1,004, 91.75 on 931). The bars were moved with it, by the same offsets the protocol set: minus 0.5 for pooled and minus 1 for the two medians.
- Rule 2b fails either way: against the protocol's own bar of 89.4 (90.4 minus 1), run 4's 89.1 is also below.

## Rule 3: storage error on the later half (103 reservoirs, 14,831 radar passes)

| Method | Median error (% of capacity) | Spearman with storage (all / clear / cloudy) | Near full (median, points) |
|---|---|---|---|
| Run 4 | 8.7 | 0.86 / 0.86 / 0.85 | -7.8 |
| Control | 8.3 | 0.87 / 0.86 / 0.86 | -6.6 |
| Seasonal baseline | 14.4 | | |
| No model | 22.5 | | |

- Run 4 is better at 43% of the reservoirs. The median paired difference is +0.16 points, within the 0.25 allowed.
- The non-inferiority test fails (p = 0.09): the data do not show that run 4 is within 0.5 points of the control.
- Every pass had a control value, and none was left out.

## Rule 4: flood mapping (water IoU, %)

| Fold | Sen1Floods11 test split, run 4 | Test split, control | Bolivia, run 4 | Bolivia, control |
|---|---|---|---|---|
| 1 | 65.5 | 66.6 | 44.5 | 48.9 |
| 2 | 65.4 | 68.6 | 46.6 | 46.4 |
| 3 | 65.5 | 66.5 | 42.4 | 43.0 |
| 4 | 65.9 | 68.0 | 42.9 | 51.2 |
| 5 | 65.4 | 68.3 | 45.6 | 52.8 |
| **Mean** | **65.55** | **67.6** | **44.4** | **48.5** |

- The flood share of every batch was the same as in run 3 (a quarter), but our reservoirs' share fell by a third to make room for S1S2-Water. The flood score fell by 2.05 points, against an allowed 2.
- Bolivia (reported, not a rule) fell by 4 points.

## What it means

1. **The false-water fix works where it was aimed.** On S1S2-Water, run 4 is better than the control at 13 of 15 scenes, and its pooled IoU rises by 20 points, mainly through precision (66 to 89). Part of this gain may come from training on the same dataset's other scenes (see the caution under rule 1).
2. **On our own reservoirs it costs a little.**
   - The median reservoir IoU against Dynamic World is 1.2 points lower.
   - The median storage error is 0.4 points of capacity higher, and non-inferiority is not shown.
   - Our tanks had almost no false water to begin with (precision about 99%), so there was nothing for the fix to gain at home.
3. **On floods it costs 2 points**, just past the limit, and Bolivia drops by 4 points.
4. **By the protocol, run 4 is not a win.** Run 3 stays the model for storage. The S1S2-Water result is still useful: it shows that the false-water weakness can be trained away when labelled scenes of that kind are added.
5. **The result is reproducible across machines.** Three folds ran on Colab and two on Kaggle (all Tesla T4), and the five folds' Sen1Floods11 scores agree within 0.5 points.

## Options for next

1. **Report run 4 as it is:** a large gain on the benchmark abroad, a small loss at home, and not a win by the fixed rules. Run 3 stays the model.
2. **Use run 4 only where it helps:** as the model for scenes unlike our tanks (dry, open land), with run 3 kept for storage. This would need its own rules, fixed in advance, before it is claimed.
3. **A run 5 (only if worth it), with rules fixed first:** keep the reservoir share at run 3's level, for example a batch of 10 reservoir, 3 flood and 3 S1S2-Water crops, so that the gain abroad costs less at home.
