# Retraining run 5: results (5 Nov 2026)

**Rules:** `PROTOCOL-5.md`, fixed on 5 Nov 2026 after an outside review and before any run 5 training (commit 055ef48, kept on the branch `full-history` since the history was compacted). The check is `scripts/score_run5.py`. Its outputs are in `runs/2026-11-05-retrain-5/`, and `protocol5_check.csv` holds the verdict. `beside_run4.csv` shows run 4 scored the same way.

**The one change from run 4:** the batch mix, 10 reservoir, 4 Sen1Floods11 and 2 S1S2-Water crops in every batch of 16 (run 4: 8, 4 and 4; the control: 12, 4 and 0). Everything else is run 4's.

**How to read it.** Run 5's change was chosen after run 4's test scores, and it is the last run in the series. As PROTOCOL-5 says:
- run 4 is the confirmatory test of the false-water fix;
- run 5 is a secondary, exploratory test of the trade-off between the gain abroad and the small losses at home;
- runs 4 and 5 were tested on the same data, so this is a repeated test. The bars and alpha were not changed for it.

**Where it ran:** Kaggle, three notebooks run from a teammate's account, each on two Tesla T4s. No session was cut off, and no seed was loaded from an earlier session.

| Part | Folds | Minutes in all | Training minutes per fold | Rule 1 scoring |
|---|---|---|---|---|
| 1 | 1 and 2, at the same time | 67 | 55 and 56 | 12 |
| 2 | 3 and 4, at the same time | 68 | 55 and 55 | 12 |
| 3 | 5 | 53 | 43 | 10 |

Parts 2 and 3 ran at the same time. Before scoring, the script checked that each fold was found once with its 3 weights, flood scores, pass areas and Dynamic World counts, and that each of the 20 S1S2-Water scenes had exactly 5 run 5 rows (folds 1 to 5). Run on run 4's outputs, the same script gives back run 4's numbers and verdict.

## Verdict by PROTOCOL-5: a win (all four rules hold)

| Rule | Run 5 | Control | Bar | Holds? | Run 4 |
|---|---|---|---|---|---|
| 1a. S1S2-Water pooled IoU | **84.63** | 65.27 | 68.3 | yes | 85.60 |
| 1b. median scene IoU | **86.33** | 73.60 | 73.6 | yes | 87.88 |
| 1c. better at 8 or more of 15 scenes | 13 of 15 (median gain +5.4) | | 8 | yes | 13 of 15 |
| 1d. one-sided Wilcoxon | p = 0.001 | | 0.05 | yes | p = 0.0008 |
| 2a. Dynamic World pooled IoU (931 chips) | 91.48 | 91.75 | 91.25 | yes, by 0.23 | 91.34 |
| 2b. median reservoir IoU | 89.48 | 90.35 | 89.35 | yes, by 0.13 | 89.12 (no) |
| 2c. low-water IoU | 86.17 | 85.98 | 84.98 | yes | 86.11 |
| 3a. median storage error (% of capacity) | 8.65 | 8.28 | 8.78 | yes | 8.69 |
| 3b. median paired difference | +0.04 | | +0.25 | yes | +0.16 |
| 3c. non-inferior within 0.5 (one-sided Wilcoxon) | p = 0.004 | | 0.05 | yes | p = 0.091 (no) |
| 4. Sen1Floods11 test split | 65.67 | 67.6 | 65.6 | yes, by 0.07 | 65.55 (no) |

All four rules hold, so by PROTOCOL-5 **run 5 becomes Module A's model**, with the caveat above: the evidence for it is weaker than a fresh test would give. Three of the checks pass by small margins (2a by 0.23, 2b by 0.13 and 4 by 0.07).

**Reported beside the rules (not rules):**
- Rule 1, Hodges-Lehmann gain over the control: **+11.0 IoU points (95% CI +2.4 to +19.8)**; run 4: +11.9 (+2.9 to +21.0).
- Rule 3, Hodges-Lehmann difference in storage error (run 5 minus the control): **+0.16 points of capacity (95% CI -0.05 to +0.40)**, inside the 0.5 margin; run 4: +0.31 (+0.05 to +0.60), whose upper end passed the margin.
- Bolivia: 41.9 (control 48.5, run 4 44.4). See rule 4.

## Rule 1: S1S2-Water (15 expert-labelled test scenes)

| | Run 5 | Run 4 | Control |
|---|---|---|---|
| Pooled IoU (mean of 5 folds) | **84.6** | 85.6 | 65.3 |
| Median scene IoU | **86.3** | 87.9 | 73.6 |
| Precision | **88.4** | 89.3 | 66.1 |
| Recall | 95.2 | 95.4 | 98.2 |
| 5 Asian extra scenes, pooled IoU | 94.3 | 94.9 | 93.5 |

Halving S1S2-Water's share kept almost all of run 4's gain: pooled IoU 84.6 against 85.6, precision 88 against 89.

**Per scene** (IoU, mean of 5 folds; the range is over the folds):

| Scene | Land cover | Run 5 | Run 5 range | Run 4 | Control |
|---|---|---|---|---|---|
| 23 | Shrubland | 30.5 | 29.4 to 32.1 | 33.4 | 2.3 |
| 25 | Tree cover, broadleaved, deciduous | 77.5 | 75.8 to 79.0 | 78.4 | 75.2 |
| 29 | Herbaceous cover | 73.4 | 71.8 to 75.1 | 74.3 | 68.0 |
| 33 | Tree cover, broadleaved, deciduous | 89.3 | 89.2 to 89.5 | 89.3 | 88.4 |
| 35 | Tree cover, needleleaved, evergreen | 94.0 | 93.3 to 94.4 | 94.1 | 61.5 |
| 47 | Cropland, rainfed | 86.3 | 86.2 to 86.4 | 86.3 | 86.8 |
| 53 | Herbaceous cover | 89.8 | 87.0 to 91.5 | 91.5 | 67.8 |
| 57 | Water bodies | 97.1 | 96.9 to 97.1 | 97.1 | 96.1 |
| 75 | Cropland, rainfed | 90.0 | 89.1 to 90.6 | 91.1 | 72.3 |
| 77 | Tree cover, needleleaved, deciduous | 76.5 | 76.2 to 76.6 | 76.5 | 74.2 |
| 78 | Grassland | 56.5 | 45.2 to 66.1 | 56.6 | 15.3 |
| 80 | Shrub or herbaceous cover, flooded | 90.2 | 89.7 to 90.5 | 90.2 | 87.3 |
| 82 | Tree cover, broadleaved, evergreen | 92.6 | 84.8 to 98.0 | 92.5 | 98.5 |
| 88 | Bare areas | 56.0 | 55.4 to 57.1 | 57.5 | 39.0 |
| 89 | Water bodies | 86.1 | 75.0 to 90.4 | 87.9 | 73.6 |

Run 5 is within 3 points of run 4 at every scene. As in run 4, it is worse than the control at scenes 47 (by 0.5) and 82 (by 6). Scenes 23, 78 and 88 are still the weakest.

**Caution in reading rule 1** (as for run 4): run 5 trained on S1S2-Water's training split and the control did not. No test scene is in training (the nearest training scene is at least 223 km away), but run 5 has learned that dataset's labelling convention, so part of the gain is that familiarity.

## Rule 2: Dynamic World on our reservoirs (run 4's 931 chips, as fixed in PROTOCOL-5)

| Group | Chips | Run 5 IoU | Control IoU | Run 5 median reservoir | Control median reservoir |
|---|---|---|---|---|---|
| All, 103 reservoirs | 931 | 91.5 | 91.8 | 89.5 | 90.4 |
| Low water (under 20% full), 30 reservoirs | 63 | 86.2 | 86.0 | 83.6 | 83.3 |

- Precision 99.3 against 99.4; recall 92.1 against 92.3.
- Run 5 is better at 22% of the 103 reservoirs (run 4: 17%), with a median gain of -0.55 (run 4: -0.65).
- Run 5 used exactly run 4's 931 chips, so no deviation was needed this time.

## Rule 3: storage error on the later half (103 reservoirs, 14,831 radar passes)

| Method | Median error (% of capacity) | Spearman with storage (all / clear / cloudy) | Near full (median, points) |
|---|---|---|---|
| Run 5 | 8.65 | 0.86 / 0.87 / 0.85 | -7.5 |
| Control | 8.28 | 0.87 / 0.86 / 0.86 | -6.6 |
| Seasonal baseline | 14.4 | | |
| No model | 22.5 | | |

- Run 5 is better at 50% of the reservoirs (run 4: 43%), and the median paired difference is +0.04 points (run 4: +0.16).
- Non-inferiority within 0.5 points is now shown (p = 0.004; run 4: p = 0.09).
- Every pass had a control value, and none was left out.

## Rule 4: flood mapping (water IoU, %)

| Fold | Test split, run 5 | Test split, run 4 | Test split, control | Bolivia, run 5 | Bolivia, run 4 | Bolivia, control |
|---|---|---|---|---|---|---|
| 1 | 65.9 | 65.5 | 66.6 | 40.3 | 44.5 | 48.9 |
| 2 | 65.5 | 65.4 | 68.6 | 38.0 | 46.6 | 46.4 |
| 3 | 66.0 | 65.5 | 66.5 | 43.4 | 42.4 | 43.0 |
| 4 | 65.4 | 65.9 | 68.0 | 44.6 | 42.9 | 51.2 |
| 5 | 65.6 | 65.4 | 68.3 | 43.3 | 45.6 | 52.8 |
| **Mean** | **65.67** | **65.55** | **67.6** | **41.9** | **44.4** | **48.5** |

- The test split passes by 0.07 points. It is still 1.9 points below the control.
- **Bolivia (reported, not a rule) is lower than in run 4:** 41.9 against 44.4, and 6.6 points below the control. Bolivia is a separate flood region held out of Sen1Floods11's training, and folds 1 and 2 score lowest there (40 and 38). Run 5 should not be described as a better flood mapper than the control.

## What it means

1. **Run 5 meets all four rules,** so it becomes Module A's model under PROTOCOL-5. The report must say that it is exploratory: its change was chosen after run 4's scores, on the same test data.
2. **The trade-off moved as intended.** Returning most of the reservoir share kept the gain abroad (S1S2-Water pooled IoU 84.6 against the control's 65.3; precision 88 against 66) and recovered enough at home:
   - storage: non-inferiority now shown (p = 0.004), with a Hodges-Lehmann difference of +0.16 points (95% CI -0.05 to +0.40);
   - Dynamic World: the median reservoir IoU passes by 0.13;
   - floods: the test split passes by 0.07.
3. **The margins at home are thin.** Run 5 is still slightly below the control on our reservoirs (Dynamic World 89.5 against 90.4 median; storage error 8.65% against 8.28%), within the allowed limits but not better.
4. **Bolivia is the clearest cost:** 6.6 points below the control. It is not a rule, but it should be reported beside the win.
5. **The result is reproducible across machines and parts:** the five folds' Sen1Floods11 test scores agree within 0.7 points, and the part runner's two-folds-at-once setup gave clean, complete outputs.

## Next steps (the team chooses)

1. **Adopt run 5 as Module A's model** and report it with the exploratory caveat, the Hodges-Lehmann CIs and the Bolivia cost; update the supervisor note in formal register.
2. **Publish the result** by pull request (the Mehara workspace and IS4990-project progress notes), with no code.
3. **Optional, offered before:** an all-data deployment model (one model trained on all 103 reservoirs, about 1 hour on Kaggle). Its scores could not be tested, because no reservoir would be held out; the five fold models above remain the evidence.
