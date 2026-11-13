# Series 2 development plan and keep rules (DEV-6, fixed 5 Nov 2026, before any development result)

Under PROTOCOL-6 part 1, section B. These runs only **choose** the recipe that part 2 will fix. They are not tests, they give no p-values, and nothing here can be claimed as a result. They use only the development data, never the fresh tests or the old tests.

## Development batch 1: training length, weight averaging, an IoU loss

Code: `kernel6-dev/retrain6.py` (run 5's `retrain5.py` with the changes below) and `run6_dev.py`. A CPU dry run passed on 5 Nov.

| Variant | What changes from run 5 | Steps |
|---|---|---|
| **control** | Nothing in the recipe (S1S2-Water scenes 39, 49, 54, 76 and 83 held out, as in every variant) | 3,000 |
| **long** | 3 times the steps (same OneCycle shape); the scored model is the weights' moving average (EMA, decay 0.999). The last weights are saved beside it but not scored | 9,000 |
| **lovasz** | As long, plus 0.5 x a Lovász hinge on water against land, per crop | 9,000 |

- **The same in every variant:**
  - folds 1 and 2 (42 held-out reservoirs), 3 seeds each, with run 5's seeds and starts;
  - the batch mix 10/4/2;
  - 24 S1S2-Water training scenes;
  - learning rate 3e-4;
  - the scheduler also steps on a batch with no label. This happened 0 times in run 5, so it changes nothing.
- **Cost:** control about 1 hour; long and lovasz about 2.5 to 3 hours each, with two folds at once on Kaggle's two T4s.

## The development scores

They are computed on the laptop, as the mean of folds 1 and 2 where a fold gives a number.

| Code | Score | Data |
|---|---|---|
| M1 | S1S2-Water pooled IoU | The 5 held-out training scenes (64 patches each) and the 5 Asian scenes, pooled over all 10 |
| M2 | Dynamic World IoU: pooled, and the median over reservoirs | The 42 held-out reservoirs' clear chips with radar dated before their later half |
| M3 | Sen1Floods11 IoU | The validation split (89 chips) |
| M4 | Storage error (% of capacity), with today's line fitted on the first quarter and scored on the second | The 42 reservoirs' first-half passes (radar gap at most 5%, storage within 2 days), paired by reservoir |

## Keep rules (against the control, or for lovasz against long)

A variant is **kept** if it gains somewhere and loses nowhere.

**Gain (at least one of):**
- M1 up by 0.5 points or more;
- M2's median up by 0.3 or more;
- M3 up by 1.0 or more;
- M4 down by 0.2 points or more.

**No loss (all of):**
- M1 down by less than 0.5;
- M2 pooled and median each down by less than 0.3;
- M3 down by less than 1.0;
- M4 up by less than 0.2.

**Order:** long is judged against the control. Then lovasz is judged against long. If long is not kept, lovasz is judged against the control, with the same rules.

**If nothing is kept,** part 2 keeps run 5's map recipe, and series 2 changes only the storage method (C1 in PROTOCOL-6).

## Later batches (written here before they run)

- **Batch 2:** terrain inputs (HAND, slope, local incidence angle) or the previous pass as extra channels. Each is judged the same way, against the best kept recipe.
- **The r6 blend for floods** (WiSE-FT, inference only): its weight is chosen on M3 alone, and it must not lose on M1, M2 or M4 by the limits above.

## Batch 1 results (6 Nov 2026): nothing kept

All three notebooks finished on Kaggle (two T4s, both folds each, no problems): control 47 minutes, long 152, lovasz 139. Scored on the laptop by `scripts/score_dev6.py`; per-fold scores in `runs/dev6/dev6_scores.csv`, the comparisons in `runs/dev6/dev6_verdict.csv`. Development scores only: no claim, no p-value.

| Score (mean of folds 1 and 2) | control | long | lovasz |
|---|---|---|---|
| M1 S1S2-Water pooled IoU (10 scenes) | 93.98 | 93.32 | 94.19 |
| (5 held-out training scenes) | 86.44 | 89.80 | 91.21 |
| (5 Asian scenes) | 94.87 | 93.71 | 94.52 |
| M2 Dynamic World pooled IoU | 88.83 | 88.53 | 88.01 |
| M2 median reservoir IoU | 86.76 | 85.29 | 83.27 |
| M3 Sen1Floods11 valid IoU | 66.15 | 67.21 | 68.02 |
| M4 storage error, % of capacity (lower is better) | 7.00 | 6.82 | 8.42 |

- **long against the control:** M1 -0.66, M2 pooled -0.30 (just inside the limit), M2 median -1.47, M3 +1.07, M4 -0.18. A gain on M3, losses on M1 and the M2 median: **not kept**.
- **lovasz against the control** (long was not kept): M1 +0.21, M2 pooled -0.82, M2 median -3.49, M3 +1.88, M4 +1.41. A gain on M3, losses on M2 and M4: **not kept**.
- Beside the rules: M4 paired by reservoir (42), long against the control has a median difference of +0.10 points (better at 18), lovasz +0.07 (better at 17). The fold-level M4 changes are within the noise between folds (control 8.03 and 5.98).
- What it shows: longer training and the Lovász loss help on floods (M3) and on the held-out S1S2-Water training scenes, but not on the reservoirs (M2, M4), which are what Module A is for. The reservoir chips' median falls in fold 1 in both (85.9 to 82.2 and 80.8).
- **Consequence (as fixed above):** run 5's map recipe stays the incumbent. Batch 2 and the r6 blend are judged against the control; series 2's main lever is now the storage method (C1).

## Error budget E0 (6 Nov 2026): where run 5's storage error comes from

`scripts/error_budget.py`, outputs in `runs/e0/`. Development data only: each reservoir's first half, today's log-log line fitted on the first quarter and scored on the second (as M4), all 103 reservoirs. Diagnostic only; nothing is chosen here. Photo areas: NDWI on every Sentinel-2 date with at least 98% of the outline clear (`out/clear_photos/`, restored from history for this).

| Measure (median over reservoirs, % of capacity) | Value |
|---|---|
| Radar (run 5), first quarter to second | 6.7 |
| Photo (NDWI), the same split | 8.4 |
| Same days only (79 reservoirs): radar / photo | 7.4 / 9.3 |
| Radar, line fitted and scored on the whole first half (in-sample) | 6.1 |
| Radar, storage on the pass day only (3,696 of 3,703 passes have it) | 6.7 |
| Pass-to-pass noise of the line at steady storage | 1.5 |
| Share of second-quarter passes at 85% full or more | 35% |
| Share outside the first quarter's storage range | 12% |

Second-quarter radar error by fill level (pooled passes; signed = the line's reading minus storage):

| Storage | Passes | Error | Signed |
|---|---|---|---|
| 0 to 20% | 325 | 2.5 | +0.8 |
| 20 to 40% | 709 | 3.0 | +0.1 |
| 40 to 60% | 645 | 4.4 | +1.7 |
| 60 to 85% | 839 | 6.4 | +2.0 |
| 85% and over | 1,185 | 10.4 | **-8.9** |

What it shows:
- **The map is not the main limit.** A clear photo's water area gives a worse storage error than run 5's radar, also on the same days. Run 5's noise between passes at steady storage is about 1.5 points.
- **The error is mostly near full, and it is a bias.** A third of the passes are at 85% full or more; there the line reads about 9 points of capacity too low, against about 3 points below 40% full. This is the flat top (finding A1): near full, the area barely grows while storage still rises, and one straight log-log line cannot bend there.
- The 2-day storage matching adds nothing (almost every pass has storage on its own day). Passes outside the fitted range add some error (7.2 against 5.6), but they are 12% of passes.
- **Consequence:** the next gain is in the storage method (C1), aimed at the top of the curve, not in more map training. The candidates are fixed in PROTOCOL-CURVE before they are scored.

## Curve study E1 (6 Nov 2026): the hinged line is chosen

As fixed in `PROTOCOL-CURVE.md` (commit 88f9e76) before any candidate was scored. `scripts/curve_study.py`, outputs in `runs/e1/`. First half, first quarter to second, 103 reservoirs.

| Candidate | Overall (median, % of capacity) | Better than L0 at (of 103) | Near full: error / signed |
|---|---|---|---|
| L0 today's line | 6.68 | - | 10.4 / -8.9 |
| L1 capped | 6.31 | 15 | 9.2 / -8.9 |
| L2 straight line, not in logs | 7.47 | 40 | 9.2 / -8.7 |
| L3 robust (Theil-Sen) | 7.02 | 50 | 9.8 / -9.0 |
| **L4 hinged** | **5.64** | **63** | 8.6 / -7.7 |
| L5 monotone | 6.35 | 58 | 8.2 / -7.3 |
| L4 + S1 (mean over 12 days) | 5.81 | 46 | 9.4 / -9.0 |
| L4 + S2 (median over 24 days) | 6.96 | 37 | 10.3 / -9.8 |

- **Step 1:** L4 is the best, and no simpler candidate is within 0.1 points. One reservoir fell back to L1.
- **Step 2:** smoothing makes it worse, so S0 is kept. Storage changes too fast between passes for a 12- or 24-day window. **C1's storage method is L4 + S0.**
- **Beside the rule:** L4's gain over L0, paired by reservoir, is a Hodges-Lehmann estimate of 0.44 points (95% CI 0.11 to 0.93), one-sided Wilcoxon p = 0.004.
  - PROTOCOL-6 claims a gain on the fresh test only at 0.5 points or more, so on this evidence C1 is likely non-inferior, and a claimed gain is uncertain.
- **The bias near full is only partly removed** (-8.9 to -7.7). Near full the area carries little information about storage; the morning passes (step 3) and other inputs (the season, the previous storage) are what is left to try there.

## Curve study step 3 (7 Nov 2026): the morning passes are not used

As fixed in `PROTOCOL-CURVE.md`'s addendum (commit ad31f5e) before any morning area existed.

**The areas.** Run 5's fold models measured the morning passes on Kaggle CPU (notebook `is4990-module-a-morning-areas`, 6 hours on 2 threads), out of fold: 13,808 passes at 102 reservoirs. Tabbowa has no morning orbit; no chip was skipped. Outputs are in `runs/morning-areas/`. Scored by `scripts/curve_morning.py` on the main orbit's second-quarter passes (103 reservoirs; 100 had first-quarter morning passes).

| Candidate | Overall (median, % of capacity) | Better / worse than O0 (of 100) |
|---|---|---|
| **O0 main orbit only (L4)** | **5.64** | - |
| O1 both orbits in one curve | 6.18 | 41 / 59 |
| O2 as O1, morning areas scaled to the main orbit | 6.01 | 46 / 53 |
| O3 O2's curve plus the latest morning reading | 5.80 | 59 / 41 |

- **The choice is O0:** no candidate is better than O0, so C1 does not use the morning passes. **C1's storage method is L4 on the main orbit, no smoothing.**
- **Beside the rule:**
  - The morning areas are on the same scale as the main orbit's (median factor 1.00), even though the morning radar reads about 1.5 dB darker. Run 5's maps absorb the difference.
  - Read on their own (O2's curve on the morning orbit's second-quarter passes), the morning passes score 5.68, about as well as the main passes.
  - So they hold no extra information for the curve, but they could give a reading every 6 days instead of every 12. That is an observation on development data only; PROTOCOL-6's fresh test scores the main orbit only.

## Outside review of 7 Nov 2026: what held up

Every point was checked against the files and recomputed where needed. Only the conclusions are kept here.
- **Right, acted on:**
  - L4's upper segment could bend down (step 4 below).
  - T1 needs its own truth rule (the pass date, else the mean of the days either side) and no radar gap in VV and VH. Both are in `scripts/score_t1.py`.
- **Right, recorded as limitations of batch 1 (rules fixed, not re-judged):**
  - M1 pools pixels, so Asian scene 11 carries about half of all water pixels. long's whole M1 loss comes from its fold 2 drop on that scene (98 to 93); it was better on 7 of the 10 scenes in fold 2.
  - The M2 median rests on about 3 to 4 chips per reservoir, so it is noisy.
  - Neither changes the practical outcome: neither variant improved storage (M4), and E0 shows the maps matter little for storage.
  - Any later batch should pool M1 per scene, written before it runs.
- **Wording:** "one-SE rule" means a fixed 0.1-point margin, not an estimated standard error (about 0.2 points). No choice changes: L4 led the next curve by 0.7 points.
- **Not supported:** that fitting in logs causes the near-full bias. The straight line without logs (L2, -8.7) and the monotone curve (L5, -7.3) are biased near full too.

## Curve study step 4 (7 Nov 2026): L4c replaces L4

As fixed in `PROTOCOL-CURVE.md`'s step 4 addendum (commit e39d36c) before scoring. `scripts/curve_l4c.py`, outputs `runs/e1/e1_l4c_*.csv` and `runs/e1/l4c-beside/`.

| | L4 | L4c |
|---|---|---|
| Overall (median, % of capacity) | 5.64 | 5.64 |
| Reservoirs with a downhill upper segment | 18 | 1 |
| Near full: error / signed | 8.6 / -7.7 | 7.6 / -5.8 |
| Gain over L0, Hodges-Lehmann (95% CI) | 0.44 (0.11 to 0.93) | 0.32 (0.19 to 0.74) |

- **L4c is chosen** (equal overall score, within the 0.1 margin). Against L4 it is better at 28 reservoirs and worse at 21. It removes the physically impossible curves and a third of the near-full bias. Its gain over L0 is smaller in the median but steadier (a narrower interval with a higher lower end).
- **Beside the rule,** with L4c:
  - smoothing S1 scores 5.55, within 0.1 of S0, so it is still not taken;
  - the morning passes are still not better than O0 (O3 5.75).
- **C1's storage method is L4c on the main orbit, no smoothing.**

## The T1 scorer (7 Nov 2026), tried on development data

`scripts/score_t1.py` implements PROTOCOL-6's T1 and step 1 exactly. The truth is the department's daily sheet whose Update Date is the pass date. On every complete day from 2 to 6 Nov, those sheets equal the history tab (rule 3's source) at 74 of 74 major reservoirs, while their own "Date" column lags for some reservoirs. If that sheet is missing, the scorer uses the mean of the days either side.

- **`--check-truth`:** the sheets agree with `storage_daily.csv` on 464 of the 470 reservoir-days both hold. On the other 6, the history was corrected later (for example 0 on the sheet, 24.6 million m3 in the history). Under PROTOCOL-6 the sheets are the truth; one bad date moves a reservoir's median little.
- **`--dev`:** the first quarter used as history, the second quarter as the window; outputs in `runs/t1/`. 100 reservoirs scored, 3 left out (fewer than 3 dates with no radar gap).

| | Median error, % of capacity |
|---|---|
| I (L0) | 6.71 |
| C1 (L4c) | 5.82 |
| seasonal | 14.75 |
| no model | 20.54 |

- **Step 1 on this trial:** no loss holds (p < 0.001). The gain test has p < 0.001, but the Hodges-Lehmann estimate is 0.33, under 0.5. So C1 would replace I, without a claimed gain. This is a trial of the code, not a result.
- **The real run** (`--window`) needs the window passes measured with `radar_gap_share_both`, after 31 Jan 2027.

## Correction and outside reviews of 9 Nov 2026

Two new reviews (read-only, development data only) were checked by recomputing their main claims on the first half (first quarter fitted, second scored, Curve 2 = L4c).

**Correction to E0 and steps 1 to 4: the near-full "bias" is mostly a selection effect.**
- E0 and the step 1 to 4 tables group passes by their **true** storage. Grouped that way, any reading from area looks low at the top: L4c's signed error is -8.9 at 95% full or more.
- Grouped by the curve's **own reading**, the bias is small:

  | L4c's reading | Signed error (median) | Absolute error (median) | Passes |
  |---|---|---|---|
  | under 60% | -0.5 | 3.3 | 1,708 |
  | 60 to 75% | -0.5 | 7.8 | 616 |
  | 75 to 85% | -2.8 | 9.1 | 451 |
  | 85 to 95% | -2.1 | 6.2 | 460 |
  | 95% and over | +1.6 | 4.0 | 468 |

- **The real problem near full is scatter, not a bias.** The radar area is already 97% of its top size at 85 to 95% full and 98% above 95%. Within a reservoir, the rank correlation of area and storage is 0.92 below 85% full and 0.33 above.
- So the sentences "the line reads about 9 points too low" (E0) and "the bias near full" (steps 1 and 4) describe this selection effect, not a fixable offset.
- L4c's choice stands: it was chosen by the score rule, and a downhill curve is impossible either way.

**Also confirmed:**
- Share of first-half passes at 85% full or more, by month: Nov 27%, Dec 51%, Jan 55%; Jul to Oct 7 to 13%. T1's window is the near-full season.
- Reservoirs often near full (more than 20% of passes at 85% or more; 72 of 103) have a median L4c error of 7.0, against 3.6 for the other 31, and their passes below 85% are also worse (7.0).
- The photo area flattens near full like the radar's, so the flat top comes from the reservoirs, not from the model.

**Rejected:** a rainfall term. PROTOCOL-6 part 1 lists what a pipeline may use: radar, plus the reservoir's history of radar, storage and clear photos. Rain is not on the list.

**Open option (not started):** one season-aware correction near full, using only the date and the reservoir's history, both allowed. It would be laptop only, with a fixed addendum before scoring and a hard stop unless it beats L4c by 0.1. The team has not decided. *(Run on 13 Nov: not kept; see "Curve study step 5" at the end.)*

## Development batch 2 (addendum, fixed 10 Nov 2026, before any run): all fold models together, and flips

**Why.** Model 3's fold models disagree on some scenes abroad: on the reused S1S2-Water test three scenes move by 15 to 20 IoU points between folds (`RESULTS-run5.md`), and on Asian scene 11 one fold scores 92 where the other four score 98. PROTOCOL-6 (T2) allows a map product other than the mean of folds. This batch tries two ways of combining the models at inference. No recipe changes: Model 3's weights, its training and the reservoirs' maps stay as they are.

**The models.** Batch 1's control, trained again by the same code and seeds (`retrain6.py --variant control`, folds 1 and 2, 3 seeds each: 6 models, S1S2-Water scenes 39, 49, 54, 76 and 83 held out). Its weights were deleted in the 9 Nov clean-up. Training on a GPU does not repeat bit for bit, so batch 1's control scores are reported beside the new ones.

**Candidates,** from the simplest:

| Code | Map product for a scene abroad |
|---|---|
| **A0** | Today's: each fold's 3 seeds averaged (mean softmax, water where p(land) < 0.5), one map per fold. A scene's IoU is the mean of the folds' IoUs, as in rule 1 and for I in T2 |
| **A1** | All folds' models averaged into one map (6 models here; 15 in the confirmatory run) |
| **A2** | A1, with each model's probabilities first averaged over the 8 turns and flips of the input (4 quarter turns, each plain and mirrored; each output turned back before averaging) |

**Data (development only, PROTOCOL-6 B):**
- the 10 S1S2-Water development scenes: held-out training scenes 39, 49, 54, 76 and 83 (their 64 patches each) and Asian scenes 6, 7, 11, 12 and 52 (tiled as `score_s1s2.py`);
- Sen1Floods11's validation split (89 chips).

**Scores:**
- **S:** the IoU of each of the 10 scenes. Pixels are not pooled over scenes (the 7 Nov note above).
- **F:** the pooled IoU of the 89 flood chips. For A0, the mean of the two folds.

**Keep rules** (each candidate against A0, on unrounded values):
- **Gain (at least one of):** the mean of the 10 per-scene differences in S is +0.3 points or more; F is up by 1.0 or more.
- **No loss (all of):** no scene is down by 1.0 point or more; F is down by less than 1.0.

**The choice.** Among the kept candidates, A1 is taken unless A2's mean scene IoU is more than 0.1 points above A1's. If only one is kept, it is taken. If none is kept, this batch gives no C2.

**Not scored here: M2 and M4.** The reservoirs keep their own fold's 3 seeds, with no flips. A reservoir's maps must come from models that never trained on it, so A1 cannot apply there. Their maps and areas are A0's under every candidate.

**What a kept candidate means for part 2.**
- C2's map product for T2 and T3 would be Model 3's 15 fold models (weights unchanged), combined as chosen here. As PROTOCOL-6 says, C2's mean of folds is reported beside it.
- On T1, C2's areas equal C1's, so step 2's T1 "no loss" holds by construction and its T1 gain is zero. The report will say so.
- By PROTOCOL-6's fixed order, gains on T2 and T3 are tested only after both T1 gains hold. So C2's differences on T2 and T3 will be reported with their Hodges-Lehmann intervals and not claimed as gains. The "no loss" rules on T2 and T3 decide whether C2 becomes Module A's map product.

**Reported beside the rules (not rules):**
- A0 with flips, per fold;
- precision and recall;
- the flood IoU per event;
- a check that this scorer's A0 counts equal `retrain6.py`'s own per-fold counts on the same scenes;
- batch 1's control scores.

**Limits, stated now.** Development has 2 folds (6 models); the confirmatory product has 15. There are 10 scenes. A scene-level difference under about half a point is within what a retraining changes.

**How it runs.** `kernel6-dev/score_ens6.py` (new), started by `run6_dev.py --ens` after the control's training, in one private Kaggle notebook (about 2 to 3 GPU hours). Scored on the laptop by `scripts/score_dev6b.py`. The 6 weights are kept in a private Kaggle dataset until part 2 is fixed.

## Batch 2 results (10 Nov 2026): nothing kept

The notebook finished on Kaggle (two T4s, 66 minutes, no problems): 49 minutes of training per fold, then 17 for the combining step. Scored on the laptop by `scripts/score_dev6b.py`; the tables are in `runs/dev6b/` (`dev6b_scenes.csv`, `dev6b_floods.csv`, `dev6b_verdict.csv`). Development scores only: no claim, no p-value.

| IoU | A0 (today's) | A1 (all models) | A2 (A1 + flips) | beside: A0 + flips |
|---|---|---|---|---|
| Asian 6 | 83.70 | 83.85 | 83.96 | 83.83 |
| Asian 7 | 75.96 | 76.26 | 76.42 | 76.18 |
| Asian 11 | 97.93 | 97.95 | 97.96 | 97.95 |
| Asian 12 | 93.00 | 93.20 | 93.04 | 92.90 |
| Asian 52 | 94.20 | 94.21 | 94.20 | 94.19 |
| Held-out 39 | 70.90 | 71.28 | 69.48 | 69.57 |
| Held-out 49 | 81.23 | 81.54 | 81.68 | 81.51 |
| Held-out 54 | 80.22 | 80.49 | 80.57 | 80.35 |
| Held-out 76 | 96.26 | 96.28 | 96.30 | 96.29 |
| Held-out 83 | 80.85 | 80.98 | 81.04 | 80.92 |
| Mean of the 10 scenes | 85.42 | 85.60 | 85.46 | 85.37 |
| **F**, floods (89 chips) | 66.15 | 66.23 | 65.66 | 65.75 |

- **A1 against A0:** the mean of the 10 scene differences is +0.18 (median +0.18, lowest +0.01, highest +0.38; up at all 10 scenes); F +0.08. No loss, but no gain either (+0.3 on the scenes or +1.0 on F was needed): **not kept**.
- **A2 against A0:** mean +0.04 (median +0.12, up at 9 of 10), but scene 39 is down by 1.41; F -0.48. A loss on a scene, and no gain: **not kept**.
- **The choice: none.** As fixed above, this batch gives no C2.

Beside the rules:
- **The check.** This scorer's A0 counts equal `retrain6.py`'s own per-fold counts: the largest IoU difference is 0.000.
- **Batch 1's control.** The retrained control's score files (`dev_s1s2`, `dev_flood`, `dw_counts` and every `areas_passes` file, both folds) are the same as batch 1's byte for byte. So the training repeated exactly on Kaggle's T4s, and A0 here is batch 1's control (scene 39: 73.9 and 67.9 in the two folds; scene 54: 76.9 and 83.5; F 66.2 and 66.1).
- **Flips alone (A0 with flips):** mean of the scenes -0.05, scene 39 -1.33, F -0.40. The loss in A2 comes from the flips, not from combining the folds. Training already uses the 8 turns and flips, so there was little for them to add.
- **Floods per event.** A1 is within 0.4 of A0 at every event. With flips, Ghana falls from 51.3 to 44.7 and Pakistan from 54.5 to 52.9; India, Paraguay and Spain rise by about half a point.
- **Precision and recall (A1).** The weak scenes are of two kinds: false water at 39 (precision 73, recall 96) and 54 (84, 95); missed water at Asian 7 (95, 79) and 49 (96, 84).

What it shows:
- Putting all the models into one map is safe but small: never worse, +0.2 on average, at most +0.4. That is under the limit fixed in advance and within what a retraining changes.
- Where the folds disagree most (39 and 54), the combined map scores about the mean of the two folds (71.3 against 70.9; 80.5 against 80.2), not the better fold. The mean of folds already takes the swing out of the score, so combining has little left to win.
- Flips do not help this model, and hurt on two scene types.

**Consequence (as fixed above).** No C2 from this batch: Model 3's map product stays as it is (for T2 and T3, the mean of the folds' scores). `score_ens6.py` stays in the code for any later batch. The 6 development weights are in the private Kaggle dataset `is4990-module-a-dev6-weights` until part 2 is fixed.

## Development batch 3 (addendum, fixed 11 Nov 2026, before any run): cleaner labels at the water's edge, radar augmentation, longer training, grown maps

**Why.** `PLAN-ACCURACY.md` section 7 (the wider search of 11 Nov) found that Model 3 is at the published level for radar alone, and that what is left to try cheaply lies in our own training recipe:
- the reservoirs' labels are weak at the water's edge. Three of four training images carry a label from a photo taken 1 to 5 days before or after the radar, and plants are labelled land also where they stand in the water;
- the only augmentation is turns and flips, which batch 2 and two published studies found useless on radar;
- longer training removed most of the false water on the forest and dry scenes in batch 1, but batch 1's rules pooled all pixels (one scene held half of them) and it was not kept.

The team chose this batch on 11 Nov ("do your best"). It is the last map batch planned before part 2.

**The recipes.** Each is trained on folds 1 and 2 with 3 seeds each, by `kernel6-dev/retrain6.py`, with everything else as the control (the same seeds, starts, batch mix 10/4/2, 24 S1S2-Water training scenes, learning rate).

| Code | Variant in the code | Reservoir labels | Augmentation | Steps |
|---|---|---|---|---|
| **R0** | `control` (batch 2's 6 models, not trained again) | As stored | Turns and flips | 3,000 |
| **R1** | `edge` | Cleaned at the edge | Turns and flips | 3,000 |
| **R2** | `aug` | As stored | Turns, flips and radar augmentation | 3,000 |
| **R3** | `edgeaug` | Cleaned at the edge | Turns, flips and radar augmentation | 3,000 |
| **R4** | `long` (batch 1's run; its score files are used, nothing is trained) | As stored | Turns and flips | 9,000, moving average |
| **R5** | `longedgeaug` | Cleaned at the edge | Turns, flips and radar augmentation | 9,000, moving average |

**The edge clean-up** (training labels of our reservoirs only; photos and dates only, no storage record):
1. Inside the outline, a pixel that the photo shows as plants and that lies within 3 pixels (30 m) of the photo's open water becomes "unsure" (it was "land"). Floating and standing plants sit beside the open water; plants further away stay land.
2. When the label's photo is 1 day or more from the radar image, the pixels within a short distance of the line between the label's open water and its land become "unsure", on both sides: 1 pixel for 1 or 2 days, 2 pixels for 3 to 5 days. Same-day labels keep their edge.

Both parts are taken from the label as stored. On a sample of 30 photos (labels only, no model), part 1 moves about 4% of the outlines' pixels to "unsure", a 1-pixel band about 3% and a 2-pixel band about 8%. The code checks that it rebuilds every stored label exactly from its photo before it changes anything.

**The radar augmentation** (every training crop, in dB, before normalising; its random numbers come from their own generator, so every recipe sees the same crops in the same order):
- contrast: each band's distance from the training mean is multiplied by one factor, uniform from 0.9 to 1.1;
- brightness: one shift for both bands, uniform from -2 to +2 dB, plus a shift per band, uniform from -0.5 to +0.5 dB;
- speckle, for half of the crops: the power of every pixel is multiplied by gamma noise with L looks, L uniform from 8 to 32.

**Grown maps (G), an inference rule scored beside every recipe.** The plain map is water where p(land) is under 0.5. The grown map adds every pixel with p(land) under 0.7 that is joined to plain water (8 neighbours). The one threshold, 0.7, is fixed here and not tuned.

**Data (development only, PROTOCOL-6 B):** the 10 S1S2-Water development scenes, Sen1Floods11's validation split (89 chips, 10 events), the 42 held-out reservoirs' first-half Dynamic World chips and first-half passes. Nothing else is read.

**Scores,** each paired with R0's, unit by unit, on unrounded values:

| Code | Score | Units |
|---|---|---|
| **S** | Water IoU of each development scene; a scene's value is the mean of the two folds' IoUs | 10 scenes |
| **F** | Water IoU pooled over the 89 flood chips, the mean of the two folds | 1 number |
| **H** | Dynamic World IoU of each reservoir, pooled over its first-half clear chips (the whole chip, as M2) | 42 reservoirs |
| **T** | Storage error of each reservoir with Curve 2, exactly as the T1 scorer's development trial (`score_t1.py --dev`: fitted on the first quarter, scored on the second quarter's main-orbit passes with no radar gap), on the recipe's own areas | About 40 reservoirs |

**Keep rules** (each recipe against R0). The "no loss" rules are PROTOCOL-6's own step 2 rules, with their margins, tried on the development data. Their p-values are used here only as fixed decision limits. They are not tests of anything, and nothing here can be claimed.

- **No loss (all of):**
  - S: a one-sided Wilcoxon on (recipe - R0 + 2) > 0 over the 10 scenes has p < 0.05 (T2's rule);
  - F: down by less than 1.0 (as batches 1 and 2);
  - H: a one-sided Wilcoxon on (recipe - R0 + 1) > 0 over the reservoirs has p < 0.05;
  - T: a one-sided Wilcoxon on (R0's error + 0.5 - the recipe's error) > 0 over the reservoirs has p < 0.05 (T1's rule).
- **Gain (at least one of):**
  - S: the Hodges-Lehmann estimate of the 10 scene differences is +0.5 points or more;
  - F: up by 1.0 or more;
  - H: the Hodges-Lehmann estimate of the reservoir differences is +0.5 points or more.

A recipe is **kept** if it gains somewhere and loses nowhere.

**The choice.** Among the kept recipes, the one with the highest mean IoU over the 10 scenes is taken, unless a simpler kept recipe is within 0.5 points of it; then the simplest such recipe is taken (simplest first: R1, R2, R3, R4, R5). If none is kept, the recipe stays R0.

**Then the grown maps.** G is judged on the chosen recipe (on R0 if none was kept): its grown maps against its plain maps, by the same four "no loss" rules and the same gains. If R4 is chosen, it is first trained again by the same code, because batch 1's weights are gone; batch 2 showed that the training repeats exactly on Kaggle's T4s.

**What a kept result means for part 2.**
- C2's map recipe would be the chosen recipe, with grown maps if G is kept. The confirmatory run trains it on all 5 folds and all 29 S1S2-Water training scenes, as PROTOCOL-6 says.
- Unlike batch 2's candidates, C2's reservoir areas would then differ from C1's, so T1's "no loss" rule is a real test for it.
- By PROTOCOL-6's fixed order, a gain on T2 or T3 is tested only after both T1 gains hold. A new map cannot move the storage error by 0.5 points (`PLAN-ACCURACY.md` section 1). So C2's differences on T2 and T3 would be reported with their intervals and not claimed, and the "no loss" rules decide whether C2 becomes Module A's map.

**Stated openly, before the run.**
- **R4's scores have been seen.** They are batch 1's "long", which batch 1's rules did not keep. The rules above were written after that, knowing that long is far better on scenes 39 and 54 and lower on scene 11 in one fold. They follow PROTOCOL-6's margins and the 7 Nov review's request to judge per scene, and they were not fitted to make R4 pass. Even so, a "kept" for R4 on this data is weaker evidence than for the other recipes, and only T2 and T3 can confirm it.
- R0, R1, R2, R3 and R5 have no batch 3 score yet. R0's plain scores are batch 2's (equal to batch 1's control byte for byte).
- Development has 2 folds (6 models per recipe), 10 scenes and 42 reservoirs. Two recipes differ by a whole retraining, so a scene-level difference under about 1 point is noise.
- H is scored against Dynamic World, which disagrees with our photo labels at the edge by about as much as the model does (`PLAN-ACCURACY.md` section 2).

**Reported beside the rules (not rules):**
- per-fold scores, precision and recall;
- the flood IoU per event (R4 has only the pooled value);
- Dynamic World inside the outline only;
- the storage error with Curve 1 (batch 1's M4);
- every Hodges-Lehmann estimate with its interval;
- a check that R0's plain score files from this run equal batch 1's byte for byte (so the new scoring code changed nothing);
- how many label pixels the clean-up changed.

**How it runs.**
1. A tiny check of the new code on Kaggle without a GPU (`--quick`: 2 steps, a handful of chips). Its numbers are not scores and are not looked at as such.
2. Two private Kaggle notebooks (two T4s each): one scores R0's saved models and trains R1, R2 and R3; the other trains R5. About 5.5 GPU hours in all.
3. The outputs are downloaded to `runs/dev6c/` and scored on the laptop by `scripts/score_dev6c.py`.
4. The weights stay in the notebooks' outputs until part 2 is fixed.

### Batch 3: the result (13 Nov 2026)

Run on Kaggle on 13 Nov: the quick check first (no GPU, all three variants ended without error), then three private notebooks on two T4s each: `dev6 batch3a` (R0 scored again; R1, R2 and R3 trained), `batch3b` (R5) and `batch3c` (R4 trained again, for its grown maps). All folds finished with no problem. The outputs, without the weights, are in `runs/dev6c/`; `scripts/score_dev6c.py` wrote `dev6c_scenes.csv`, `dev6c_floods.csv`, `dev6c_reservoirs.csv` and `dev6c_verdict.csv` there. Everything below is development data: nothing can be claimed from it.

**The checks.** R0's plain score files from this run equal batch 2's byte for byte (48 of 48), so the new scoring code changed nothing. R4 trained again equals batch 1's "long" byte for byte (48 of 48), so the training repeated exactly once more.

| Scene (S, water IoU) | R0 | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|---|
| Asian 6 | 83.70 | 83.86 | 85.95 | 86.27 | 86.69 | 85.62 |
| Asian 7 | 75.96 | 76.15 | 75.10 | 75.84 | 74.26 | 75.15 |
| Asian 11 | 97.93 | 97.92 | 97.94 | 97.95 | 95.54 | 97.51 |
| Asian 12 | 93.00 | 92.82 | 92.67 | 92.66 | 94.14 | 93.74 |
| Asian 52 | 94.20 | 94.06 | 94.22 | 94.08 | 94.11 | 93.87 |
| Held-out 39 | 70.90 | 75.90 | 68.25 | 70.22 | 91.28 | 89.07 |
| Held-out 49 | 81.23 | 81.40 | 80.99 | 81.21 | 81.93 | 81.54 |
| Held-out 54 | 80.22 | 81.32 | 77.39 | 78.11 | 87.05 | 84.97 |
| Held-out 76 | 96.26 | 96.35 | 96.04 | 96.22 | 96.15 | 96.21 |
| Held-out 83 | 80.85 | 80.24 | 79.80 | 79.82 | 82.89 | 81.69 |
| Mean of the 10 scenes | 85.42 | 86.00 | 84.83 | 85.24 | 88.40 | 87.94 |
| **F**, floods (89 chips) | 66.15 | 66.08 | 64.97 | 65.63 | 67.21 | 66.31 |
| **H**, median reservoir | 86.00 | 86.93 | 85.97 | 86.33 | 86.27 | 87.67 |
| **T**, median error, Curve 2 | 5.74 | 5.95 | 7.19 | 7.16 | 5.87 | 6.34 |

Each recipe against R0, by the fixed rules (HL = Hodges-Lehmann estimate; H on 38 reservoirs, T on 39):

| Recipe | S: HL (interval), lowest scene, no-loss p | F | H: HL (interval), no-loss p | T: HL gain (interval), no-loss p | Gains | Losses | Verdict |
|---|---|---|---|---|---|---|---|
| **R1** edge | +0.13 (-0.16 to +2.20), -0.60, 0.001 | -0.07 | +0.74 (+0.57 to +0.95), < 0.001 | +0.01 (-0.17 to +0.09), 0.002 | H | none | **kept** |
| **R2** aug | -0.51 (-1.53 to +0.01), -2.83, 0.005 | -1.18 | -0.24 (-0.47 to -0.01), < 0.001 | -0.07 (-0.39 to +0.10), 0.011 | none | F | not kept |
| **R3** edgeaug | -0.19 (-1.03 to +0.23), -2.11, 0.002 | -0.52 | +0.25 (-0.11 to +0.54), < 0.001 | -0.11 (-0.44 to +0.09), 0.013 | none | none | not kept |
| **R4** long | +1.37 (-0.50 to +8.99), -2.40, 0.003 | +1.07 | +0.04 (-0.52 to +0.76), < 0.001 | -0.06 (-0.40 to +0.24), 0.005 | S, F | none | **kept** |
| **R5** longedgeaug | +0.75 (-0.19 to +8.68), -0.81, 0.001 | +0.16 | +0.82 (+0.22 to +1.53), < 0.001 | -0.32 (-0.81 to +0.18), 0.230 | S, H | T | not kept |

- **Kept: R1 and R4. The choice: R4** (the highest mean over the scenes, 88.40; R1, the only simpler kept recipe, is 2.40 below it, more than the 0.5 allowed).
- **Grown maps (G) on R4, against R4's plain maps: not kept.** S: mean 86.37 (-2.03), HL -1.72 (-3.99 to -0.30), up at 2 of 10 scenes, lowest -8.54 (scene 39: 91.28 to 82.73), no-loss p 0.42: a loss. F +0.68; H HL +2.21 (+1.74 to +2.78), up at 35 of 38; T HL -0.05 (no-loss p < 0.001).

Beside the rules:
- **Where R4 gains.** False water on the forest and dry scenes: precision at scene 39 goes from 73 to 94 (recall 96 to 97), at scene 54 from 84 to 92, at Asian 6 from 89 to 93. The two folds also agree better (scene 39: 73.9 and 67.9 for R0, 91.1 and 91.5 for R4).
- **Where R4 loses.** Asian 11 falls from 97.93 to 95.54, from one fold only (97.95 and 93.13; recall 98 to 96), and Asian 7 from 75.96 to 74.26 (72.67 and 75.84). Both were known before the rules were written.
- **Floods per event** (R4 has only the pooled value). R5: Ghana 51.3 to 56.3, Nigeria 86.4 to 81.3, the others within 2 points. R2: Ghana falls to 45.0.
- **The edge clean-up** moved about 3.2% of the outlines' pixels to "unsure" (2.5 and 2.0 million pixels in the two folds), and rebuilt every stored label exactly before it changed anything. It lifts H at 36 of 38 reservoirs. H is scored against Dynamic World, which has its own edge error.
- **The radar augmentation does not help:** alone it lowers the scenes and the floods, and it takes away the clean-up's gain (R3).
- **R5** has R4's gain on the scenes and R1's on H, but its storage error is less steady: the no-loss rule on T fails (p 0.23; HL -0.32, better at 15 of 39).
- **Storage error.** No recipe moves it: R4's median is 5.87 against 5.74, HL -0.06 (-0.40 to +0.24). With Curve 1 the same (-0.06, -0.30 to +0.24). This agrees with E0: the map is not the main limit of the storage estimate.
- **Grown maps on the other recipes** behave as on R4: the scenes fall by 2.5 to 3.4 points on average (lowest -10.5 to -13.1), while H rises by about 2. Growing adds water at the edge that Dynamic World also calls water but the scene labels do not.

What it shows:
- Longer training with the moving average is the one map change that holds across the scenes by the per-scene rules: +3.0 on average, most of it from two scenes with false water, and no loss anywhere by PROTOCOL-6's margins.
- Cleaner edge labels are a real but small gain on our own reservoirs, and only against Dynamic World.
- The stated caveat stands: R4's scores had been seen before these rules were written, so this is weaker evidence than a first look. Only T2 and T3 can confirm it.

**Consequence (as fixed above).** C2's map recipe is **R4: 9,000 steps with the weights' moving average, plain maps** (no grown maps, labels as stored, turns and flips only). Part 2's confirmatory run trains it on all 5 folds and all 29 S1S2-Water training scenes. T1's "no loss" rule is then a real test for C2; C2's differences on T2 and T3 are reported with their intervals and not claimed. The development weights of this batch stay in the three notebooks' outputs on Kaggle until part 2 is fixed.

## Curve study step 5 (13 Nov 2026): the season-aware correction is not kept

As fixed in `PROTOCOL-CURVE.md`'s step 5 addendum (commit 4d3afc1) before any candidate was scored. `scripts/curve_season.py`, outputs `runs/e1/e1_season_*.csv`. First half, first quarter to second, 103 reservoirs; L4c scores 5.64 again.

| Candidate | Overall (median, % of capacity) | Against L4c | Better / worse (of 103) | One-sided p |
|---|---|---|---|---|
| Z0 = L4c | 5.635 | - | - | - |
| **Z1** blend with the seasonal value, everywhere | 5.608 | +0.03 | 31 / 26 | 0.16 |
| **Z2** blend where L4c reads 60% or more | 5.608 | +0.03 | 22 / 20 | 0.23 |
| **Z3** yearly wave inside the curve | 6.157 | -0.52 | 43 / 58 | 0.94 |
| The seasonal value alone | 14.20 | -8.57 | 17 / 86 | - |

- **The rule:** the best is Z1, and it is also the simplest. It is 0.03 points better than L4c, where 0.1 was needed, and the paired test does not hold (p 0.16; Hodges-Lehmann gain 0.00, interval 0.00 to +0.04). **Not kept.**
- **Consequence (as fixed):** C1's storage method stays **L4c on the main orbit with no smoothing**, and no further curve candidate is tried before T1.

Beside the rule:
- **The weights are small.** Z1's weight is 0 at 42 reservoirs and its median is 0.01; it is above 0.1 at 37. Z2's is 0 at 50. Where the first quarter says the seasonal value helps, it gets some weight; at most reservoirs it says it does not.
- **By L4c's own reading** (pooled second-quarter passes, median absolute error): under 60%, 3.29 for L4c and 3.24 for Z1; 60 to 85%, 8.23 and 7.57; 85% and over, 5.15 and 5.02. The blend helps a little where the curve scatters most, as hoped, but not enough to move the reservoirs' medians.
- **November to January** (102 reservoirs with at least 3 such passes): L4c 4.66, Z1 4.82, Z2 4.77, Z3 6.35. No gain in T1's own season.
- **A few reservoirs move a lot:** Z1's largest gain is 17.1 points and its largest loss 8.7; the mean gain is 0.49, the median 0. On the T1 scorer's trial (100 reservoirs) Z1 reads 5.70 against L4c's 5.82, and its Hodges-Lehmann gain over today's method is 0.59 (0.29 to 1.11) against L4c's 0.33. That comes from those few reservoirs: against L4c itself the paired test has p 0.22. It is not a reason to keep Z1, and the rule was fixed so that it could not become one.
- **The yearly wave (Z3) overfits:** fitted at 101 reservoirs, worse at 58. Two extra numbers per reservoir are too many for one or two years of passes.
- **The record has a hole.** The department's record holds no storage from April to September 2021, so 759 of the 3,703 second-quarter passes (a fifth) had no seasonal value and read L4c. Elsewhere the seasonal value rests on one or two earlier years.

What it shows:
- With one or two years of history, the season says too little about a reservoir's storage to correct the curve: the seasonal value alone is off by 14 points, against 5.6 for the curve. Storage differs too much from year to year.
- The limit stated before the run stands: part 2 would have 6 years of history, so a seasonal value could be worth more there. That cannot be tested on development data, and by the fixed stop it is not tried before T1. It can be looked at after February, on data that T1 does not use.
- Near full, the remaining error needs information the area does not carry: water height (Module B).
