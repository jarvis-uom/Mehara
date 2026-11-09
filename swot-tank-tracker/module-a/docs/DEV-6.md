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

**Open option (not started):** one season-aware correction near full, using only the date and the reservoir's history, both allowed. It would be laptop only, with a fixed addendum before scoring and a hard stop unless it beats L4c by 0.1. The team has not decided.
