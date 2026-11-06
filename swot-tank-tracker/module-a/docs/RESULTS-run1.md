# Retraining run 1: results (3 Nov 2026)

Run on Kaggle from 22:36 to 00:56 (139 minutes; https://www.kaggle.com/code/mehararothila/is4990-module-a-retrain-1). Outputs are in `runs/2026-11-02-retrain-1/`. The rules were fixed before the run in `PROTOCOL.md`. Since the outside reviews (2 Nov, 23:00), run 1 counts as a **trial**. Its labels left too little land, a few held-out reservoirs leaked into training chips, and it used one seed. Run 2 (`PROTOCOL-2.md`) is the main test.

**Design:**
- 107 reservoirs, in five folds by reservoir;
- the radar U-Net of run 2, fine-tuned to 3 classes with the "lake high enough" labels (version 1);
- each reservoir scored on every radar pass by the model of the fold that held it out.

## Storage error on the later half (median, % of capacity)

| Group | Reservoirs | Retrained | Run 2 | VV tuned to the photo | Seasonal | No model |
|---|---|---|---|---|---|---|
| All | 107 | **8.9** | 12.1 | 12.6 | 14.2 | 21.9 |
| Under 5 million m3 | 41 | 12.1 | 13.7 | 13.2 | 16.2 | 25.7 |
| 5 to 30 million m3 | 35 | 7.5 | 11.6 | 12.1 | 14.4 | 17.2 |
| 30 million m3 and more | 31 | 5.2 | 8.9 | 10.5 | 13.6 | 22.9 |

Spearman with storage over all passes: 0.86 retrained, 0.80 run 2, 0.77 tuned threshold. On cloudy-photo passes: 0.84 against 0.79.

**Paired tests over the 107 reservoirs** (gain in points of capacity; Wilcoxon):

| Comparison | Measure | Median gain | Better at | p |
|---|---|---|---|---|
| Retrained vs run 2 | storage error | 1.6 | 68% | 1e-5 |
| Retrained vs tuned threshold | storage error | 1.4 | 61% | 2e-4 |
| Retrained vs seasonal | storage error | 4.5 | 77% | 2e-10 |
| Retrained vs run 2 | Spearman, all passes | 0.02 | 64% | 2e-4 |
| Retrained vs run 2 | error near full (size) | -1.6 | 40% | 0.58 (no change) |

## The four rules of `PROTOCOL.md`

| Rule | Result | Holds? |
|---|---|---|
| 1. Lower storage error than run 2 and the tuned threshold (p < 0.05, better at over half) | 68% and 61%; p = 1e-5 and 2e-4 | yes |
| 2. Lower than the seasonal baseline (p < 0.05) | p = 2e-10 | yes |
| 3. Low water: water share closer to the photo's than run 2's | photo 32%, retrained 39%, run 2 49% of the outline (124 clear days) | yes |
| 4. Bolivia water IoU at most 2 points below run 2's (mean over folds) | 56.1 against 58.6 (folds 50 to 62) | **no, by 0.5 point** |

**Verdict by its own rules: not a win.** The model tracks Sri Lankan storage much better. But on Sen1Floods11's Bolivia flood (15 chips, expert labels), it lost 2.5 points of water IoU, more than the 2 points allowed. Fine-tuning on tanks traded a little flood-mapping skill for tank skill. With 15 chips and a range of 50 to 62 between folds, much of that gap may be noise. Run 2 averages 3 seeds.

## The dry lake bed (the fix this run was for)

On passes under 20% of capacity, take the spots the storage rule knows are dry, and count how many each model calls water:

| Model | Known-dry spots called water |
|---|---|
| Run 2 | 3.29 million pixels |
| VV tuned to the photo | 1.53 million |
| **Retrained** | **0.86 million (74% fewer than run 2)** |

The retrained model calls fewer known-dry spots water at 63 of 79 reservoirs (p = 5e-10). By eye (`runs/2026-11-02-retrain-1/maps_Akathimuruppu.png`, 1 Sep 2025, 10% storage): run 2 marks most of the bed as water, while the retrained model keeps it to the pool in the south.

## Other notes

- **Pixel IoU against the storage reference** (agreement, not accuracy) over all passes: retrained 98.2, run 2 97.3, tuned threshold 92.7. Under 20% storage the water IoU is 0 for every method. That happens because only 32 of 79 reservoirs have any spot the rule can confirm as water then, so the count of known-dry spots called water (above) is the measure to use at low water. Run 2's script should add it.
- **Near full** nothing changed. The reading still flattens (the outline and the physics; see `temp/module-a-reservoirs/README.md`).
- Bolivia IoU of run 2 here is 58.6, close to the 59% that run 2 scored on Kaggle on 1 Nov.

## Added 3 Nov 03:15: the "last clear photo carried forward" baseline

Without radar, a user would read storage from the most recent clear photo. PROTOCOL-2 left this baseline for afterwards, to be scored from the saved areas by the same storage-error rule.

**How it was made:**
- `scripts/clear_photo_areas.py` (Earth Engine) counts the clear pixels (Cloud Score+ at least 0.6) and the photo's water pixels (NDWI at least -0.06) inside the fixed outline, on every Sentinel-2 date from Aug 2020 to Oct 2026 (`out/clear_photos/`, 107 reservoirs). On our own clear chips the counts match exactly.
- `scripts/score_last_clear.py` takes, on each radar pass, the last photo with at least 98% of the outline clear. It scores that photo with run 1's own functions, on the same passes as the models.
  - 124 of 15,437 early passes had no clear photo before them yet. They were dropped for every method (at 17 reservoirs).
  - At the other 90 reservoirs, the new model's errors equal the run's own.
- The curve that turns the photo's water area into storage was fitted in two ways:
  - on the first half's radar passes, as for the other methods;
  - on the first half's clear-photo days, with the storage on their own dates. This way no stale photo is paired with a later storage.

  The second is the stronger, so the tests below use it.

**How often there is a clear photo:**
- A median of 20 a year per reservoir.
- The longest gap between clear photos is a median of 138 days.
- November and December have the fewest, when the Maha rains fill the tanks.

On the later-half passes, the last clear photo is a median of 12 days old. It is over 30 days old on 29% of passes and over 60 days old on 16%.

| Storage error, later half (median, % of capacity) | All passes | Photo 10 days old or less | Photo over 30 days old |
|---|---|---|---|
| Retrained (run 1) | **8.8** | **8.0** | **10.4** |
| Run 2 | 12.1 | 11.2 | 11.2 |
| VV threshold tuned to the photo | 12.3 | 12.1 | 11.7 |
| Last clear photo, curve from photo days | 12.9 | 9.5 | 19.9 |
| Last clear photo, curve from passes | 14.4 | 13.1 | 18.6 |
| Seasonal | 14.2 | 14.2 | 13.6 |
| No model | 21.9 | 22.3 | 21.2 |

There are 107 reservoirs in the "All passes" column. 89 of them have 3 or more passes with a photo over 30 days old.

**Paired tests against the last clear photo** (curve from photo days). Each cell gives the median gain in points of capacity, the share of reservoirs where the first method is better, and p (Wilcoxon):

| First method | All passes | Photo 10 days old or less | Photo over 30 days old |
|---|---|---|---|
| Retrained | 3.0, 76%, p = 2e-8 | 1.4, 65%, p = 0.001 | 7.3, 79%, p = 9e-10 |
| Run 2 | 1.5, 58%, p = 0.02 | -1.1, 48%, p = 0.39 (no difference) | 6.1, 78%, p = 4e-8 |
| Seasonal | -0.3, 50%, p = 0.2 (no difference) | -3.2, 30%, p = 1e-5 | 4.2, 67%, p = 0.001 |

**What it means:**
- The radar's gain comes mostly from cloudy spells. When the last clear photo is over a month old, the retrained model is 7 points of capacity closer than the photo at the median reservoir. Large reservoirs (30 million m3 and more) gain most: with an old photo, 9.0 against 26.9.
- With a fresh photo, the retrained model is still a little better (1.4 points), while run 2 only ties with the photo.
- **A limit:** at small reservoirs (under 5 million m3) with an old photo, the seasonal baseline (9.6) beats every map (retrained 16.1, run 2 15.7). In the wet season, 10 m radar sees little of a small tank.
- Picture: `out/last_clear/2026-11-02-retrain-1/example.png`. The two reservoirs were picked by rank, not by eye:
  - **Minneriya** (the middle reservoir by the gain): from October 2024 to March 2025 the last clear photo stays at 49% while the tank fills to 100%, and the radar follows the fill.
  - **Hakwatunawa wawa** (the lower quarter): both methods are 9% off. The radar under-reads near full, as everywhere.
