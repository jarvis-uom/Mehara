# Protocol for series 2 (PROTOCOL-6), part 1: the fresh tests and the development data (FIXED 5 Nov 2026)

**Status:** fixed on 5 Nov 2026 at about 23:10 (Sri Lanka time), with the team leader's approval, as draft 1 (commit 7b4bdb4) with no change. It was written before any series 2 work. Nothing from the fresh tests below had been downloaded, prepared or scored. The only series 2 work already started was model-free: the download of the morning passes' history (section D).

From now on it changes only by a recorded amendment, made before any confirmatory score is seen.

**Why a series 2.** Runs 1 to 5 used the same four tests five times or more, and run 5 passed three checks by 0.07 to 0.23 points. More tuning against those tests would not be believed. Series 2 therefore:
- **develops** only on data named in section B;
- **confirms** once, on data that no model has seen (section A);
- reports the old tests as **reused** (section C).

The ideas to be developed come from the check of three outside reviews.

**Part 1 vs part 2.**
- **Part 1** (this file) fixes the fresh tests, the comparisons, the rules and the development data.
- **Part 2** is written after development and before the confirmatory run. It fixes the storage method, the map recipe, the inference, the seeds and the code commit. Part 2 cannot change anything in part 1.

## A. The confirmatory tests (fresh)

### The pipelines compared

| Name | Water maps | Storage method |
|---|---|---|
| **I, the incumbent** | Run 5's fold models (each reservoir by the fold that never trained on it) | Today's method: one least-squares line of log storage on log area per reservoir, one pass at a time |
| **C1** | Run 5's fold models, as I | Series 2's storage method (part 2): for example a new curve, smoothing over time, the morning passes |
| **C2** | Series 2's map models (part 2), out-of-fold | The same storage method as C1 |

If no map change survives development, C2 is not run, and only I and C1 are compared.

**The order of decisions:**
1. C1 is compared with I. This shows what the storage method alone gives.
2. C2 is compared with C1. This shows what the new maps add.

**Module A's storage pipeline after series 2:**
- C2, if every "no loss" rule of step 2 holds;
- otherwise C1, if the "no loss" rule of step 1 holds;
- otherwise I.

### T1. The prospective storage test (the primary test)

- **Reservoirs:** run 5's 103, with run 5's folds.
- **Window:** every radar pass acquired from **1 Nov 2026 to 31 Jan 2027**, inclusive. None had been downloaded on 5 Nov 2026.
- **Evaluation dates:** the dates in the window with a pass on the reservoir's **main orbit**, the orbit of its history in run 5. That is orbit 27 (evening) for 101 reservoirs, orbit 19 for Tabbowa and orbit 121 for Thumpankerny.
  - A date is scored only if that pass has no radar gap inside the outline (VV and VH both).
- **What each pipeline may use for a date:**
  - radar acquired on or before that date (main and second orbit);
  - the reservoir's history to 31 Oct 2026 (radar, storage, clear photos);
  - no storage record from the window, no photo from the window, and nothing acquired after the date.
- **Curves:** each pipeline fits its curve once per reservoir, on all its history passes to 31 Oct 2026 (storage within 2 days, as in rule 3), using its own out-of-fold areas. The curve is not refitted in the window.
- **The truth:** the Irrigation Department's daily storage, from the sheets saved by `jarvis-uom/water-data-logger` (`reservoir-status/`), in the same units and against the same capacity as rule 3.
  - The sheet of the pass date is used.
  - If it is missing, the mean of the day before and the day after is used, if both exist. If only one exists, that one is used.
  - Otherwise the date is not scored.
- **The error at a reservoir:** the median absolute error over its scored dates, in % of capacity. A reservoir with fewer than 3 scored dates is left out for all pipelines, and the number left out is reported.
- **Baselines** (reported):
  - seasonal: the median history storage within 15 days of the same day of the year;
  - no model: the median history storage.

### T2. S1S2-Water's validation split (14 scenes)

- **Scenes:** 10, 13, 17, 22, 28, 32, 37, 51, 55, 62, 68, 73, 87 and 93 (from `s1s2water-check/data/scenes.json`).
  - None of these was ever downloaded, prepared or scored; the logs name only the 20 scored scenes and the training scenes.
  - The other two validation scenes, 7 and 11, were scored as Asian extras and are not part of T2.
- **Preparation:** exactly as the 20 scored scenes (`fetch_gee_radar.py`: Earth Engine radar of each scene's own Sentinel-1 product, the expert mask resampled to 10 m and registered by the same rule). No model is run during preparation.
- **When a scene is left out:** only if its products are not in Earth Engine, or its registration fails the checks used for the 20 scenes. These reasons are fixed now.
- **Score:** pooled water IoU (and per-scene IoU, for the paired tests), as in rule 1.
- **Maps compared:**
  - I uses run 5's mean of 5 fold ensembles, as in rule 1;
  - C2 uses its map product as defined in part 2;
  - if part 2's product differs from a mean of folds (for example a 15-model ensemble), C2's mean of folds is also reported.

### T3. An outside expert-labelled dataset

- **First choice: Kuro Siwo's test events** (Bountos et al., NeurIPS 2024; 10 flood events held out by location; 2015 to 2022).
  - Its own radar is speckle-filtered (Lee Sigma 3 x 3, from its SNAP graph). Our speckle check shows that filtered radar breaks our models (r6 on Bolivia: 66.0 to 6.7), so scoring its radar would measure the processing.
  - So, as for S1S2-Water, T3 uses Earth Engine radar of each test sample's own post-event Sentinel-1 acquisition, with Kuro Siwo's masks reprojected to it.
  - Water = permanent water or flood (codes 1 and 2), land = 0, and pixels marked invalid are left out.
- **Preparation check (model-free, fixed now).** T3 is used only if, by **15 Jan 2027**:
  - the test events' masks and post-event acquisition dates can be obtained;
  - Earth Engine has those Sentinel-1 products;
  - the registration passes the same checks as S1S2-Water.
- **Fallback, fixed now:** if Kuro Siwo cannot be prepared by then, MMFlood (Montello et al., IEEE Access 2022) is tried under the same rules, provided its data licence allows research use. If neither can be prepared, T3 is dropped and the report says so. No other dataset is chosen after this point.
- **Score:** pooled water IoU, and per-event IoU for the paired tests.

### The rules

The tests are one-sided paired Wilcoxon tests at p < 0.05, on unrounded values.

**Step 1: C1 against I (storage method only).**
- **No loss:** non-inferiority within 0.5 points of capacity on T1. That is a one-sided Wilcoxon on (I's error + 0.5 − C1's error) > 0 over reservoirs, p < 0.05.
- **Gain (claimed only if the no-loss rule holds):** both of:
  - a one-sided Wilcoxon on (I's error − C1's error) > 0, p < 0.05;
  - a Hodges-Lehmann estimate of that difference of at least **0.5 points**.

**Step 2: C2 against C1 (new maps; only if C2 exists).**
- **No loss, all of:**
  - T1: within 0.5 points of capacity, as in step 1;
  - T2: per-scene non-inferiority within 2 IoU points, a one-sided Wilcoxon on (C2 − C1 + 2) > 0, p < 0.05;
  - T3 (if prepared): per-event non-inferiority within 2 IoU points, as for T2.
- **Gains, in a fixed order.** Each is tested only if the one before it holds:
  1. T1 storage, as in step 1, with the 0.5-point minimum;
  2. T2, a one-sided Wilcoxon on (C2 − C1) > 0;
  3. T3, the same.

**Error control.**
- Within each step, all "no loss" rules must hold together (an intersection-union test), so no correction is needed.
- The gain claims form one fixed sequence: step 1's gain, then step 2's three gains. Each is tested at 0.05 only if every earlier gain held, so the chance of any false gain claim stays at most 0.05.

**Reported beside every rule (not rules):**
- the Hodges-Lehmann estimate with its exact 95% confidence interval (signed-rank);
- the share of reservoirs, scenes or events that are better;
- the baselines.

**If no rule holds, run 5 with today's method stays Module A's pipeline,** and the report says so.

**Power, stated in advance.**
- T1 has about 100 reservoirs with about 7 or 8 evaluation dates each. Run 5 against the control showed that a paired difference of about 0.2 points is detectable over 103 reservoirs.
- T2 (14 scenes) and T3 (10 events) can show only large differences. Their no-loss margins matter more than their p-values.

### Reported with T1 (not rules)

- the near-full subset (storage at 85% of capacity or more): signed and absolute error;
- size classes (under 5, 5 to 30, and 30 million m3 or more);
- the final model of each pipeline on the same dates (trained on all 103 reservoirs; a hold-out in time only);
- the error on second-orbit dates, for pipelines that use them;
- the number of scored dates per reservoir.

## B. The development data (the only data used to choose)

| What is chosen | Development data | Notes |
|---|---|---|
| The storage method (curve, smoothing, the morning passes, area calibration) | **The first half of each reservoir's history** (the passes before rule 3's split date). Fit on the first quarter, choose on the second | The candidates are listed in a short PROTOCOL-CURVE before any score. Keep the simplest candidate within 0.1 points of the best (a one-SE rule) |
| Water maps at home | Out-of-fold maps on **the clear Dynamic World chips dated in each reservoir's first half** (of the 931), and the first-half pass areas as above | These chips were part of rule 2's tests, so they are not fresh. That is fine for development, and the later-half chips are not looked at |
| Water maps abroad | **S1S2-Water training scenes 39, 49, 54, 76 and 83**, held out of every development run (one per land-cover group, drawn with seed 6: forest, grassland, dry shrubland, water bodies, flooded shrub). Also the 5 Asian scenes (6, 7, 11, 12 and 52), never trained on | Development runs train on the other 24 training scenes. The confirmatory run trains on all 29, as run 5 did |
| Floods | **Sen1Floods11's validation split** (89 chips) | It chose r6's best epoch, so it is not fresh, but it was never a test. Bolivia and the test split are not looked at |
| Fair comparisons | A **run 5 recipe trained the same way** (24 S1S2-Water scenes, the same folds and steps) as the development control | Two folds (1 and 2) are enough for development runs |

**Not looked at during development:**
- the T1 window;
- the T2 scenes;
- T3 (only its model-free preparation);
- the old tests: the later half of the storage passes, the later-half Dynamic World chips, S1S2-Water's 15 test scenes, the Sen1Floods11 test split and Bolivia.

## C. Reported as reused (not rules)

After the confirmatory run, I, C1 and C2 are scored once on the old tests, as in PROTOCOL-5. They are reported with the words "reused five times or more":
- the later-half storage error;
- Dynamic World on the 931 chips;
- S1S2-Water's 15 test scenes;
- the Sen1Floods11 test split;
- Bolivia.

## D. How it runs

- **The morning passes** (the team's decision, 5 Nov 2026).
  - Each reservoir's second orbit is the most frequent orbit of the other direction on its first-check radar dates (`module-a-reservoirs/out/radar_passes.csv`): orbit 19 or 121 (morning) for the 101 evening reservoirs, and orbit 27 for Tabbowa and Thumpankerny. 116 of the 117 reservoirs have one.
  - The history (Oct 2020 to Oct 2026, the same date rule as the main passes) is downloaded now, model-free, on the same chip grid as the main passes.
  - The window's passes (both orbits) are downloaded after 31 Jan 2027.
  - Run 5's fold models measure the morning passes too (out-of-fold, as for the main passes), so that C1 can use them.
- **The confirmatory run** (after part 2 is fixed): fold models of the kept recipe; out-of-fold areas on every history pass and window pass (both orbits); the final model on all 103 reservoirs, for use.
- **The scoring code** for T1 to T3 is written and tried on development data (never on the window) before part 2 is fixed. Its commit is named in part 2.
- **One look:** the confirmatory scores are computed once. If a bug is found afterwards, it is fixed, everything is scored again, and both results are reported.
- **Dates:**

  | When | What |
  |---|---|
  | Week of 9 Nov 2026 | Part 1 fixed; morning history downloaded; the error budget and the curve study on the first half |
  | Nov to Dec | Development runs |
  | By 15 Jan 2027 | Part 2 fixed; confirmatory models trained; T2 and T3 scored |
  | 1 to 10 Feb 2027 | The window's passes downloaded; T1 scored once |
