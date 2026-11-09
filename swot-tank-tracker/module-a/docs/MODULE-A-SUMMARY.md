# Module A: what we have done, step by step (to 9 Nov 2026)

Module A turns satellite radar into **water area** for Sri Lanka's reservoirs. This page tells the whole story in order: what we tried, what worked, what did not, and what is left. The details are in the files named at each step.

**In one paragraph:** radar sees through cloud, so it can read a reservoir every 12 days in any weather. We built a water map model (a U-Net) for the radar, then retrained it on our own reservoirs until it passed every check (Model 3). To check its water area, we turn the area into storage with a curve and compare with the department's records. A better curve (Curve 2) brought that storage error from about 7% to about 6% of capacity on our development data. The final, independent test runs on new satellite passes from November 2026 to January 2027.

## The short version to remember: 3 models and 2 curves

| Name | Date | What it is | Result | Old name in the other files |
|---|---|---|---|---|
| **Model 1, the base model** | 2 Nov | Learnt water from flood pictures of other countries (radar only) | Works through cloud, but calls the dry lake bed water. Storage error about 11 to 12% | r6 (first series, run 6) |
| **Model 2, the in-between tries** | 2 to 5 Nov | Retrained on our reservoirs: four tries, each fixing one problem | Storage error down to 8 or 9%, but each try lost something elsewhere (floods or abroad) | runs 1 to 4 |
| **Model 3, the current best** ⭐ | 5 Nov | The same, with a balanced mix of our reservoirs, floods and expert-labelled lakes | **Passed every check.** Its "final" version is trained on all 103 reservoirs and is the one we use | run 5, run 5 final |
| **Curve 1, the straight line** | from the start | Turns area into storage: one straight line per reservoir | About 7% error on development data | L0 |
| **Curve 2, the bending line** ⭐ | 7 Nov | Can bend upward near full (never downward) | About 6% error on development data | L4c (L4 was a step on the way) |

- **The February test (T1)** compares "Model 3 + Curve 1" (today's method) with "Model 3 + Curve 2" (the new method).
- **Who owns which part:** in the project plan, Module A delivers the **water area**, and turning area into storage is **Module B's** job (with water height from SWOT). We use a curve only to **check** our area against the department's storage records, and we hand the curve findings to Module B.

---

## Part 1: the data and the first checks (1 to 2 Nov)

### Step 1. The data

- **Storage, the truth.** The Irrigation Department's daily storage for 117 reservoirs, Oct 2020 to Oct 2026 (`temp/model-check/storage_daily.csv`, private).
  - The records have occasional stale or wrong entries, which we clean.
  - From 26 Oct 2026, a robot (`jarvis-uom/water-data-logger`) saves the department's daily sheet three times a day.
- **Satellite chips** from Earth Engine (`temp/module-a-reservoirs/`):
  - 3,843 chips of clear and cloudy days (photo + radar);
  - 16,552 radar chips, every evening pass of every reservoir, Oct 2020 to Oct 2026;
  - later, 15,414 morning-pass chips (the second orbit).
- All chips are on Kaggle (two public datasets, the rest private). Links: `temp/KAGGLE-LINKS.md`.

### Step 2. The base model, Model 1 (`temp/module-a-kaggle/`)

- We trained U-Nets on Sen1Floods11, a public flood dataset with water drawn by experts, on Kaggle: radar only, photo only, and both together.
- Radar + photo was best on clear days (about 83 IoU). Under thick cloud, though, only radar keeps working.
- So the radar-only model (3 seeds) became **Model 1, the base** for everything after.

### Step 3. Does radar follow storage at our reservoirs? (`temp/module-a-reservoirs/README.md`)

- **Yes.** On about 15,000 passes at 107 reservoirs, the flood-trained radar model's water area follows storage well (Spearman about 0.8, on clear and cloudy days alike). This check used r2, an earlier flood model of the same kind; Model 1 itself does a little better (about 11%).
- **Storage error about 12% of capacity**, against 22% with no model (just guessing the usual storage).
- **Two problems found:**
  - at low storage, the model calls the **exposed lake bed water**;
  - near full, the area **stops growing** while storage still goes up.

---

## Part 2: retraining on our own reservoirs (Models 2 and 3, 2 to 5 Nov)

### The method (the same for every try)

- **Labels without hand labelling.** Labels come from clear photos: an automatic water test on each clear Sentinel-2 photo marks the water. The label is copied to the nearest radar pass at most 5 days away.
  - Tries 1 to 3 also used a "lake high enough" rule (each spot on the bed goes under water above a certain storage), which used storage to move labels. It did no better than the photo labels and was dropped.
  - From try 4, the training files hold **no storage numbers at all**.
  - Every training image and its label source is listed in `out/label_pairs.csv`: 2,785 images (1,444 clear photos and 1,341 radar passes; most passes are 0 to 2 days from their photo).
- **Five folds by reservoir.** Every reservoir is tested by a model that never saw it, so the scores are honest. One try makes 15 models (5 folds × 3 seeds).
- **Rules fixed before each try** (`PROTOCOL.md` to `PROTOCOL-5.md`). A try "wins" only if every rule holds. This stops us from choosing whatever looks best afterwards.

### The tries

| Try | What changed | Storage error (% of capacity) | Verdict | Why |
|---|---|---|---|---|
| Model 2, try 1 | First retraining, with the "lake high enough" rule | 9 (the flood model: 12) | Not a win | Lost too much on Bolivia (a flood test) |
| Model 2, try 2 | Better labels, cleaned storage sheet, 3 seeds, a photo-label control | 8 | Not a win | Bolivia fell further. The photo labels did as well as the rule, so the gain comes from training on our reservoirs |
| Model 2, try 3 | Flood chips mixed into every batch | 8 | Not a win | Bolivia recovered a lot but not enough. The rule is dropped |
| Model 2, try 4 | S1S2-Water (expert-labelled water from many countries) added | 9 | Not a win | Fixed the false water abroad (IoU 86 against 65), but missed three "no loss" checks by tiny amounts |
| **Model 3** | **Try 4 with a different batch mix (10 reservoir, 4 flood, 2 S1S2-Water crops)** | **9** | **A win** | **All four rules held** |

- **Model 3 is Module A's model.** It is reported as exploratory, because its change was chosen after try 4's scores and the same tests had been used before. Three checks passed by thin margins.
- **Model 3 final:** Model 3's recipe trained on all 103 reservoirs, for use (`FINAL-MODEL-5.md`). Weights are in the private Kaggle dataset `is4990-module-a-run5-weights`.

### The outside checks that Model 3 passed

| Check | What it measures | Model 3 |
|---|---|---|
| S1S2-Water, 15 test scenes | Water maps against expert masks, abroad | IoU 85 (control 65) |
| Dynamic World, 931 chips | Water maps against Google's land-cover maps, at home | IoU 91 |
| Storage, 103 reservoirs | The later half of each reservoir's history | About 9% of capacity, no worse than the control |
| Sen1Floods11 test split | Floods | IoU 66 |

---

## Part 3: series 2, aiming higher with fresh tests (5 to 9 Nov)

### Step 4. Fresh tests, fixed first (`PROTOCOL-6.md`, 5 Nov)

Model 3's tests had been used five times, so they are no longer independent. Series 2 is judged only on tests no model has seen:

| Test | What | When |
|---|---|---|
| **T1** | Storage on **new** passes, 1 Nov 2026 to 31 Jan 2027, against the department's daily sheets | Scored early Feb 2027 |
| **T2** | 14 S1S2-Water scenes never downloaded or scored | By 15 Jan 2027 |
| **T3** | Kuro Siwo's 10 flood events (MMFlood if Kuro Siwo cannot be prepared) | By 15 Jan 2027 |

The methods compared (old codes in brackets):
- **today's method** (I): Model 3 + Curve 1;
- **the new-curve method** (C1): Model 3 + a new curve;
- **the new-maps method** (C2): a new model + the new curve, only if a map change survives development.

Development (choosing things) uses only old data: the first half of each reservoir's history, and some held-out scenes.

### Step 5. Can longer training improve Model 3? (`DEV-6.md`)

Two try-outs of Model 3's recipe, run on Kaggle GPUs by a teammate:

| Try-out (old name) | Floods | Reservoir maps | Storage error | Kept? |
|---|---|---|---|---|
| Model 3's recipe again (control) | 66 | 87 | 7.0 | - |
| Longer training (long) | 67 | 85 | 6.8 | No |
| Longer + an extra loss (lovasz) | 68 | 83 | 8.4 | No |

- Both help floods but hurt the reservoir maps, and neither helps storage. **Model 3 stays.**
- **Limitation found later:** one big scene dominates one score, and another rests on few chips. It does not change the outcome, because storage did not improve either.

### Step 6. Where does the storage error come from? (the error budget, E0)

On the development data, in % of capacity:

| Measure | Value |
|---|---|
| Storage from Model 3's radar maps | 6.7 |
| Storage from clear photos instead (as if the map were perfect) | 8.4 |
| Noise between passes when storage is steady | 1.5 |
| Error below 40% full | about 3 |
| **Error at 85% full or more** (a third of all passes) | **10** |

- **The map is not the main limit.** Even photos do worse than our radar, and on 9 Nov two reviews showed why: the photo's water area also stops growing near full, exactly like the radar's.
- **The error comes from near full.** There the water area hardly changes (it is already 97% of its top size at 85 to 95% full), while storage still moves a lot.
- **A correction (9 Nov).** We first wrote that the line "always reads about 9 points too low" near full. That was mostly an illusion of how we grouped the passes: if you look only at the days when a reservoir was really full, any guess looks too low, because a guess can almost never be above full. Grouped by what the curve reads, the bias is small (−3 to +2). **The real problem is scatter:** when the curve reads 75 to 85% full, it is off by about 9 points either way.
- So the next gain is not in more map training. Near full, the area alone cannot tell 70% from 95% full; only extra information can (Module B's water height).

### Step 7. The curve study E1 (`PROTOCOL-CURVE.md`, fixed before scoring)

Fitted on the first quarter of each reservoir's history, judged on the second. We keep the simplest curve within 0.1 points of the best.

| Step | Candidates | Winner |
|---|---|---|
| 1. The curve | The straight line (Curve 1); capped; plain line; robust line; **bending line**; monotone curve | **The bending line (L4): 5.6 against 6.7**, better at 63 of 103 reservoirs |
| 2. Smoothing over time | None; mean over 12 days; median over 24 days | None (no clear help) |
| 3. The morning passes | Not used; added to the curve; scaled first; averaged with the latest morning reading | **Not used.** They carry the same information, though alone they are about as accurate (a reading every 6 days instead of 12 would be possible) |
| 4. A fix from an outside review | **The bending line, never bending down** | **Chosen as Curve 2** (L4c): same score; physically impossible curves 18 to 1; error near full about 9 to 8 |

The morning passes were measured by Model 3's test models on Kaggle's free CPU (6 hours, 13,808 passes, no GPU used).

**The new-curve method is Model 3 + Curve 2, main orbit, no smoothing.**

| | Curve 1 (today) | **Curve 2 (new)** |
|---|---|---|
| Storage error on development data (median) | 6.7% | **5.6%** |
| Gain over Curve 1, reservoir by reservoir (Hodges-Lehmann, 95% range) | - | **0.32 points** (0.19 to 0.74) |

The fair number to quote is the paired gain, 0.32 points: the two medians differ by about 1 point, but reservoir by reservoir the typical gain is smaller.

### Step 8. The T1 scorer, written and tried now (`scripts/score_t1.py`)

- **The truth** is the department's daily sheet saved on the pass date. On every complete day it matches the department's history at 74 of 74 reservoirs.
- **A pass counts** only with no radar gap in either radar band.
- **Trial on old data:** today's method 6.7%, the new method 5.8%, the simple baselines 15% and 21%.
  - The new method passes the "no loss" rule.
  - Its gain (0.33) is under the 0.5 needed to **claim** a gain.
- **What this means for February:** the new method will most likely replace today's, but may not be a claimable improvement. The February window is the season when reservoirs are full most often (27% of passes in November, about 50% in December and January), so the result could land above or below the trial's 0.33.

### Step 9. Outside reviews

Outside reviews on 5, 7 and 9 Nov. Every claim was checked against the files before acting (`REVIEW-2026-11-05.md`, and `DEV-6.md` for 7 and 9 Nov). Many points were right, and some were wrong.
- **Useful finds:** the downhill curves (step 7.4), T1's exact truth rule, the scoring weaknesses of the try-outs, and (9 Nov) the near-full correction in step 6 and the February season effect.
- **Wrong claims we rejected:** invented datasets, wrong citations, that fitting in logs causes the near-full error, and that rainfall could be added (PROTOCOL-6 allows only radar and the reservoir's own history).

---

## Where we stand

| Part | Status |
|---|---|
| Detecting water | ✅ **Done: Model 3** (and Model 3 final, trained on all 103 reservoirs). Works through cloud and passed every check |
| Water area | ✅ **Done:** the area of every radar pass at 103 reservoirs, 2020 to 2026 |
| Reservoirs with plants around them | ✅ Works: about 92 IoU against Dynamic World at plant-covered reservoirs |
| **Water under plants** | ❌ **Not done.** Model 3 has a slot for it, but from try 4 on it had no examples to learn from, and there is no test yet. It is part of RQ1 |
| The storage check | Curve 2: about 6% of capacity against about 7% on development data. Final test T1 in Feb 2027 |
| Near full | Still the weak spot: the reading scatters by about 9 points there. The fix is **water height** (SWOT, ICESat-2), which is Module B's job |
| GPU | Not needed for the storage work. A better map would not fix the near-full scatter |

## What is left

1. **Water under plants (RQ1):** find a trustworthy way to get examples without hand labelling, then train and test.
2. **Decide on one more curve try:** a season correction near full (laptop, about 1 day, kept only if it beats Curve 2 by 0.1), or stop and hand over.
3. Scorers for T2 and T3, tried on development data (no fresh data touched).
4. Prepare T3: Kuro Siwo masks and Earth Engine radar, by 15 Jan 2027.
5. Fix PROTOCOL-6 part 2 by 15 Jan 2027 (the new curve fixed; C2 not run).
6. After 31 Jan 2027: download the window's passes, measure them with Model 3's test models (Kaggle CPU), and run `score_t1.py --window`.
7. Hand over to Module B: Model 3's areas for the 103 reservoirs, Model 3 final for new passes, and the curve findings.
8. The supervisor note on Model 3 (formal register).

## The honest limits

- Model 3's win is exploratory: its tests had been used before. That is why series 2 has fresh tests.
- All the series 2 numbers above are **development** numbers. Only T1 to T3 can be claimed.
- Water under plants is not done yet.
- Small reservoirs (under 5 million m3) are read worse than big ones (about 12% against 5% of capacity), and reservoirs that are often full are read worse than the rest (about 7% against 4%).
- The department's storage comes from a depth reading and the reservoir's own table, which may be old; silt can make it wrong. It is still the best truth we have.
- The department's sheets have occasional wrong entries (6 of 470 days in the overlap check). One bad day moves a reservoir's median little.

## Where things are

| What | Where |
|---|---|
| Models 2 and 3: rules and results | `PROTOCOL.md` to `PROTOCOL-5.md`, `RESULTS-run1.md` to `RESULTS-run5.md` (tries 1 to 4 = runs 1 to 4; Model 3 = run 5), `FINAL-MODEL-5.md` |
| Series 2 rules | `PROTOCOL-6.md`, `DEV-6.md` (top), `PROTOCOL-CURVE.md` |
| Series 2 results | `DEV-6.md` (bottom), outputs in `runs/dev6/`, `runs/e0/`, `runs/e1/`, `runs/t1/`, `runs/morning-areas/` |
| Training labels | `out/label_pairs.csv` (`scripts/list_label_pairs.py`) |
| Scripts | `scripts/` (score_dev6, error_budget, curve_study, curve_morning, curve_l4c, score_t1) |
| Kaggle notebooks and datasets | `temp/KAGGLE-LINKS.md` |
| The map of all folders | `temp/README.md` |
