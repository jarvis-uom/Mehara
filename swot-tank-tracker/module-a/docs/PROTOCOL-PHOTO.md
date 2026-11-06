# Protocol: Module A's water maps against the clear photo at our reservoirs (fixed 4 Nov 2026, before any scoring)

**Status:** fixed before the job runs (the team chose step 2 on 4 Nov 2026). Nothing below changes after the first score is seen. Anything learnt later goes into an amendment, made before scoring, or into a new protocol.

**Purpose.** S1S2-Water gave pixel accuracy on expert labels abroad. This gives **pixel agreement on our own Sri Lankan reservoirs**, without hand labels, as the proposal and the supervisor note promise. It is **agreement, not ground truth.** Nothing is trained.

## Data

- **Reservoirs:** the 103 of run 3 (`runs/2026-11-03-retrain-3/prep/folds.csv`).
- **Days:** every clear first-check chip of those reservoirs (`kind = clear` in the manifests): 1,571 chips. Each has a Sentinel-2 photo at least 95% clear and a Sentinel-1 radar image 0 to 2 days from it, on the same 10 m grid.
- **Reference A, the photo's water test** (`scripts/labels.py`, `photo_class`, version 2.1):
  - water: NDWI ≥ -0.06, MNDWI ≥ 0 and B11 < 0.10;
  - not water: both tests negative (bare ground or plants);
  - not scored: unsure pixels, and pixels with Cloud Score+ under 0.6.
  - *Caveat:* run 3 and its control learned from labels built with this same test (on other reservoirs), so reference A favours them by construction.
- **Reference B, Dynamic World** (`GOOGLE/DYNAMICWORLD/V1`, Brown et al., 2022): Google's land-cover map made from the same Sentinel-2 image (the chip's own `s2_images`), fetched on the chip's grid.
  - water: class 0;
  - not water: classes 1, 2, 4, 5, 6 and 7;
  - not scored: class 3 (flooded vegetation, which in Sri Lanka also includes paddy), class 8, Cloud Score+ under 0.6, and chips with no Dynamic World image.
  - **None of our models learned from it, so reference B is the main reference.**
- **Pixels scored:** the whole chip (the reservoir and its surroundings), where the radar is valid and the reference scores the pixel.

## Methods

The same water rules as in runs 2 and 3, and in PROTOCOL-S1S2:

| Method | Water |
|---|---|
| r2 | p(water) > 0.5 |
| r6 (3 seeds averaged) | p(water) > 0.5 |
| Run 3 (the reservoir's own fold, 3 seeds averaged) | p(land) < 0.5 |
| Run 3's photo-label control (own fold, 3 seeds averaged) | p(land) < 0.5 |
| Otsu | VV after a 3 × 3 median, below the Otsu threshold of the chip's valid pixels (run 2's `otsu()`) |

Each reservoir is mapped by the run 3 models of its own fold, which never saw it.

## Scores

- **Against each reference:** water IoU, F1, precision and recall, pooled over all chips. **The main result is the pooled IoU and F1 against reference B.**
- **For each reservoir:** IoU pooled over its chips; the median over the reservoirs is reported.
- **Paired Wilcoxon tests over the reservoirs** (two-sided), against reference B and against reference A:
  1. run 3 against r6;
  2. the control against run 3;
  3. run 3 against Otsu.
- **Groups, reported apart:**
  - reservoir size: under 5, 5 to 30, and 30 million m³ or more;
  - low water: storage under 20% of capacity on the photo date (the department's sheet, the nearest day within 2 days);
  - plant-covered chips: reference A's plants cover 20% or more of the outline (the JRC outline, as in run 3);
  - Dynamic World's flooded vegetation: the share of those pixels that each method calls water (a first look at water under plants, not a score).

## Questions fixed now (descriptions, not win rules)

1. On our own reservoirs, does run 3 agree with Dynamic World better than r6? (Pooled IoU difference and Wilcoxon p.)
2. Does the photo-label control agree better than run 3? (The same.)
3. How much lower is agreement on plant-covered chips and at low water?

## Where it runs

1. **The laptop** fetches Dynamic World for the 1,571 chips (`scripts/fetch_dynamic_world.py`; light, downloads only). It goes to the private Kaggle dataset `is4990-module-a-dynamic-world`.
2. **A private Kaggle CPU notebook** (`kernel/photo_check.py`, notebook `is4990-module-a-photo-check`) runs the models and scores them. It uses no GPU hours and does no training.
3. **The inputs:**
   - the first-check chips, tables and code datasets;
   - the weights of runs 2, 6 and 3 from their notebook outputs;
   - run 3's folds from its prep notebook.
