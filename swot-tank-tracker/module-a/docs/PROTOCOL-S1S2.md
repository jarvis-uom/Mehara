# Protocol: Module A on S1S2-Water's expert labels (fixed 3 Nov 2026, before any scoring)

**Status:** fixed before the job runs. The team agreed on 3 Nov 2026. Nothing below changes after the first score is seen. Anything learnt later goes into a new protocol.

## Amendment 1 (3 Nov 2026, 23:40, before any S1S2-Water score)

**Why.** The local check of the job found two things before any S1S2-Water scene was scored. The one scene it tried was left out on format before scoring. The only numbers seen were the speckle check's, on Sen1Floods11.
- **Our U-Nets fail on smoothed radar.** On Sen1Floods11's 15 Bolivia chips, r6 scores 66.0 as they are (its known score), but 6.7 after a 3 × 3 median. The water's median VV hardly moves (for example -18.0 dB before and after), so the models rely on the radar's speckle texture. S1S2-Water's radar went through exactly this filter.
- **S1S2-Water's pixel size varies.** It is 7, 9 or 10 m (for example scenes 77, 23 and 47), while our models were made for 10 m.

So S1S2-Water's own radar would measure the processing difference, not how well the models map water.

**What changes:**
1. **Input radar:** Earth Engine's `COPERNICUS/S1_GRD` for the scene's own Sentinel-1 products (its `s1_srcids`), VV and VH mosaicked, as dB × 100 in int16. This is exactly how our training radar was made (`module-a-reservoirs/scripts/export_passes.py`). It is fetched at 10 m on the scene's UTM grid, from the same corner as S1S2-Water's files.
2. **Truth:** S1S2-Water's Sentinel-1 water mask, resampled to the 10 m grid by area.
   - A 10 m cell is valid if at least half of it is valid.
   - It is water if at least half of its valid area is water.
3. **Registration:** S1S2-Water's masks were drawn on their own radar, which was geocoded with another elevation model.
   - For each scene, the mask is shifted by the whole-pixel offset (-3 to 3 in each direction) that gives the largest VV contrast: the mean Earth Engine VV outside the water minus inside it, over valid cells.
   - This uses only the radar and the mask, never a model's output. The offsets are reported.
4. **Scenes:**
   - The "int16 at 10 m" rule no longer applies, because the radar now comes from Earth Engine.
   - Scene 31 stays out: its mask is at 29 m, three times coarser than the radar.
   - A scene whose products are not in Earth Engine is left out and listed.
5. **Where:** the laptop makes the radar and the resampled truth (`scripts/fetch_gee_radar.py`, light: downloads only). They go to the private Kaggle dataset `is4990-s1s2water-gee`, and the Kaggle job reads them from there.

Everything else stays as below: the sets, the methods, the scores, the three questions, the speckle check (still run and reported, as the reason for this amendment) and the pictures.

**Purpose.** RQ1 asks how accurately we map water. Without hand labels, pixel accuracy comes from expert-labelled public data. This adds S1S2-Water (Wieland et al., 2024) to Sen1Floods11. Its scenes are lakes, rivers and reservoirs in normal conditions, closer to our tanks than flood scenes. This check only runs models that already exist; nothing is trained.

## Data

- **Main set:** S1S2-Water v1.0.1's official test split, 16 scenes.
  - Scene 31 is left out: its radar is uint16 at 29 m and is not in the documented format (dB × 100). It was seen in `README.md` before this protocol.
  - Any other scene whose radar is not int16 at 10 m is also left out and listed.
- **Extra set (reported apart):** the 5 scenes from the other splits in South and Southeast Asia: 6 (Vietnam), 7 (Thailand), 11 (Vietnam), 12 (India) and 52 (Bangladesh). None of our models ever saw S1S2-Water, so these are new to them too. They are never pooled with the main set.
- **Input:** S1S2-Water's Sentinel-1 image, VV and VH in dB (stored value ÷ 100).
- **Truth:** S1S2-Water's Sentinel-1 water mask (1 water, 0 not).
- **Scored pixels:** the valid mask is 1 and the radar is finite. Invalid pixels go into the model as the training mean (0 after scaling), as in our other tests.

## Methods

Each method's water is decided as in runs 2 and 3:

| Method | Water |
|---|---|
| r2: Sen1Floods11 radar U-Net, run 2 | p(water) > 0.5 |
| r6: Sen1Floods11 radar U-Net, run 6, 3 seeds averaged | p(water) > 0.5 |
| Run 3, retrained with the "lake high enough" rule: per fold, its 3 seeds averaged | p(land) < 0.5 (open water or water under plants) |
| Run 3's photo-label control: per fold, its 3 seeds averaged | p(land) < 0.5 |
| Otsu radar threshold | VV after a 3 × 3 median below the Otsu threshold of the scene's valid pixels (run 2's `otsu()`: 256 bins over -30 to 5 dB) |

- The scenes are cut into 1024 × 1024 tiles with a 64-pixel margin on each side; only each tile's centre is kept.
- For run 3 and the control, each of the 5 folds is scored on its own. A method's score is the mean over the 5 folds, with the range.

## Scores

- **For each scene:** water IoU, F1, precision and recall.
- **Pooled over the scenes of a set:** the same four, from the summed pixel counts. **The main result is the pooled water IoU and F1 on the main set.**
- **Paired Wilcoxon tests** on the per-scene IoU of the main set (two-sided). Run 3's and the control's per-scene IoU is the mean of their 5 folds.
  1. Run 3 against r6;
  2. run 3 against the control;
  3. each U-Net against Otsu.

## Questions fixed now

1. **Does retraining for tanks keep water-mapping skill on ordinary water bodies?** The answer is "no loss" if run 3's pooled IoU is at least r6's minus 2 points (the same margin as PROTOCOL-3's rule 5), and "a loss" if not.
   - This does **not** change run 3's verdict under PROTOCOL-3. That verdict stays "not a win".
2. **Do the U-Nets beat a simple radar threshold?** Pooled IoU higher than Otsu's, with the Wilcoxon p.
3. **Run 3 against the control:** the pooled IoU difference and the p, as a description.

## Speckle check

S1S2-Water's radar went through a 3 × 3 median speckle filter (SNAP); our training radar did not.

- **Test:** Sen1Floods11's test split (90 chips) and Bolivia (15 chips) are scored with every model, as they are and after a 3 × 3 median on VV and VH (no-data pixels stay no data).
- **Reading rule:** a model whose IoU moves by more than 2 points from the filter alone has its S1S2-Water score read with that shift in mind. The scores are not corrected.

## Reported, not part of the answers

- each scene's land cover and water share;
- small preview pictures (truth, r6, run 3 fold 1, Otsu) for every scene.

## Where it runs

A private Kaggle GPU notebook, `temp/s1s2water-check/kernel/score_s1s2.py`.

- It reads each scene's three radar files from Zenodo, one at a time, and deletes them after scoring.
- It writes each scene's counts as soon as the scene finishes.
- The weights come from the run 2, run 6 and run 3 notebooks' outputs. Sen1Floods11 comes from the private dataset `is4990-sen1floods11-radar`.
