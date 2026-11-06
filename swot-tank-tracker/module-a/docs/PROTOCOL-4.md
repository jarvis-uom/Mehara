# Protocol for retraining run 4 (FIXED 4 Nov 2026, before any training data was prepared)

**Status:** fixed on 4 Nov 2026 after an outside review. The team leader approved it. It was fixed before any training data was prepared. Nothing below changes, except by a recorded amendment made before any score is seen.

**Aim.** Cut the false water on dark, dry land, the main weakness found by S1S2-Water (precision 60 to 66%), without losing what runs 2 and 3 gained on our reservoirs.

**The comparison** is with **run 3's photo-label control**. Run 4 changes several things together (items 1 to 5 below), so a win shows that the package works, not which part did it.

## What changes from run 3's photo-label control

1. **Photo labels, plus never-water land.** The photo label gains `(photo unsure) & land_out → land`.
   - `land_out` is ground outside the outline that JRC never saw as water, over 30 m from any water it saw. It uses no storage records.
   - **Chip lists:** run 3's lists are kept, so the comparison is fair. Those lists keep only clear photos that had a storage record within 2 days. So the *labels* use no storage records, but the *choice of dates* came from run 3.
   - **Class 2** (water under plants) gets no labels in this mode, as in the control. Water is p(land) < 0.5.
2. **S1S2-Water's training scenes join the training.**
   - **Which scenes:** the training split, without the Asian extra scenes 6, 12 and 52, which stay unseen. That leaves 30 scenes. Validation and test scenes are never trained on.
   - **Each scene is checked first:** the mask must be 10 m or finer, and Earth Engine's radar must cover at least 70% of it. Scenes that fail are listed and left out.
   - **Radar:** Earth Engine's, for each scene's own products, stored as dB × 100 (divided by 100 when read; −32768 = no data).
   - **Masks:** resampled and registered as in PROTOCOL-S1S2.
   - **Patches:** 64 patches of 512 × 512 per scene, at least 90% valid, drawn with seed 4:
     - 32 contain shoreline (5 to 95% water);
     - 32 contain dark land (at least 20% of the patch is land with VV under −14 dB).

     If a scene lacks enough of one kind, valid patches drawn at random fill the gap.
   - **Labels:** expert water becomes "water of either kind", the rest land. Invalid cells, and cells whose resampled water share is between 0.2 and 0.8, are left out.
3. **The batch mix:** every batch of 16 has 8 crops from our reservoirs, 4 Sen1Floods11 training chips and 4 S1S2-Water crops.
   - Batch position `i % 4 == 0` is Sen1Floods11, `i % 4 == 2` is S1S2-Water, and the rest are our reservoirs.
   - Run 3 had 12 reservoir crops and 4 flood crops, so our reservoirs' share falls by a third.
4. **More dry beds.** Within each reservoir's pool, a chip counts twice if its photo shows bare ground on at least 30% of the outline's cloud-free pixels, and at least half of the outline is cloud-free. This applies only to clear chips and to radar passes within 2 days of such a photo.
5. **Crop positions, fixed.** Crops are 256 × 256 everywhere. Positions are uniform over all windows that fit (`integers(0, h − 256 + 1)`), or centred on the outline with the same 80% chance as before. This fixes a bias in run 3, which put a quarter of the crops of a 512 chip exactly on each edge.

Everything else is as in run 3:
- the start, run 6 (seed s from run 6's seed s);
- the same 103 reservoirs and five folds;
- 3 seeds, 10 epochs of 300 batches, learning rate 3e-4, keeping the last epoch.

## Win rules (all four must hold)

The tests are one-sided paired Wilcoxon tests at p < 0.05, on unrounded values. Because all four must hold, there is no correction for several tests.

1. **Less false water abroad.** On S1S2-Water's 15 test scenes (PROTOCOL-S1S2 as amended), all of:
   - pooled IoU at least **68.3** (the control's 65.3 plus 3);
   - median scene IoU at least the control's **73.6**;
   - better than the control at **8 or more of the 15 scenes** (run 4 and the control are each the mean of 5 folds);
   - one-sided Wilcoxon p < 0.05.
2. **No loss on our reservoirs, pixels.** Against Dynamic World (PROTOCOL-PHOTO, the same 1,004 chips):
   - pooled IoU at least **91.2** (the control's 91.7 minus 0.5);
   - median reservoir IoU at most 1 point below the control's **90.4**;
   - low-water IoU at most 1 point below the control's **85.5**.
3. **No loss on our reservoirs, storage** (later half, as in run 3). With d = run 4's error minus the control's for each reservoir, all of:
   - the group median error is at most **8.8%** (the control's 8.3% plus 0.5);
   - the median d is at most **+0.25** points of capacity;
   - non-inferiority within 0.5 points: one-sided Wilcoxon on (the control's error + 0.5 − run 4's error) > 0, p < 0.05.
4. **No loss on floods in general.** On the Sen1Floods11 test split (90 chips), IoU at least **65.6** (the control's 67.6 minus 2).

**Reported, not rules:**
- Bolivia (15 chips);
- the 5 Asian S1S2-Water scenes;
- precision and recall;
- plant-covered chips;
- Dynamic World's flooded vegetation;
- run 3 and r6 as further comparisons;
- the GPU each fold ran on.

## How it runs

- **Laptop (downloads only):** about 2 GB of Earth Engine radar patches and the S1S2-Water masks for the 30 scenes. They become the private dataset `is4990-s1s2water-train`.
- **One private dataset `is4990-module-a-run4-inputs`:**
  - the training script, which trains photo mode only and needs no storage table;
  - run 6's weights and run 3's folds;
  - the reservoir inputs with **all storage numbers removed**;
  - the photo labels, made once in advance, so the GPU reads only radar and labels.
- **GPU:** about 5 T4-hours, one fold per session (about 1.5 hours each). It runs on whatever is available: Colab's free T4 (inputs copied to the local disk, 2 workers, each fold saved to Drive), Kaggle after the weekly reset, a teammate's own account (run by them) or Lightning AI. The same code runs everywhere, and the GPU name is logged.
- **Scoring:** with the same code as before:
  - `score_s1s2.py` (rule 1);
  - `photo_check.py` (rule 2);
  - run 3's storage scoring on the laptop (rule 3);
  - the Sen1Floods11 scoring (rule 4).
