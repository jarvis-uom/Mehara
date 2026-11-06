# S1S2-Water check: results (4 Nov 2026)

**Rules:** fixed before any scoring in `PROTOCOL-S1S2.md` (commit a2a101d), with amendment 1 (commit c05991d). The amendment was also made before any S1S2-Water score was seen.

**What it is.** Module A's existing models were scored against expert water masks. Nothing was trained.
- **Truth:** S1S2-Water's Sentinel-1 water masks (Wieland et al., 2024).
- **Radar:** Earth Engine's, for each scene's own Sentinel-1 products. It is made exactly like our training radar, without a speckle filter.
- **Masks:** resampled to 10 m, then registered by a radar-only contrast test. The shifts were 0 to 2 pixels (`gee/scenes.csv`).

**Where it ran.** Kaggle's weekly GPU hours were used up, so it ran on Colab's free T4 (`colab/s1s2water_colab.ipynb`).
- It used the same code as the Kaggle notebook (`kernel/score_s1s2.py`).
- It took 76 minutes for 20 scenes, with no errors.
- **Outputs:** `runs/2026-11-04-s1s2water/`. The same files are on Google Drive (`MyDrive/is4990-s1s2water-check`) and in the private Kaggle dataset `is4990-s1s2water-check-results`.

## Main result: the 15 test scenes, pooled

| Method | Water IoU | F1 | Precision | Recall | Median IoU of a scene |
|---|---|---|---|---|---|
| Otsu radar threshold | 33.6 | 50.3 | 35.4 | 87.0 | 22.7 |
| r2 (Sen1Floods11 U-Net) | 52.0 | 68.4 | 52.2 | 99.1 | 67.7 |
| r6 (Sen1Floods11 U-Net, 3 seeds) | 53.0 | 69.3 | 53.3 | 99.0 | 66.9 |
| **Run 3** (fine-tuned with the "lake high enough" rule; 5 folds) | **59.6** (56.6 to 62.5) | **74.6** | 60.1 | 98.5 | 71.1 |
| **Run 3's photo-label control** (5 folds) | **65.3** (63.7 to 66.6) | **79.0** | 66.1 | 98.2 | 73.6 |

## The protocol's three questions

1. **Does fine-tuning for tanks keep water-mapping skill on ordinary water bodies?** **No loss; in fact a gain.**
   - Pooled, run 3 scores 59.6 against r6's 53.0, and the bar was at least 51.0.
   - By scene, it is better at 11 of 15 scenes, with a median gain of 1.9 and p = 0.022.
   - This does not change run 3's verdict under PROTOCOL-3 ("not a win" on Bolivia's floods).
2. **Do the U-Nets beat a simple radar threshold?** **Yes.** All four score above Otsu pooled (52 to 65, against 34). By scene, against Otsu:

   | U-Net | Better at | p |
   |---|---|---|
   | Photo-label control | 14 of 15 scenes | 0.0009 |
   | Run 3 | 12 of 15 | 0.005 |
   | r6 | 12 of 15 | 0.022 |
   | r2 | 11 of 15 | 0.064 |

3. **Run 3 against its photo-label control:** **the control maps water better.**
   - Pooled, run 3 is 5.7 points lower.
   - Run 3 is better at only 4 of 15 scenes (p = 0.03).

## Each scene (water IoU; run 3 and the control are the mean of 5 folds)

| Set | Scene | Region | Land cover | Water share | Otsu | r2 | r6 | Run 3 | Control |
|---|---|---|---|---|---|---|---|---|---|
| test | 23 | US south-west | shrubland, desert | 1% | 1 | 1 | 1 | 2 | 2 |
| test | 25 | western Russia | broadleaved forest | 1% | 2 | 68 | 65 | 73 | 75 |
| test | 29 | US south | herbaceous | 4% | 16 | 40 | 48 | 66 | 68 |
| test | 33 | Canada (Quebec) | broadleaved forest | 7% | 82 | 68 | 77 | 88 | 88 |
| test | 35 | US north-west | needleleaved forest | 4% | 5 | 52 | 67 | 63 | 62 |
| test | 47 | Colombia | rainfed cropland | 8% | 80 | 76 | 82 | 87 | 87 |
| test | 53 | Russian Far East | herbaceous | 21% | 62 | 45 | 50 | 61 | 68 |
| test | 57 | Caspian region | water bodies | 59% | 71 | 89 | 93 | 95 | 96 |
| test | 75 | Sea of Azov region | rainfed cropland | 9% | 21 | 37 | 45 | 71 | 72 |
| test | 77 | Siberia | needleleaved forest | 1% | 2 | 74 | 76 | 76 | 74 |
| test | 78 | US Great Plains | grassland | 2% | 3 | 9 | 6 | 15 | 15 |
| test | 80 | Argentina | flooded shrub and herbaceous | 14% | 23 | 84 | 84 | 86 | 87 |
| test | 82 | south-west Australia | evergreen forest | 27% | 87 | 98 | 99 | 98 | 98 |
| test | 88 | Egypt (Lake Nasser) | bare land | 22% | 50 | 31 | 32 | 32 | 39 |
| test | 89 | Qatar coast | water bodies | 55% | 71 | 72 | 71 | 68 | 74 |
| extra | 6 | Vietnam | evergreen forest | 3% | 9 | 45 | 60 | 78 | 79 |
| extra | 7 | Thailand | rainfed cropland | 3% | 5 | 43 | 46 | 66 | 72 |
| extra | 11 | Vietnam | water bodies | 38% | 93 | 93 | 96 | 98 | 98 |
| extra | 12 | India | natural vegetation | 6% | 60 | 52 | 73 | 83 | 87 |
| extra | 52 | Bangladesh | irrigated cropland | 14% | 83 | 87 | 89 | 93 | 94 |

## The extra set: 5 South and Southeast Asian scenes (pooled, reported apart)

| Method | Water IoU | F1 |
|---|---|---|
| Otsu | 41.6 | 58.7 |
| r2 | 78.0 | 87.6 |
| r6 | 85.6 | 92.2 |
| Run 3 | **92.3** (91.7 to 92.9) | 96.0 |
| Control | **93.5** (93.2 to 94.0) | 96.6 |

These scenes come from S1S2-Water's training and validation splits. They are new to our models, but they are not the dataset's own test split.

## Where the models fail

- **The errors are false water, not missed water.** Every U-Net finds 98 to 99% of the experts' water (recall), but 34 to 48% of what it calls water is not water (precision 52 to 66%).
- **Dry, dark land is the weak spot.** Every method fails on three test scenes:
  - the desert of scene 23 (IoU 1 to 2);
  - the grassland of scene 78 (5 to 15);
  - the bare land around Lake Nasser, scene 88 (31 to 39).

  Smooth dry ground and terrain shadows return as little radar as water.
- **Large lakes are mapped well:** scenes 57, 82 and 11 score 95 to 98.

## Speckle check (Sen1Floods11, water IoU; run 3 and the control are the mean of 5 folds)

| Model | Bolivia as is | after a 3 × 3 median | Test split as is | after a 3 × 3 median |
|---|---|---|---|---|
| r2 | 58.6 | 40.0 | 65.1 | 63.4 |
| r6 | 66.0 | 6.7 | 65.8 | 52.2 |
| Run 3 | 54.4 | 14.9 | 66.5 | 59.8 |
| Control | 48.4 | 19.5 | 67.6 | 61.2 |

- **Every U-Net moves by more than 2 points from the filter alone**, except r2 on the test split (-1.7). Under the protocol's reading rule, scoring S1S2-Water's own median-filtered radar would have measured the processing, not the models. Amendment 1 was needed.
- **The "as is" scores reproduce run 3's figures** (Bolivia 54.4, test split 66.5), so this run agrees with the earlier ones.
- **In practice:** the models must be given unfiltered radar, as Earth Engine supplies it.

## What it means

1. **RQ1, pixel accuracy on expert labels.** The fine-tuned Module A U-Net scores **IoU 60, F1 75** over 15 scenes worldwide that it never saw. On 5 South and Southeast Asian scenes it scores **IoU 92, F1 96**.
2. **Fine-tuning on Sri Lankan tanks improved water mapping in general,** by 7 IoU points over r6. Run 3's loss on Bolivia is specific to floods.
3. **Photo labels beat the "lake high enough" rule on pixels too** (65 against 60). Together with runs 2 and 3, that is three results pointing the same way: Module A should train with photo labels.
4. **The main weakness is false water on dark, dry land.** A run 4 could add S1S2-Water's 33 training scenes (expert labels), train with photo labels only, and add dry-bed examples from our tanks. Its rules would be fixed first.
5. **Radar processing matters:** unfiltered radar only.

## Limits

- **The scenes are not Sri Lankan tanks.** The official test split has no South Asian scene; the Asian extras come from the other splits.
- **The truth was adjusted:**
  - it was resampled from 6 to 10 m pixels to 10 m;
  - it was shifted 0 to 2 pixels by a radar contrast test;
  - our Earth Engine radar can differ slightly in position from the radar the experts drew on.
- **One scene was only partly covered:** Earth Engine's radar covers only 29% of scene 77.
- **The tests are small:** the Wilcoxon tests use 15 scenes, with no correction for testing several pairs.
