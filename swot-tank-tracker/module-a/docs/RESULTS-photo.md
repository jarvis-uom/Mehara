# Photo check (step 2): results (4 Nov 2026)

**Rules:** `PROTOCOL-PHOTO.md`, fixed before any scoring (b426861).

**Where it ran:** the Kaggle CPU notebook `is4990-module-a-photo-check`. It took 150 min and used no GPU hours.

**Outputs:** `runs/2026-11-04-photo-check/`, with `pooled.csv`, `tests.csv`, `flooded_vegetation.csv`, `chips.csv` and the counts for each reservoir.

**What was scored:**
- 1,571 clear chips at 103 Sri Lankan reservoirs. Each reservoir was mapped by the run 3 models of its own fold.
- **Reference B:** Dynamic World, independent of our models. It covers 1,004 chips at all 103 reservoirs, and it is the main reference.
- **Reference A:** the photo's water test, which our training labels came from. It covers all 1,571 chips.

These are **agreement scores, not ground truth.**

## Main result: Dynamic World (reference B), all chips pooled

| Method | Water IoU | F1 | Precision | Recall | Median IoU of a reservoir |
|---|---|---|---|---|---|
| Otsu | 79.9 | 88.9 | 85.6 | 92.3 | 79.8 |
| r2 | 88.6 | 93.9 | 91.3 | 96.7 | 87.7 |
| r6 | 90.2 | 94.9 | 93.4 | 96.4 | 89.6 |
| **Run 3** | **93.0** | **96.4** | 98.8 | 94.1 | 91.3 |
| Photo-label control | 91.7 | 95.7 | 99.4 | 92.2 | 90.4 |

## Groups (Dynamic World, water IoU)

| Group | Reservoirs (chips) | Otsu | r6 | Run 3 | Control |
|---|---|---|---|---|---|
| Low water (under 20% full) | 30 (68) | 43.9 | 74.9 | **86.9** | 85.5 |
| Plant-covered (plants ≥ 20% of the outline) | 79 (367) | 71.0 | 85.6 | **92.0** | 90.5 |
| Under 5 million m³ | 39 (380) | 49.2 | 83.2 | **85.2** | 83.8 |
| 5 to 30 million m³ | 34 (335) | 79.5 | 88.4 | **91.0** | 89.4 |
| 30 million m³ and more | 30 (289) | 82.9 | 91.1 | **94.0** | 92.8 |

## The protocol's questions (Wilcoxon over the 103 reservoirs)

1. **Does run 3 agree with Dynamic World better than r6? Yes.**
   - Run 3 is better at 76% of reservoirs, with a median gain of 1.7 (p = 1e-7).
   - The photo test gives the same answer: better at 88% of reservoirs (p = 5e-14).
2. **Does the photo-label control agree better than run 3? It depends on the reference:**
   - **Dynamic World: no.** The control is better at only 9% of reservoirs, with a median of -1.4 (p = 1e-13).
   - **The photo test: yes,** better at 79% (p = 2e-10). That is expected, because the control learned from this same test.
   - The two models differ mostly in recall: the control is more cautious. On S1S2-Water, the control was better. So between the two label types, the evidence is now mixed and the differences are small: 1 to 6 points.
3. **How much lower is agreement at low water and on plant-covered chips?** Against Dynamic World:
   - **Low water:** run 3 drops from 93.0 to 86.9, r6 to 74.9 and Otsu to 43.9. The fine-tuning helps most here, which is the dry-bed problem it was made for.
   - **Plant-covered chips:** run 3 drops only slightly, to 92.0, while r6 falls to 85.6.
   - Run 3 beats Otsu at 99% of reservoirs (p = 1e-18).

## Dynamic World's flooded vegetation (a first look at water under plants)

Share of Dynamic World's flooded-vegetation pixels that each method calls water (3.6 million pixels):

| Method | Share called water |
|---|---|
| r2 | 42% |
| r6 | 42% |
| Run 3 | 21% |
| Control | 17% |
| Otsu | 29% |

The fine-tuned models call less of it water. In Sri Lanka, Dynamic World's flooded vegetation includes paddy, so this is not a score. The vegetated-water class still needs its own test (step 4).

## What it means

1. **On our own reservoirs, the maps agree closely with an independent map:** run 3 scores **IoU 93, F1 96** against Dynamic World. That is clearly better than before fine-tuning (90) and than Otsu (80).
2. **Fine-tuning helps most where it was meant to:** at low water (87 against 75) and on small and plant-covered tanks.
3. **Photo labels against the storage rule is now mixed.** The control was better on S1S2-Water (65 against 60) but slightly worse against Dynamic World here (92 against 93). Photo labels stay the practical choice: they need no storage records, and the gap is small either way.

## Limits

- Dynamic World is itself a model made from the photo: agreement, not truth.
- The radar is 0 to 2 days from the photo, so the water can change slightly between them.
- Only clear days are scored. Under cloud there is no photo; the storage check and S1S2-Water's scenes cover that.
- Dynamic World exists only for photo tiles under 35% cloud, which gives 1,004 of 1,571 chips.
