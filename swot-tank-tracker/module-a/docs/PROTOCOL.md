# Retraining run 1: evaluation rules, fixed before the run (2 Nov 2026, 22:25)

Written before the results, as the independent review advised. Nothing below may change after the results are seen. Any later change is a new run, with its own date.

**Question.** On reservoirs it never saw, does the radar U-Net fine-tuned with the "lake high enough" labels map water better than run 2, than a radar threshold tuned for each reservoir, and than the seasonal storage baseline?

**Design.**
- **Reservoirs:** the 107 used in the every-pass check (`passes/out/pass_flags.csv`). Nalanda's outline is fixed (the review's H6).
- **Folds:** five folds by reservoir, made by sorting by capacity and putting every fifth in the same fold. Each reservoir is scored by a model trained without it and without its storage.
- **Training:** fixed settings (10 epochs of 300 batches of 16 crops, 256 x 256, lr 3e-4; 40% clear days, 10% cloudy, 50% radar passes). One seed per fold. The last epoch is kept. Nothing is chosen on the held-out reservoirs.
- **Methods:**
  - the new model (water = open water plus water under plants);
  - run 2;
  - a VV threshold tuned for each reservoir so that its water matches the photo's on that reservoir's clear days (no storage used);
  - two baselines without a map: flat (median storage of the first half) and seasonal (median storage of first-half passes within 15 days of the same day of the year).

**Primary measure.** The storage error on the later half of each held-out reservoir's passes, read with a power law fitted on its first half (% of capacity). It is reported as the median over reservoirs, with paired Wilcoxon tests over reservoirs.

**The new model wins only if all four hold:**

1. Its storage error is lower than run 2's and lower than the tuned threshold's (each: p < 0.05, better at more than half of the reservoirs).
2. Its storage error is lower than the seasonal baseline's (p < 0.05).
3. **Less over-reading at low water:** on held-out clear days under 20% of capacity, its water share of the outline is closer to the photo's than run 2's is.
4. **No loss on expert labels:** its water IoU on Sen1Floods11's Bolivia chips is at most 2 points below run 2's (mean over folds).

**Reported, but not part of the win:**
- Spearman over all, clear-photo and cloudy-photo passes;
- the error near full;
- results by size;
- the area of water under plants;
- **pixel IoU and F1 against the storage reference**, over all passes, under 20%, at 85% and more, and on cloudy-photo passes. The reference is the rule built from the held-out reservoir's own clear photos and storage. It is a consistency score, not ground truth: it decides only spots it is sure of, and it cannot see water under plants.

**If it does not win,** report it as it is. Run 2 stays the radar model, and the switch (the fused U-Net where clear, radar under cloud) stays version 1, as the review proposes.

**Known limits:**
- the labels inherit the photo's mistakes (haze is filtered by the SWIR test, not removed);
- one seed per fold;
- cascade neighbours can fall in different folds;
- plants on the dry bed at the lowest storage are left unsure.
