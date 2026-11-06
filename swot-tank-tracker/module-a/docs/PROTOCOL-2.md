# Retraining run 2: evaluation rules, fixed before the run (2 Nov 2026, 23:35)

Written before the results of run 2, and before the results of run 1 had come in. It follows the independent review and the outside reviews. Nothing below may change after the results are seen. Run 1's rules (`PROTOCOL.md`) stay as they were, for run 1.

**Question.** On reservoirs it never saw, does the radar U-Net fine-tuned with the "lake high enough" labels (version 2) track storage better than the existing radar models, the radar thresholds and the storage baselines? And does the storage rule itself help, beyond simply training on Sri Lankan chips?

**Design (`kernel2/retrain.py`).**
- **Reservoirs:** the 107 of the every-pass check, less those whose capacity looks wrong in the cleaned sheet (over a fifth of days above 105% of capacity). The storage sheet is cleaned before labels and scores: one-day spikes and flat runs of 60 days or more below 98% full are dropped.
- **Folds:** five folds, with neighbours together (chips that overlap, or centres within 10 km).
- **Training:** two modes, each with 3 seeds averaged:
  - "rule": labels version 2, including the photo labels moved to radar passes within 5 days and 1% of capacity;
  - "photo": the control, with photo labels only and no storage.
  
  Training is otherwise as run 1: 10 epochs of 300 batches of 16 crops, lr 3e-4, the last epoch kept.
- **Methods:**
  - the new model (water = open water plus water under plants);
  - the control;
  - run 2;
  - run 6 (3 seeds averaged);
  - the VV threshold tuned per reservoir to its clear photos, on single-orbit radar;
  - a per-scene Otsu threshold in a 300 m band;
  - the flat and the seasonal storage baselines.

**Primary measure.** The storage error on the later half of each held-out reservoir's passes, read with a power law fitted on its first half (% of capacity). Paired Wilcoxon tests over reservoirs.

**The new model wins only if all five hold:**

1. **It beats the radar U-Nets:** its storage error is lower than run 2's and lower than run 6's, each with p < 0.05 and a median gain of at least 1 point of capacity.
2. **It beats the radar thresholds:** its storage error is lower than the tuned threshold's and the Otsu threshold's, each with p < 0.05 and a median gain of at least 1 point.
3. **It beats the seasonal baseline:** p < 0.05 and a median gain of at least 1 point.
4. **Less over-reading at low water:** on held-out clear days under 20% of capacity, take |map's water share - photo's share| on pixels where the photo is sure (open water or bare ground). Its per-reservoir median gap is lower than run 2's (p < 0.05).
5. **No loss on expert labels:** its water IoU on Sen1Floods11's Bolivia chips is at most 2 points below run 2's (mean over folds).

**Separate claim, "the storage rule helps":** made only if the new model's storage error is lower than the control's (p < 0.05, median gain of at least 0.5 point). If it is not, any gain is credited to training on Sri Lankan chips, not to the rule.

**Reported, but not part of the win:**
- Spearman (all, clear-photo and cloudy-photo passes) and the error near full;
- results by size;
- the label coverage by source and storage band;
- the water-under-plants area. Its go/no-go is a later, separate test with its own rules, and class 2 makes no RQ1 claim here;
- pixel IoU and F1 against the storage reference, with the share of the outline it decides. These are agreement with a storage-derived reference, not accuracy.

**Module B** may use only out-of-fold maps: each tank is mapped by the fold models that held it out.

**If it does not win:** report it as it is. Version 1 stays the switch (fused U-Net where clear, radar under cloud).

**Known limits:**
- the labels inherit the photo's mistakes, and plants on water that is never open stay unlabelled;
- the training dates overlap the test dates (a date-split check is still to do);
- the first-check radar chips are two-orbit mosaics for most clear days;
- the "last clear photo carried forward" baseline is not in this run. It needs every clear Sentinel-2 date (an Earth Engine table) and will be scored afterwards from the saved areas, by the same storage-error rule.
