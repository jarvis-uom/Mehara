# Retraining run 2: results (3 Nov 2026)

**Run:** on Kaggle from 23:31 to 10:35 (659 minutes; https://www.kaggle.com/code/mehararothila/is4990-module-a-retrain-2).

**Where things are:**
- outputs in `runs/2026-11-03-retrain-2/` (the 30 models, about 730 MB, are git-ignored);
- the win rules, fixed before the run, in `PROTOCOL-2.md` (commit ea15cad);
- the rule check from `scripts/check_protocol2.py` (`protocol2_check.csv` in the run folder).

**What changed from run 1 (the trial):**
- **Reservoirs:** 103. Four were left out for a doubtful capacity: Ellewela, Morana, Peramaduwa tank and Yan Oya.
- **Folds:** five, keeping neighbours together. The largest neighbour group has 11 reservoirs.
- **Labels, version 2:** more land, a cloud-shadow guard, and photo labels moved to radar passes within 5 days and 1% of capacity.
- **Storage sheet:** cleaned of spikes and flat days.
- **Seeds:** 3 per fold, averaged.
- **New comparisons:** a photo-label control, plus run 6 (3 seeds), Otsu and the tuned threshold. The control trains on clear days only, with the photo's labels: no storage, no cloudy days and no radar passes (`kernel2/retrain.py`, `MIX`). The rule model also trains on cloudy days and on radar passes.

## Verdict by PROTOCOL-2: not a win

| Rule | Result | Holds? |
|---|---|---|
| 1. Lower storage error than run 2 and run 6 (p < 0.05, gain at least 1 point) | gain 1.7 and 1.2; better at 79% and 77% of 103; p = 2e-10 and 1e-7 | yes |
| 2. Lower than the tuned threshold and Otsu | gain 1.6 and 2.5; better at 71% and 80%; p = 1e-7 and 2e-10 | yes |
| 3. Lower than the seasonal baseline | gain 4.7; better at 82%; p = 3e-11 | yes |
| 4. Low water: water share closer to the photo's than run 2's | better at 83% of 35 reservoirs; p = 2e-4 | yes |
| 5. Bolivia water IoU at most 2 points below run 2's (mean over folds) | 44.8 against 58.6 (folds 34.4 to 51.1) | **no, by 11.8 points** |
| Separate claim: the "lake high enough" rule beats the photo-label control (gain at least 0.5) | gain -0.3; better at 41%; p = 0.31 | **no** |

The retrained model reads storage far better than every other method, and rules 1 to 4 hold with wide margins. But rule 5 fails clearly. On Sen1Floods11's Bolivia flood chips (expert labels), its water IoU fell to 44.8, while the model it started from scores 58.6 and the limit was 56.6. The separate claim fails too: photo labels alone do as well as the rule.

## Storage error on the later half (median, % of capacity)

| Group | Reservoirs | Retrained | Photo-label control | Run 2 (r2) | Run 6 (r6) | VV tuned to the photo | Otsu | Seasonal | No model |
|---|---|---|---|---|---|---|---|---|---|
| All | 103 | **7.9** | 8.5 | 12.5 | 10.8 | 11.8 | 11.6 | 14.4 | 22.5 |
| Under 5 million m3 | 39 | 11.0 | 11.5 | 13.7 | 12.8 | 13.1 | 14.7 | 15.2 | 26.1 |
| 5 to 30 million m3 | 34 | 7.9 | 7.5 | 10.5 | 10.1 | 11.2 | 11.5 | 14.4 | 17.3 |
| 30 million m3 and more | 30 | 5.4 | 5.2 | 8.8 | 6.9 | 10.2 | 9.3 | 13.3 | 22.7 |

Spearman with storage over all passes: retrained 0.86, control 0.87, r6 0.85, r2 0.81, tuned threshold 0.79 and Otsu 0.78.

**What this means:** run 1's gain holds under stricter folds, a cleaned sheet and 3 seeds. It even grew: 7.9 against 12.5 here, where run 1 had 8.9 against 12.1. Run 1's gain was not luck or a fold leak.

## The dry lake bed

On clear low-water days at 35 reservoirs, we took the gap between each model's water share of the outline and the photo's (median):

| Model | Gap (points of the outline) |
|---|---|
| Retrained | **3.8** |
| Photo-label control | 5.0 |
| VV tuned to the photo | 5.2 |
| Otsu | 6.6 |
| Run 6 | 8.6 |
| Run 2 | 9.2 |

The retrained model is closer to the photo than run 2 at 83% of these reservoirs (p = 2e-4). It is no different from the control (p = 0.54).

## Near full

The reading still flattens. Near full, the median reading is 10.4 points of capacity too low (run 2: 11.1; control: 7.6). These reservoirs still read 15 points or more too low:

| Model | Reservoirs |
|---|---|
| Retrained | 36 |
| Run 2 | 42 |
| Run 6 | 34 |
| Control | 23 |
| Otsu | 49 |

The retrained model is better than run 2 near full at 65% of reservoirs (p = 6e-4), but the limit remains.

## Against the last clear photo

Scored with `scripts/score_last_clear.py --kernel kernel2`; the outputs are in `out/last_clear/2026-11-03-retrain-2/`.

**Self-check:** at the 86 reservoirs that kept every pass, the new model's error here equals the run's own. At the other 17, 124 early passes had no clear photo before them yet, and every method is scored on the shorter set.

| Storage error, later half (median, % of capacity) | All passes | Photo 10 days old or less | Photo over 30 days old |
|---|---|---|---|
| Retrained (run 2) | **7.9** | 7.9 | 8.4 |
| Photo-label control | 8.1 | **7.4** | **7.7** |
| Run 2 (r2) | 12.3 | 11.2 | 10.9 |
| Run 6 (r6) | 10.6 | 10.3 | 8.9 |
| Last clear photo, curve from photo days | 12.9 | 9.5 | 19.5 |
| Seasonal | 14.2 | 13.7 | 13.6 |
| No model | 22.5 | 23.0 | 21.2 |

**Paired tests against the last clear photo** (curve from photo days). Each cell gives the median gain, the share of reservoirs where the first method is better, and p (Wilcoxon):

| First method | All passes | Photo 10 days old or less | Photo over 30 days old |
|---|---|---|---|
| Retrained | 4.7, 81%, p = 2e-10 | 1.6, 68%, p = 4e-5 | 8.7, 86% of 85, p = 9e-12 |
| Run 2 (r2) | 1.6, 58%, p = 0.02 | -0.9, 48%, p = 0.53 (no difference) | 6.1, 79%, p = 7e-8 |
| Seasonal | 0.0, 50%, p = 0.31 (no difference) | -2.5, 31%, p = 5e-5 | 4.2, 68%, p = 0.001 |

- At large reservoirs (30 million m3 and more) with an old photo, the retrained model is off by 6.4 against 25.8 for the photo.
- **A limit:** at small reservoirs (under 5 million m3) with an old photo, the seasonal average (9.6) is still the best. The control gives 9.9 there, and the retrained model 10.9.
- **Picture:** `out/last_clear/2026-11-03-retrain-2/example.png`. The two reservoirs were picked by rank:
  - Kantale, the middle by the radar's gain: radar 2% against 7% for the photo;
  - Parakrama Samudraya, the lower quarter: 4% against 6%.

## Bolivia, fold by fold (water IoU, %)

| Fold | Run 2 (r2) | Run 6 (r6, 3 seeds) | Retrained | Photo-label control |
|---|---|---|---|---|
| 1 | 58.6 | 66.0 | 43.7 | 38.1 |
| 2 | 58.6 | 66.0 | 50.2 | 44.5 |
| 3 | 58.6 | 66.0 | 34.4 | 35.5 |
| 4 | 58.6 | 66.0 | 51.1 | 41.6 |
| 5 | 58.6 | 66.0 | 44.7 | 49.8 |
| **Mean** | **58.6** | **66.0** | **44.8** | **41.9** |

Both models trained on our tank labels lose flood-mapping skill, whichever labels they use. Run 1 lost 2.5 points. Run 2, with more "land" labels, loses about 14. The models specialise: they read tank storage better, but they map flood water worse.

## The labels (version 2)

Average share of the outline in each class (`label_coverage.csv`, on up to 3 chips per source and reservoir):

| Label source | Storage | Land | Water | Unsure |
|---|---|---|---|---|
| Clear photo | 0 to 20% | 27% | 21% | 52% |
| Radar pass with a photo within 5 days | 0 to 20% | 40% | 16% | 45% |
| Radar pass, the rule alone | 0 to 20% | 14% | 5% | 81% |
| Cloudy photo, the rule alone | 0 to 20% | 8% | 16% | 76% |
| Clear photo | 85% and more | 0% | 91% | 9% |

At low storage, half or more of the bed is still unsure, and on the passes that only the rule labels, 81% is. This may be why the rule adds nothing measurable over the photo labels.

## What it means

- **Fine-tuning on our own reservoirs works.** It reads storage within 7.9% of capacity, against 12.5% for the model it started from, 14.4% for the seasonal average and 12.9% for the last clear photo. With a photo over a month old, it is 8.4% against 19.5%.
- **The gain does not come from the "lake high enough" rule.** The control does as well, and better near full, though it trains only on clear-day photo labels (about 1,400 scenes, with no storage and no radar passes). So neither the rule's labels nor the extra radar passes and cloudy days added skill we can measure. The independent review pointed out that the two models do not see the same scenes. A clean test of the rule needs a control trained on the same passes, with photo labels where a photo is near.
- **The cost is flood skill.** On Bolivia, both retrained models fall far below the model they started from. By the rules we fixed in advance, that makes run 2 not a win.
- **Two limits remain:**
  - the flattening near full (36 of 100 reservoirs are still 15 points or more too low);
  - small tanks under weeks of cloud, where the seasonal average is still best.

## Options for next

1. Report run 2 as it is to the supervisors: not a win by its rules, with the storage gain and the trade-off.
2. A run 3 that keeps flood skill, with its rules fixed first. Options:
   - mix Sen1Floods11's hand-labelled chips into every batch;
   - start from run 6 (Bolivia 66);
   - use a lower learning rate or fewer epochs.
3. Prefer the simpler photo labels, unless a clean test (a control on the same passes) shows the rule helps.
4. Near full, use SWOT's water height (Module B) for the top of the range.
5. **From the independent review, before any run 3:**
   - fix the dry-bed label bug (`labels.py:113`): plants on a fully dry bed stay unsure when the photo shows no water at all;
   - sample a reservoir first, then a chip, so the big reservoirs don't dominate training;
   - write the tables after every fold, and keep the run well under Kaggle's 12 hours (this run took 659 of 720 minutes);
   - bundle the 15 Bolivia chips with the code, instead of downloading them during the run;
   - report errors in million m3 as well as % of capacity, and the signed error near full as well as its size;
   - add S1S2-Water's expert labels as a second outside test, and a date-split check (train before 2024, test after).
