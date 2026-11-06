# Retraining run 3: results (3 Nov 2026)

**Rules:** fixed before the run in `PROTOCOL-3.md` (commit 67f0ca6). The rule check is `scripts/check_protocol3.py` (`protocol3_check.csv`), after `scripts/merge_run3.py` (outputs in `runs/2026-11-03-retrain-3/`).

**What changed from run 2:**
- **Start:** run 6, whose Bolivia score is 66.0.
- **Flood chips:** a quarter of every batch is Sen1Floods11's hand-labelled training chips.
- **A fair photo control:** it is trained on clear days and on every radar pass within 5 days of a clear photo, with no storage.
- **Labels 2.1:** the dry-bed fix.
- **Sampling:** a reservoir is picked first, then one of its chips.

Everything else is as in run 2: the same 103 reservoirs and folds, the same training length, and 3 seeds.

**Where it ran:**

| Job | Where | Minutes |
|---|---|---|
| Prep | Kaggle CPU | 11 |
| Part A (folds 1 to 3) | Kaggle GPU | 262 |
| Fold 4 | Kaggle GPU | 162 |
| Fold 5 | Kaggle GPU | 90 |

That is about 8.6 GPU hours. Fold 4 also ran on Colab's free T4, as a cross-check. With the same code and seeds, it gave the same flood scores and the same water area on all 2,886 passes, to 0.00% of the outline. The results do not depend on the machine.

## Verdict by PROTOCOL-3: not a win

| Rule | Result | Holds? |
|---|---|---|
| 1. Lower storage error than run 2 and run 6 (p < 0.05, gain at least 1 point) | gain 2.4 and 1.3; better at 76% and 80% of 103; p = 7e-10 and 4e-9 | yes |
| 2. Lower than the tuned threshold and Otsu | gain 1.7 and 2.1; better at 76% and 78%; p = 2e-8 and 3e-10 | yes |
| 3. Lower than the seasonal baseline | gain 4.5; better at 82%; p = 8e-12 | yes |
| 4. Low water: water share closer to the photo's than run 6's | better at 80% of 35 reservoirs; p = 2e-4 | yes |
| 5. Bolivia water IoU at most 2 points below run 6's (66.0, so at least 64.0) | 54.4 over 5 folds (52.7 to 56.2) | **no, by 9.6 points** |
| Separate claim: the "lake high enough" rule beats the fair control (gain at least 0.5) | gain -0.1; better at 47%; p = 0.96 | **no** |

Rules 1 to 4 hold with wide margins, and run 3 clears rule 1 against run 6 more clearly than run 2 did (1.3 against 1.2). Rule 5 fails. The flood practice raised Bolivia from 44.8 (run 2) to 54.4, but the bar was 64.

## Storage error on the later half (median, % of capacity)

| Group | Reservoirs | Run 3 | Run 2's model | Fair control | Run 6 (r6) | Run 2 (r2) | Seasonal | No model |
|---|---|---|---|---|---|---|---|---|
| All | 103 | **7.8** | 7.9 | 8.3 | 10.8 | 12.5 | 14.4 | 22.5 |
| Under 5 million m3 | 39 | 11.8 | 11.0 | 11.2 | 12.8 | 13.7 | 15.2 | 26.1 |
| 5 to 30 million m3 | 34 | 7.9 | 7.9 | 7.9 | 10.1 | 10.5 | 14.4 | 17.3 |
| 30 million m3 and more | 30 | 5.3 | 5.4 | 5.0 | 6.9 | 8.8 | 13.3 | 22.7 |

**Other measures:**
- **In volume:** the median error at a reservoir is 0.83 million m3. That compares with 0.85 for run 2's model, 0.98 for the control, 1.06 for r6, 1.17 for r2, and 1.45 for the seasonal and no-model baselines.
- **Spearman with storage:** 0.87 for run 3 and the control, 0.86 for run 2's model and 0.85 for r6.
- **Near full:** the median reading is 9.1 points of capacity too low (run 2's model: 10.4; the control: 6.6). These reservoirs still read 15 points or more too low:

  | Model | Reservoirs |
  |---|---|
  | Run 3 | 31 |
  | Run 2's model | 36 |
  | r6 | 34 |
  | Control | 25 |

- **Low water:** the median gap between the map's and the photo's water share is 4.3 points of the outline. That compares with 8.6 for r6, 9.2 for r2, 3.8 for run 2's model and 4.8 for the control.
- **Against run 2's model:** no difference (gain -0.1, p = 0.53). The flood mix and the start from run 6 cost no storage skill, and they added none.

## Flood mapping (water IoU, %)

| Fold | Bolivia, run 3 | Bolivia, control | Sen1Floods11 test split, run 3 | Test split, control |
|---|---|---|---|---|
| 1 | 53.0 | 48.9 | 65.7 | 66.6 |
| 2 | 54.0 | 46.4 | 67.0 | 68.6 |
| 3 | 52.7 | 43.0 | 66.6 | 66.5 |
| 4 | 56.0 | 51.2 | 66.9 | 68.0 |
| 5 | 56.2 | 52.8 | 66.4 | 68.3 |
| **Mean** | **54.4** | **48.5** | **66.5** | **67.6** |

**References from run 2's check:**
- run 6: 66.0;
- run 2 (r2): 58.6;
- run 2's retrained model: 44.8.

**What the table shows:**
- **Bolivia** is steadier than in run 2 (folds 53 to 56, where run 2 ran from 34 to 51) and almost 10 points higher.
- **The test split** comes from the same countries as Sen1Floods11's training chips, and there run 3 keeps about 67. What drops is the unseen Bolivia flood. Training on tanks hurts how well the model carries over to a new kind of flood scene, not flood mapping as such.

## What it means

1. **Training on our own reservoirs works, now twice.** Two runs with different starts, labels and controls both bring the storage error from about 11 to 12% down to about 8%.
2. **The flood mix helped at no cost.** Bolivia rose by almost 10 points with the same storage error. But it was not enough for a 2-point limit against run 6.
3. **The "lake high enough" rule adds nothing measurable over photo labels, even against a fair control.**
   - Photo labels on clear days and on passes within 5 days do as well.
   - They need no storage records, so the same training could run on tanks with no gauge, such as the village tanks.
4. **Two limits remain:**
   - Near full, the area hardly grows, so storage there needs SWOT's height (Module B).
   - At small reservoirs, every map is about 11 to 12% off.
5. **The result is reproducible:** fold 4 came out the same on Kaggle and on Colab.

## Options for next

1. **Report runs 2 and 3 to the supervisors as they are:** a robust storage gain, not a win by our strict flood rule, and no benefit from the rule.
2. **Choose the photo-label training** as Module A's method for storage. It is simpler, needs no storage data, and is as good.
3. **A run 4 (only if worth it), with rules fixed first.** Options:
   - half flood chips, a lower learning rate or a frozen encoder;
   - asking the supervisor whether Bolivia's 15 chips, or the 90-chip test split, should be the flood check.
4. **Score run 3 against the last clear photo**, as was done for run 2 (it is reported, not part of the win).
