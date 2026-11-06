# Series 2 development plan and keep rules (DEV-6, fixed 5 Nov 2026, before any development result)

Under PROTOCOL-6 part 1, section B. These runs only **choose** the recipe that part 2 will fix. They are not tests, they give no p-values, and nothing here can be claimed as a result. They use only the development data, never the fresh tests or the old tests.

## Development batch 1: training length, weight averaging, an IoU loss

Code: `kernel6-dev/retrain6.py` (run 5's `retrain5.py` with the changes below) and `run6_dev.py`. A CPU dry run passed on 5 Nov.

| Variant | What changes from run 5 | Steps |
|---|---|---|
| **control** | Nothing in the recipe (S1S2-Water scenes 39, 49, 54, 76 and 83 held out, as in every variant) | 3,000 |
| **long** | 3 times the steps (same OneCycle shape); the scored model is the weights' moving average (EMA, decay 0.999). The last weights are saved beside it but not scored | 9,000 |
| **lovasz** | As long, plus 0.5 x a Lovász hinge on water against land, per crop | 9,000 |

- **The same in every variant:**
  - folds 1 and 2 (42 held-out reservoirs), 3 seeds each, with run 5's seeds and starts;
  - the batch mix 10/4/2;
  - 24 S1S2-Water training scenes;
  - learning rate 3e-4;
  - the scheduler also steps on a batch with no label. This happened 0 times in run 5, so it changes nothing.
- **Cost:** control about 1 hour; long and lovasz about 2.5 to 3 hours each, with two folds at once on Kaggle's two T4s.

## The development scores

They are computed on the laptop, as the mean of folds 1 and 2 where a fold gives a number.

| Code | Score | Data |
|---|---|---|
| M1 | S1S2-Water pooled IoU | The 5 held-out training scenes (64 patches each) and the 5 Asian scenes, pooled over all 10 |
| M2 | Dynamic World IoU: pooled, and the median over reservoirs | The 42 held-out reservoirs' clear chips with radar dated before their later half |
| M3 | Sen1Floods11 IoU | The validation split (89 chips) |
| M4 | Storage error (% of capacity), with today's line fitted on the first quarter and scored on the second | The 42 reservoirs' first-half passes (radar gap at most 5%, storage within 2 days), paired by reservoir |

## Keep rules (against the control, or for lovasz against long)

A variant is **kept** if it gains somewhere and loses nowhere.

**Gain (at least one of):**
- M1 up by 0.5 points or more;
- M2's median up by 0.3 or more;
- M3 up by 1.0 or more;
- M4 down by 0.2 points or more.

**No loss (all of):**
- M1 down by less than 0.5;
- M2 pooled and median each down by less than 0.3;
- M3 down by less than 1.0;
- M4 up by less than 0.2.

**Order:** long is judged against the control. Then lovasz is judged against long. If long is not kept, lovasz is judged against the control, with the same rules.

**If nothing is kept,** part 2 keeps run 5's map recipe, and series 2 changes only the storage method (C1 in PROTOCOL-6).

## Later batches (written here before they run)

- **Batch 2:** terrain inputs (HAND, slope, local incidence angle) or the previous pass as extra channels. Each is judged the same way, against the best kept recipe.
- **The r6 blend for floods** (WiSE-FT, inference only): its weight is chosen on M3 alone, and it must not lose on M1, M2 or M4 by the limits above.
