# The curve study E1: candidates and choice rule (PROTOCOL-CURVE, fixed 6 Nov 2026, before any candidate is scored)

Under PROTOCOL-6 part 1, section B: the storage method for C1 is chosen on development data only. The error budget (E0, `DEV-6.md`) found that most of run 5's storage error is a bias near full, where one straight log-log line reads about 9 points of capacity too low. Only today's line (L0) has been scored on these data; no other candidate has.

## Data

- **Areas:** run 5's out-of-fold pass areas (`runs/2026-11-05-retrain-5`, column "run 5, water"), main orbit only.
- **Passes:** each reservoir's first half only (dates before `later_from` in `kernel6-dev/split_dates.csv`), with rule 3's filter: radar gap at most 5%, storage on record within 2 days.
- **Split:** fitted on the first quarter (before `q2_from`), scored on the second (`q2_from` up to `later_from`).
- **Reservoirs:** all 103 with a split. The later halves, the old tests and the fresh tests are not used.

## Score

- **Per reservoir:** the median absolute error on the second quarter, in % of capacity.
- **Overall:** the median over reservoirs.
- If a candidate cannot be fitted at a reservoir, that reservoir uses L1 for that candidate.

## Step 1: the curve

Every candidate except L0 is capped to between 0 and capacity. They are listed from the simplest to the most complex:

| Code | Curve |
|---|---|
| **L0** | Today's line: least squares of log storage on log area (`kernel3/retrain.py` `fit`), not capped. The comparison |
| **L1** | L0, capped |
| **L2** | A straight line of storage on area (least squares, not in logs), capped |
| **L3** | A robust log-log line (Theil-Sen), capped |
| **L4** | A hinged log-log line: log storage = c + b1 x + b2 max(0, x - h), x = log area. The hinge h is chosen among the first quarter's 50th, 60th, 70th, 80th and 90th percentiles of x by the least squared error, with at least 3 passes on each side; L1 if no hinge qualifies. Capped |
| **L5** | A monotone curve: isotonic regression of storage on area over the first quarter, read by linear interpolation, flat beyond the fitted range. Capped |

**The choice:** keep the simplest candidate whose overall score is within 0.1 points of the best (the one-SE rule in PROTOCOL-6).

## Step 2: smoothing over time (on the step 1 winner)

Causal only: each pass uses only passes up to its own date.

| Code | Smoothing |
|---|---|
| **S0** | None |
| **S1** | The mean of the readings at this pass and the passes in the 12 days before it |
| **S2** | The median of the readings at this pass and the passes in the 24 days before it |

The readings come from the curve fitted on the first quarter. Earlier passes of either quarter may enter the window. The choice follows the same rule, simplest first.

## Step 3: the morning passes (addendum, fixed 6 Nov 2026 before any morning area exists)

**The areas.** Run 5's fold models measure the morning passes on Kaggle (CPU), each reservoir by the fold that never trained on it, with run 5's own water rule and outline (`kernel6-morning/measure_morning.py`; on the laptop it reproduces run 5's main-pass areas to 0.1 ha).

**The filter.** Morning passes use the same filter and dates as the main passes: radar gap at most 5%, storage within 2 days, before `later_from`, with the first quarter before `q2_from`.

**What is scored.** Every candidate is scored on the same targets as steps 1 and 2: the main orbit's second-quarter passes, with the same score. All use the step 1 and 2 winner (L4, no smoothing). From the simplest:

| Code | How the morning passes are used |
|---|---|
| **O0** | Not used (the step 2 winner) |
| **O1** | One L4 curve fitted on both orbits' first-quarter passes, with the morning areas as measured |
| **O2** | As O1, with the morning areas first put on the main orbit's scale. The factor is the median, over first-quarter pairs of a main pass and a morning pass at most 6 days apart, of main area / morning area. The factor is 1 with fewer than 3 pairs |
| **O3** | O2's curve, and each main pass's reading averaged with the reading of the latest morning pass in the 6 days before it, if there is one (causal) |

- **The choice:** the same rule, the simplest within 0.1 points of the best.
- **Reservoirs with no morning passes** (Tabbowa, and any without first-quarter morning passes) use O0 under every candidate.
- **Reported beside the choice:** O2's error on the morning orbit's own second-quarter passes.

## Step 4: L4 kept from bending down (addendum, fixed 7 Nov 2026 before L4c is scored)

**Why.** An outside review (checked on 7 Nov) found that L4's upper segment is unconstrained. On the first quarter, its extra slope c2 is negative at 49 of 102 reservoirs, and the whole upper slope is negative at 18. There, more water area reads as less storage, which is physically impossible; only the cap hides part of it. E0's bias near full (the line reads too low) also points the other way.

| Code | Curve |
|---|---|
| **L4c** | L4 with c2 >= 0: the upper segment is at least as steep as the lower. For each hinge, if the unconstrained c2 is negative, that hinge's fit is the plain log-log line; this is the exact constrained least-squares solution. The hinge is chosen by the least squared error, as L4. Capped |

- **Scored** exactly as step 1 (first quarter to second, 103 reservoirs).
- **The choice:** L4c replaces L4 if its overall score is at most L4's + 0.1. Within the same margin, the curve that cannot read less storage for more area is preferred.
- **Steps 2 and 3 are not repeated.** Smoothing and the morning passes lost to L4 by 0.17 points or more; they are re-scored with L4c only beside the choice.
- **Reported beside it:** how many reservoirs have a downhill upper segment under L4 and under L4c, and the error and signed error at 85% full or more.
- **A note on wording.** "One-SE rule" here means the fixed 0.1-point margin of PROTOCOL-6, not a standard error estimated from the data (about 0.2 points over 103 reservoirs).

## Reported beside the choice (not part of it)

- The paired difference from L0 by reservoir: its median, and how many reservoirs improve.
- The error and the signed error at 85% full or more.

## The result

- The winner of steps 1 and 2 becomes C1's storage method in PROTOCOL-6 part 2, fitted there on all passes to 31 Oct 2026.
- The winner of step 3 decides how C1 uses the morning passes. If it is O0, C1 does not use them.
- Step 4 decides whether C1's curve is L4 or L4c.
- Code: `scripts/curve_study.py`. Its outputs go to `runs/e1/`.
