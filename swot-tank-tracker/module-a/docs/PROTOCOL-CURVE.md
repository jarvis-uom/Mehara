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

## Step 5: one season-aware correction (addendum, fixed 13 Nov 2026 before any candidate is scored)

**Why.** The 9 Nov correction (`DEV-6.md`) showed that the problem near full is scatter: there the water area barely changes while storage still does, so the area alone cannot tell 85% full from 100%. The date is allowed information (PROTOCOL-6 part 1), and so is the reservoir's own storage history. T1's window, November to January, is the season when the reservoirs are most often near full. The option was recorded on 9 Nov as one try with a hard stop; the team chose it on 13 Nov. None of the candidates below has been scored on any data.

**Data and score.** Exactly as steps 1 and 4: run 5's out-of-fold areas on the main orbit, each reservoir's first half, the curve fitted on the first quarter and scored on the second, the median absolute error in % of capacity per reservoir, then the median over the 103 reservoirs. The comparison is L4c (5.64).

**The seasonal value m(d)** of a reservoir for a date d: the median of its recorded daily storage on the history days whose day of the year is within 15 days of d's, leaving out every day within 182 days of d. So it comes from other years only, and it carries no "last known storage".
- The history is the reservoir's daily storage record before `q2_from` (in part 2: to 31 Oct 2026), as read and cleaned by rule 3's code (`kernel3/retrain.py` `sheet`). Every recorded day is used, not only the pass days. Days of the year are compared around the year's end (1 January is 1 day from 31 December).
- If no history day qualifies, the date has no seasonal value, and every candidate reads L4c for it.

**The candidates,** from the simplest. All are capped to between 0 and capacity, and all read L4c wherever they cannot be fitted.

| Code | Reading |
|---|---|
| **Z0** | L4c (the comparison) |
| **Z1** | A blend everywhere: (1 - w) x L4c's reading + w x m(d), with one weight w per reservoir |
| **Z2** | The same blend, only where L4c's own reading is 60% of capacity or more; L4c's reading below that |
| **Z3** | A seasonal term inside the curve: L4c's form plus one yearly wave, log storage = c + b1 x + b2 max(0, x - h) + p sin(2 pi t) + q cos(2 pi t), x = log area, t = day of the year / 365.25 |

- **The weight w (Z1, Z2)** is fitted on the first quarter only, by least squares of the true storage on the blend, and clipped to between 0 and 1: w = sum((s - c)(m - c)) / sum((m - c)^2), where c is L4c's capped reading and m the seasonal value of each first-quarter pass. Z1 uses every first-quarter pass that has a seasonal value; Z2 only those where c is 60% of capacity or more. With fewer than 5 such passes, w = 0.
  - Both c and m are in-sample or near it on the first quarter (the curve was fitted there; m leaves out the 182 days around each pass). That is accepted: the score is taken on the second quarter.
- **Z3** is fitted on the first quarter by least squares, with the hinge chosen as in L4c (the same percentiles, at least 3 passes on each side, the least squared error) and b2 >= 0 kept as in L4c (a hinge whose unconstrained b2 is negative is fitted without the hinge term). Where no hinge qualifies, it is the plain log-log line plus the wave. It needs at least 12 first-quarter passes that span at least 365 days; otherwise the reservoir reads L4c.
- **The 60% limit** is fixed here from what is already known (9 Nov: by L4c's own reading, the absolute error is 3.3 below 60% and 6 to 9 between 60 and 95%). It is not tuned.

**The rule (the hard stop).** The candidate with the best overall score is looked at first; if a simpler candidate is within 0.1 points of it, the simplest such candidate is taken instead (Z1, Z2, Z3 in that order). It is **kept only if both** hold:
- its overall score is at least 0.1 points better than L4c's on the same reservoirs;
- reservoir by reservoir, a one-sided Wilcoxon on (L4c's error - the candidate's error) > 0 has p < 0.05.

Otherwise nothing is kept, **C1's storage method stays L4c on the main orbit with no smoothing, and no further curve candidate is tried before T1.**

**If one is kept,** it becomes C1's storage method in part 2: the curve and the weight fitted once per reservoir on all history passes to 31 Oct 2026, the seasonal value taken from the daily storage record to that date, nothing refitted in the window.

**Stated openly, before the run.**
- Development has at most 2 years of storage history before the second quarter (the record starts on 1 Oct 2020), so the seasonal value rests on one or two earlier years. Part 2 would have 6. Development may therefore undervalue the seasonal value; it cannot be made to look better than it is.
- The second quarter has been scored many times with other curves. This is a choice on development data, and nothing can be claimed from it.
- The margin is 0.1 points while the overall score's own standard error is about 0.2; this is why the paired test is required as well.

**Reported beside the rule (not rules):**
- each candidate's Hodges-Lehmann gain over L4c with its interval, and how many reservoirs are better and worse;
- the fitted weights (median, how many are 0), and how many reservoirs fell back to L4c;
- the error on the second quarter's November to January passes, and by L4c's own reading (under 60%, 60 to 85%, 85% and over);
- the seasonal value alone, read as a method;
- for the taken candidate: the T1 scorer's development trial (`score_t1.py --dev` rules: gap-free passes, T1's truth rule).

Code: `scripts/curve_season.py`. Its outputs go to `runs/e1/` (`e1_season_*.csv`).

## Reported beside the choice (not part of it)

- The paired difference from L0 by reservoir: its median, and how many reservoirs improve.
- The error and the signed error at 85% full or more.

## The result

- The winner of steps 1 and 2 becomes C1's storage method in PROTOCOL-6 part 2, fitted there on all passes to 31 Oct 2026.
- The winner of step 3 decides how C1 uses the morning passes. If it is O0, C1 does not use them.
- Step 4 decides whether C1's curve is L4 or L4c.
- Step 5 decides whether C1 adds a seasonal correction to L4c. **Result (13 Nov, `DEV-6.md`): none is kept; C1 stays L4c on the main orbit with no smoothing, and no further curve is tried before T1.**
- Code: `scripts/curve_study.py`. Its outputs go to `runs/e1/`.
