# Feasibility check results (25 Oct 2026)

One question: can we build the dataset? Part A (SWOT) only. Part B (Roadside) is on hold and was not run.

## A. SWOT Tank Water Tracker

**Verdict: GO**, with the changes listed below.

Study box 80.40–80.80E, 8.10–8.60N (east of pass 273's blind strip). SWOT passes 273 and 092, data from July 2023 to 11 October 2026.

| Check | Numbers | Result |
|---|---|---|
| A1. Tanks in SWOT's lake list | 402 SWOT lakes in the box. Tanks of 6.25 ha or more with a lake ID: 270 of 469 (58%, OpenStreetMap outlines); Irrigation Department tanks: 43 of 50 (86%). By size: 34% at 6.25–10 ha, 53% at 10–25 ha, 65% at 25–100 ha, 100% over 100 ha. | **RISK** |
| A2. Usable heights | 26,641 readings on 400 lakes. Best quality flag only: 10 of 23 tanks of 25 ha or more have 25+ readings. Allowing the "suspect" flag: 23 of 23 (median 75 readings), and 103 of 104 tanks of 6.25–25 ha have 10+. Clear yearly rise and fall: median 3.0 m (25 ha+), 2.3 m (6.25–25 ha). | **RISK** |
| A3. Right level? | SWOT minus gauge: Hurulu Wewa −0.26 m, Mahakanadarawa +0.34 m, Nuwara Wewa +0.77 m, Nachchaduwa −1.14 m (readings 13–20 days apart). No big datum error. Same-day gauge readings are needed to split real change from a fixed offset. | **OK** (sanity check) |
| A4. Water area from Sentinel-2 | Only 17% of Sentinel-2 dates are clear over a tank: 0–3 a month, fewest in November–February. Area against height (Spearman): Nuwara Wewa 0.70 (only 5 pairs); village tanks 0.68 (after dropping one bad SWOT reading) and 0.22. Green plants cover 49–81% of the village tanks' surface. | **RISK** |
| A5. Training data | Sen1Floods11 hand-labelled set: 446 chips, 1.75 GB, open. S1S2-Water: open (CC BY 4.0), but 165 GB in 7 parts of about 30 GB. | **PASS** |

**What this means for the proposal**
1. **Study area:** keep the box east of Anuradhapura city and pick a cascade whose tanks have lake IDs (list: `A_lake_ids.txt`). Big and medium tanks are solid: Nachchaduwa, Hurulu Wewa, Nuwara Wewa, Mahakanadarawa and Manankattiya each have 50–70 best-quality readings.
2. **Village tanks:** SWOT gives them heights, but marks almost all of them "suspect", and a few readings are metres off. Add outlier screening, test the suspect readings against gauges as part of RQ2, and keep the SWOT pixel cloud for tanks without a lake ID (move it to core if the chosen cascade needs it).
3. **Water area:** a plain water index fails where plants cover the water, and clear optical scenes are rare. So Sentinel-1 and the "water under plants" class are core, not extras. Fallback for village tanks: SWOT heights with Imbulana's (2022) depth–area–capacity formula (reading notes, paper 2).
4. **Ground truth:** save the Irrigation Department sheet every day from now (`temp\data-checks\check_access.py`), and request daily levels since July 2023. The next SWOT passes here are about 25 Oct (pass 092) and 1 Nov (pass 273).
5. **Training data:** use the Sen1Floods11 hand-labelled set and one part of S1S2-Water, not all 165 GB.

**Charts**
- [A_swot_heights.png](A_swot_heights.png): SWOT heights for 5 big and 5 village tanks
- [A_area_vs_height.png](A_area_vs_height.png): Sentinel-2 water area against SWOT height
- [A_tank_images.png](A_tank_images.png): true-colour views of 3 tanks at high and low water

**Tables:** `A_lake_ids.txt`, `A_pld_lakes.csv`, `A1_tanks.csv`, `A2_heights_by_lake.csv`, `A3_gauge_vs_swot.csv`, `A4_pairs.csv`, `A4_village_cover.csv`. Scripts are in `..\work\`.

**Notes:** a good reading has quality flag 0, crossover flag 1 or less, and dark water 10% or less (Köhn & Nielsen 2026). A usable reading also allows quality flag 1. Tank size is SWOT's reference area in A2 and the OpenStreetMap outline in A1. Water is NDWI > 0 inside the tank outline, on scenes with under 30% cloud within 500 m and 5% or less on the tank, paired with SWOT within 3 days.

## Decision (25 Oct): option B, the 5 big reservoirs plus a group of village tanks

**Big reservoirs:** Hurulu Wewa, Mahakanadarawa, Manankattiya, Nachchaduwa and Nuwara Wewa. All five have a SWOT lake ID, 53–71 best-quality readings and a daily Irrigation Department gauge. They give firm results, and their gauges show how far SWOT's "suspect" readings can be trusted.

**Village tanks:** 49 tanks of 6.25–25 ha near the big five pass the SWOT filters (40+ usable readings, a yearly rise and fall of 1 m or more, few bad readings). Mahakanadarawa has the most nearby (19), so the shortlist is 8 tanks 0.4–5 km from it, with a mix of water types on a clear February–March 2026 image:

| No | SWOT lake ID | Outline (ha) | Water | Usable readings |
|---|---|---|---|---|
| 1 | 4580005172 | 5.2 | open | 73 |
| 2 | 4580010522 | 4.6 | open | 73 |
| 3 | 4580006282 | 26.6 | mixed | 73 |
| 4 | 4580010542 | 29.2 | mixed | 86 |
| 5 | 4580011762 | 15.0 | mixed | 77 |
| 6 | 4580011422 | 31.1 | plant-covered | 90 |
| 7 | 4580007392 (Bulankulama Lake) | 31.7 | plant-covered | 47 |
| 8 | 4580043692 | 42.6 | plant-covered | 89 |

Map: [A_study_area_map.png](A_study_area_map.png). Positions, distances and plant shares: `A6_village_shortlist.csv`.

**Before the proposal is final:** check on a cascade map that the chosen tanks share one cascade, and get their local names (most are unnamed in OpenStreetMap). Tanks 1 and 2 are close to SWOT's 6.25 ha size limit, which makes them a useful test of that limit.

## B. Roadside Tree & Power-Line Risk Map

On hold (25 Oct). Not run.
