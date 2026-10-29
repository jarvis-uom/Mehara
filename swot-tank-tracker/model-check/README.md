# Model check: will the three models work? (28 October 2026, corrected after review)

Before the proposal went in, we tested the models of Member B and Member C on real data. An outside review of our first run found real mistakes; we checked each point and fixed what was wrong. **All numbers below come from the corrected run.** The changes are listed at the end.

The scripts are in this folder, and the result tables in `results/`. The data they read is not included (see "Run it again").

## Short answer

| Member | Model | Trained in this check? | What we found | Our view |
|---|---|---|---|---|
| A | U-Net water map | No | Simple rules missed most of the water at a plant-covered tank, so there is real room for a model | Should work for open water. "Water under plants" is the research risk, and labels are the hard part |
| B | SWOT trust screen (gradient boosting) | Yes: 84 reservoirs, 4,044 SWOT readings | Keeping about half the readings, it ties SWOT's strict flag (12 cm). Keeping 85%, its error is 15 cm, against 21 cm for SWOT's loose flag | Works. Its value is keeping more readings at a low error |
| B | Storage model: SWOT height (and water area) → % full | Yes: 83 reservoirs, 3,374 screened readings | On reservoirs it never saw: 8.4% of capacity against 10.8% for one formula (height only), and 6.3% against 9.5% with the water area too | Works. This became B's main model on 29 October |
| C | Storage forecast (gradient boosting and an LSTM) | Yes: 117 reservoirs, 197,046 daily values | Gradient boosting beat "no change" by 10–22% and a seasonal rule by 9–12%. The LSTM came close behind | Works, with a modest gain |

## Member C: storage forecast

### Data

- **Storage:** the Irrigation Department's daily storage sheet (its hidden tabs): 117 reservoirs (74 major ones since October 2020 and 43 medium ones since April 2023), with 197,046 daily values up to 27 October 2026 and each reservoir's capacity.
- **Gaps:** the major reservoirs have no Yala 2021 (April–September 2021). The medium ones have no Maha 2023/24.
- **Days with no change:** on 59% of days (median reservoir), storage did not change from the day before.
- **The sheet's "long-term average" column is not used.** It is the mean of every season in the sheet, test years included: 80% of its rows equal that mean exactly. Instead, each reservoir's seasonal average is built from the training years only (before 1 April 2025), for each day of the year, smoothed over ±15 days.
- **Rain:** daily ERA5 rain at each reservoir, from the free Open-Meteo archive. The real project would use CHIRPS.

### How we tested

- **Target:** the change in storage, as a share of capacity, 7, 14 and 28 days after each day.
- **Train:** targets before 1 April 2025. **Test:** forecasts from 1 April 2025 to 27 October 2026 (Yala 2025, Maha 2025/26, Yala 2026 and the start of Maha 2026/27).
- **Error:** mean absolute error (MAE), in % of each reservoir's capacity.
- **Model inputs:**
  - storage today, and its change over the last 1, 3, 7, 14 and 28 days;
  - the seasonal average today and on the target day;
  - the day of the year, capacity, and major or medium;
  - rain over the last 3, 7, 30 and 90 days.
- **Gradient boosting:** LightGBM, 400 trees, L1 loss, no tuning.
- **LSTM:**
  - Inputs: the last 60 days of storage, its daily change, the seasonal average, rain and the season, plus capacity, type and the seasonal change to each target day.
  - Output: how far the change will differ from the seasonal change.
  - Training: 48 hidden units, dropout 0.2, L1 loss, early stopping on the last training year.

### Results (MAE in % of capacity; lower is better)

| Method | 1 week | 2 weeks | 4 weeks |
|---|---|---|---|
| No change | 4.5 | 7.6 | 12.5 |
| Seasonal average (storage = the past-years average on the target day) | 16.7 | 16.7 | 16.7 |
| Seasonal change (today's storage plus the past-years average's change) | 4.6 | 7.3 | 10.8 |
| Gradient boosting, without rain | 4.1 | 6.7 | 9.9 |
| **Gradient boosting, with past rain** | **4.0** | **6.6** | **9.8** |
| The same, on reservoirs it never saw (5 folds by reservoir) | 4.0 | 6.7 | 9.9 |
| The same, if the coming weeks' rain were known (an upper limit) | 3.8 | 6.1 | 9.0 |
| LSTM | 4.3 | 6.9 | 10.0 |

- **Gradient boosting** is 10%, 13% and 22% better than "no change", and 12%, 9% and 9% better than the seasonal change.
- **The LSTM** is 4%, 9% and 20% better than "no change", and 2–7% behind gradient boosting.
- **Big moves** (10% of capacity or more): "no change" was off by 21–24%, the seasonal change by 16–19%, and gradient boosting by about 15–18% (13–17% with known rain).
- **What the model used most:** the seasonal-average change (45% of the gain at 4 weeks).

### Harder test: closer to a village tank

Village tanks have no long record for a seasonal average, and SWOT gives a reading only about every 10 days. Both tests are on reservoirs the model never saw.

| Inputs | Horizon | No change | Seasonal change | Model with a seasonal average | Model without one |
|---|---|---|---|---|---|
| Daily, exact | 1 week | 4.5 | 4.6 | 4.0 | 3.9 |
| Daily, exact | 2 weeks | 7.6 | 7.3 | 6.7 | 6.6 |
| Daily, exact | 4 weeks | 12.5 | 10.8 | 9.9 | 10.3 |
| Every 10 days, about 3% noise | 1 week | 7.4 | 7.0 | 6.6 | 6.4 |
| Every 10 days, about 3% noise | 2 weeks | 9.9 | 9.1 | 8.4 | 8.5 |
| Every 10 days, about 3% noise | 4 weeks | 14.3 | 12.0 | 10.9 | 11.7 |

**What it means for C:**
- The model works, but the gain is modest: about 10–20% better than "no change" and about 10% better than a seasonal rule.
- A village tank does not need a long record: the model without any seasonal average does almost as well (0–7% worse).
- SWOT-like readings still work: with a reading every 10 days and no seasonal average, the model is 14% better than "no change" at 2 weeks and 19% better at 4 weeks.
- Rain forecasts would help: knowing the coming rain cut the error by another 6–9%.

**Weak points:**
- one 18-month test period, and no tuning;
- the department's numbers have errors and many unchanged days;
- in the 10-day test, the model is not told how old the last reading is;
- these are reservoirs of about 100 to over 100,000 acre-feet, not village tanks.

## Member B, part 1: SWOT trust screen

### Data

- **Matching:** the 117 reservoirs' coordinates were matched to SWOT's prior lakes; 110 matched (70 major, 40 medium).
- **SWOT readings:** every Lake Single-Pass reading (version D, from Hydrocron), July 2023 to October 2026.
- **Pairing:** each reading was paired with the department's storage on the same day (or the day before). Left out:
  - days when the tank was 97% full or more, or 3% or less;
  - days when the department's value had not changed for more than 5 days (1,135 pairs), because the sheet may not have been updated.
- **Final set:** 85 reservoirs had enough pairs. Kande Ela was then left out, because more than half its readings were "bad" (a wrong lake match or broken data). That leaves **84 reservoirs and 4,044 readings**, from 0.13 km² up.

### The label

- **The curve:** for each reservoir, SWOT height against stored water, h = h0 + a·x^b with a > 0 and 0.2 ≤ b ≤ 1, so it always rises and bends the way a real tank does. It is a robust fit (soft-L1) to the readings with wse_u under 0.5 m.
- **The label:** each reading's distance from that curve. "Bad" means more than 30 cm off; 16% of readings are bad.
- **The check:** at our 5 big reservoirs, the curve distance agrees with the distance from the department's gauge heights: correlation 0.99 over 362 readings, and the same good-or-bad call 98% of the time.

### How we tested

- **Features:** only SWOT's own fields, and only what is known on the day:
  - quality_f, xovr_cal_q, dark_frac, partial_f;
  - cross-track distance and wse_u;
  - area against SWOT's reference area, lake size and month;
  - the distance from the median of the same lake's readings in the **past** 16 days.
- **Model:** LightGBM classifier, 300 trees.
- **Split:** 10 folds by reservoir, so every reading is scored by a model that never saw its reservoir.
- **Fair comparison:** inside each reservoir, the model keeps as many readings as the flag rule keeps there.

### Results

| Method | Readings kept | Share bad among kept | Mean error of kept | 90th-percentile error |
|---|---|---|---|---|
| All readings | 4,044 (100%) | 16.2% | 26.8 cm | 47 cm |
| SWOT flag: good only (quality_f = 0) | 2,223 (55%) | 8.3% | 12.1 cm | 27 cm |
| Trust screen, same number in each reservoir | 2,223 (55%) | 8.5% | 11.9 cm | 27 cm |
| SWOT flag: good or suspect (quality_f ≤ 1) | 3,440 (85%) | 12.6% | 20.7 cm | 35 cm |
| **Trust screen, same number in each reservoir** | 3,440 (85%) | **11.1%** | **14.8 cm** | **32 cm** |
| Our current rule (flags, crossover, dark water, 3.5 MAD) | 2,672 (66%) | 11.0% | 14.6 cm | 31 cm |
| **Trust screen, same number in each reservoir** | 2,672 (66%) | **9.6%** | **12.9 cm** | **29 cm** |

- **AUC inside each reservoir** (median): trust screen 0.82, quality_f 0.73, wse_u 0.63. The screen beat quality_f in 66% of reservoirs.
- **Small reservoirs** (under 2 km², 43 reservoirs): at the 85% level, the error fell from 21.9 cm to 19.4 cm.
- **Rescued readings:** SWOT marks 253 readings "bad" (quality_f = 2), but 86% of them are within 30 cm of the curve.
- **What it used most:** xovr_cal_q (22%), wse_u (20%), cross-track distance (14%), lake size (13%), area against the reference area (9%), agreement with past readings (9%).

**Is the data enough?** We held out about 17 reservoirs, trained on 5, 10, 20, 40 or all of the rest, and repeated this 25 times. The numbers are the mean error of the kept readings, in cm:

| Reservoirs trained on | Readings | Keep about half: screen | Keep about half: SWOT flag | Keep 85%: screen | Keep 85%: SWOT flag |
|---|---|---|---|---|---|
| 5 | 243 | 13.2 | 12.1 | 18.5 | 20.7 |
| 10 | 474 | 12.6 | 12.1 | 16.8 | 20.7 |
| 20 | 986 | 11.9 | 12.1 | 15.8 | 20.7 |
| 40 | 1,935 | 11.8 | 12.1 | 15.1 | 20.7 |
| All (67–68) | 3,235 | 11.8 | 12.1 | 14.9 | 20.7 |

After about 40 reservoirs the curve is flat, so the amount of data is enough. Better features or labels would help more than more data.

**Weak points:**
- the curve is fitted to the same readings it labels, though it was checked against gauges at 5 reservoirs;
- only the 30 cm cut-off was tried;
- the smallest test reservoir is 13 ha, so village tanks of 5–20 ha are at the edge of what was tested.

## Member B, part 2: storage model (added 28 October; B's main model since 29 October)

**Question:** can one model, trained on many gauged reservoirs, turn a tank's SWOT height (and water area) into % full, for tanks it has never seen? This is better than one formula for all tanks.

- **Test:** 5 folds by reservoir. Error: mean absolute error in % of capacity.
- **Inputs:** only what a village tank also has: height above the tank's lowest screened reading, the height span of its readings, the full-supply area and outline shape (from the department's map layer), and the catchment. The area version adds the Global Water Watch area share within 5 days. No capacity and no scheme type.
- **Baseline:** one formula for all tanks, y = c·((h − low) / span)^k, fitted on the training reservoirs.
- **Ceiling:** each tank's own curve, fitted on its own pairs. A new tank has no such history.

| Version | Readings used | Reservoirs | One formula | Model | Model wins at | Under 2 km²: formula / model | Each tank's own curve |
|---|---|---|---|---|---|---|---|
| Height only | all readings | 83 | 11.5 | 9.7 | 66% | 14.5 / 13.1 | 7.2 |
| Height only | SWOT loose flag | 83 | 11.1 | 9.2 | 73% | 14.0 / 13.2 | 6.5 |
| **Height only** | **B's trust screen** | 83 | **10.8** | **8.4** | **76%** | **13.9 / 11.3** | 6.1 |
| Height + area | all readings | 54 | 10.7 | 6.9 | 89% | 13.4 / 10.1 | 6.1 |
| Height + area | SWOT loose flag | 54 | 10.3 | 6.6 | 80% | 13.5 / 10.2 | 5.2 |
| **Height + area** | **B's trust screen** | 54 | **9.5** | **6.3** | **80%** | **13.4 / 10.4** | 4.7 |

- **Area only** (Global Water Watch areas, 69 reservoirs, 11,484 pairs): one power law 15.0, area × fixed depth 27.7, model 14.2, each tank's own curve 7.7. On its own, area is not enough. Height is what makes the model work.
- **The screen shows up in storage:** with height only, the error falls from 9.2 (loose flag) to 8.4 (screen), and on small reservoirs from 13.2 to 11.3.
- **Still open:** Imbulana's (2022) tank equation needs each tank's full depth (bed level), which the department's map layer does not have. B has to find it (tank tables) or compare against the one-formula baseline only.

## Member A: water map

Not trained in this check. The evidence:
- **Published results:** U-Nets trained on S1S2-Water reach IoU 0.85–0.97 for open water.
- **Our Kanhindawewa check:** the tank's full-supply area is 83.4 ha (from the department's drone survey), and it is often covered by water plants. On each satellite day, each simple rule's water area was compared with the true area at SWOT's height:

| Simple rule | Days | Median area error | Main problem |
|---|---|---|---|
| Sentinel-2 NDWI | 15 | 65% | misses water under plants |
| Sentinel-2 scene class (SCL) | 15 | 100% | finds no water |
| Sentinel-1 Otsu threshold | 36 | 57% | misses water and finds false water |
| Sentinel-1 at −18 dB | 36 | 92% | misses water under plants |
| SWOT's own water area | 41 | 36% | misses water under plants |

- **Good areas matter:** storage change worked out from height and area over three periods came within 0.5% of the truth with perfect areas. With SWOT's own areas, it was off by 1%, 1% and 21%.
- **Advice from the review** (not yet checked):
  - give the model each tank's full-supply outline as an input;
  - about 150–250 hand-labelled 256 × 256 chips after pre-training;
  - L-band radar (NISAR) as a stretch goal;
  - try Prithvi-EO-2.0 fine-tuned on Sen1Floods11.

## What changed after the review

| Item | First run | Corrected | Why |
|---|---|---|---|
| C: seasonal change, MAE at 1 / 2 / 4 weeks | 3.9 / 5.9 / 8.5 | 4.6 / 7.3 / 10.8 | The sheet's long-term average included the test years |
| C: gradient boosting | 3.4 / 5.4 / 8.0 | 4.0 / 6.6 / 9.8 | The same leak: it was the model's top input |
| C: LSTM | 3.7 / 5.8 / 8.2 | 4.3 / 6.9 / 10.0 | The same leak. Also new: residual target, dropout, early stopping |
| C: better than "no change" by | 23–37% | 10–22% | |
| B: keep about half, screen against flag | 11.8 against 13.9 cm | 11.9 against 12.1 cm | Fair test inside each reservoir, physical curve, stale days removed, past-only features |
| B: keep 85%, screen against flag | 14.6 against 22.9 cm | 14.8 against 20.7 cm | The same |
| B: label check against gauges | 0.97 | 0.99 | Physical curve |

Two review points we did not accept as stated:
- **"Target contamination" in the 10-day test.** The target is the true storage later minus the last noisy reading. That is correct, because the error we report is the error of the forecast storage.
- **"Gradient boosting beats the clean seasonal rule by 25–30%."** That compared the clean rule with the leaky model. Done properly, the gain is 9–12%.

## Run it again

Run the scripts from this folder (`model-check/`). Each writes its output here; our copies are in `results/`.

| Step | Script | Reads | Writes |
|---|---|---|---|
| 1 | `build_storage.py` | the daily sheet's four hidden tabs, saved as CSV in `../data-sources/ground-data/downloads/sheet_tabs/` (not included) | `storage_daily.csv` (12 MB, not included) |
| 2 | `fetch_rain.py` | `storage_daily.csv` (reservoir positions), the Open-Meteo archive | `rain_daily.csv` (9 MB, not included) |
| 3 | `c_gbm.py`, `c_lstm.py`, `c_variants.py` (with `c_prep.py`) | the two files above | `c_results.csv`, `c_lstm_results.csv`, `c_variants.csv` |
| 4 | `b_match.py` | `storage_daily.csv`, the SWOT lake files (`SWOT_L2_HR_LakeSP_Prior_*.zip`, about 125 MB, free NASA Earthdata login) and the Hydrocron fetcher in `../feasibility-check/work/` (`a2_fetch.py`) | `b_matched.csv` |
| 5 | `b_trust_v2.py` (`b_trust.py` is the first version) | `b_matched.csv`, `storage_daily.csv`, the 5 big reservoirs' daily gauge levels in `../data/timing-gauges/levels_daily.csv` (not included) | `b_results.csv`, `b_readings_labelled.csv`, `b_learning_curve.csv` |
| 6 | `gww_pairs.py` | the Global Water Watch archive, `gww_reservoir_timeseries.nc` (527 MB, [Zenodo](https://zenodo.org/records/22658942), CC BY 4.0), in `../data-sources/gww-archive/`; `dept_layer.geojson`; `sheet_to_layer.csv` | `gww_pairs.csv`, `gww_match.csv` |
| 7 | `area_storage_test.py swot` or `gww`, then `height_storage_test.py` | the files above | `area_storage_*.csv`, `height_storage_results.csv` |

- `dept_layer.geojson` is the Irrigation Department's reservoir layer ([ArcGIS](https://services3.arcgis.com/J7ZFXmR8rSmQ3FGf/ArcGIS/rest/services/Scheme_Data_2/FeatureServer/10), 354 outlines, 6 MB), saved as GeoJSON. It is not included.
- `sheet_to_layer.csv` links each sheet reservoir to its outline in that layer (116 of 117).
- **Python packages:** pandas, numpy, scipy, lightgbm, torch, geopandas, xarray, netCDF4.

**Data sources:**
- Irrigation Department of Sri Lanka: the daily reservoir sheet and the reservoir layer.
- SWOT: NASA/JPL and CNES, Lake Single-Pass product version D, through PO.DAAC's Hydrocron.
- ERA5 rain: Copernicus Climate Change Service, through Open-Meteo (CC BY 4.0).
- Global Water Watch: its September 2026 archive on Zenodo (CC BY 4.0).
