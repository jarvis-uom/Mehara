# Past research summary: SWOT Tank Water Tracker

This file holds every study used for the SWOT Tank Water Tracker proposal, in simple words.
- **Part 1:** the 10 papers we read in full, each with its link. This part replaces the earlier "Reading notes" PDF.
- **Part 2:** the other studies in the proposal's literature table.
- **Checked:** the key numbers were checked against the papers on 26 October 2026.
- **Newer finds** from the 26 October research scan (3D-LAKES curves, the Global Water Watch archive, Cyclone Ditwah reports) are in the scope expansion ideas file, not here.

## The 10 papers at a glance

Six of the ten only partly worked, or were tested on very little data. That leaves room for our project.

| # | Paper | What it gives us | Did it work? |
|---|---|---|---|
| 1 | Lopane et al. 2025: which small tanks to fix first | National tank lists and outlines; why tank data matters | Partly |
| 2 | Imbulana 2022: village tank depth, area and capacity | A cheap formula to turn tank area and depth into volume | Yes |
| 3 | Madumalsha et al. 2025: ICESat-2 levels of Sri Lankan reservoirs | Our reservoirs and gauges; shows the laser satellite is too sparse | Partly |
| 4 | Wang et al. 2025: the SWOT Prior Lake Database | The lake IDs we need to get each tank's SWOT data | Yes |
| 5 | Köhn and Nielsen 2026: SWOT on small lakes | A tested recipe for cleaning SWOT data of small lakes | Yes |
| 6 | Wieland et al. 2024: the S1S2-Water dataset | Free training data and a U-Net recipe for our water maps | Yes |
| 7 | Rajakaruna et al. 2023: Sentinel-1 area of Sri Lankan reservoirs | A simple baseline our water maps must beat | Partly |
| 8 | Das and Hossain 2025: storage change in the SWOT era | The closest study to our whole pipeline | Partly |
| 9 | Li et al. 2026: from SWOT level to storage | Why each tank's curve must be tied to gauge data | Partly |
| 10 | Meyen 2018: deep learning forecast of reservoir levels | A starting point for our 2–4 week forecast | Yes, on little data |

# Part 1: the 10 papers

The papers follow the order our project works in: the problem, water height, water area, height to storage, then forecasting.

## A. The problem in Sri Lanka

Most small tanks in Sri Lanka have no up-to-date volume curve or water record.

### 1. Which small tanks to fix first (Lopane et al., 2025)

*Optimising water storage for climate resilience: geospatial targeting for small tanks rejuvenation in Sri Lanka.* Journal of Environmental Management, 374, 124031. https://doi.org/10.1016/j.jenvman.2025.124031

- **Problem:** Sri Lanka has about 23,000 small tanks, and about 21% do not work. There was no clear way to choose which ones to repair first.
- **What they did:** they combined national data:
  - a Ministry of Agriculture list of 23,163 tanks;
  - UNDP 2018 outlines of 11,014 tanks;
  - population, CHIRPS rainfall, soil erosion and well yields.

  They joined a supply score (siltation × erosion) with a demand score (farm-dependent people × rainfall variability), and added a groundwater filter on top.
- **What they found:** Kurunegala (4,434 tanks) and Anuradhapura (2,905 tanks) ranked highest on both scores. The method gives ranked lists by tank, division and district.
- **Did it work?** Partly. The ranking is clear and repeatable, but the paper reports no validation.
- **Weak points:**
  - Links between tanks in a cascade were not modelled, because the data was missing.
  - Demand uses a 1 km circle around each tank, not the real command area.
- **Why it matters for us:** it is not a satellite paper, but it shows why tank water data matters. It also names the national tank lists and outlines. Slide 2 of our deck quotes its figure of about 23,000 tanks.

### 2. Depth, area and capacity of village tanks (Imbulana, 2022)

*Depth-area-storage capacity relationships of village tanks.* Engineer: Journal of the Institution of Engineers, Sri Lanka, 55(2), 53–61. https://doi.org/10.4038/engineer.v55i2.7508

- **Problem:** to know how much water a tank holds, you need its depth–area–capacity curve. Most of Sri Lanka's roughly 21,000 village tanks have none, because surveys are slow and costly.
- **What they did:**
  - used survey data from about 150 tanks in 10 cascades, in the Malwathu Oya, Yan Oya and Mi Oya basins;
  - tested the old Kennedy rule (V = 0.4 × D × A) and a straight-line rule;
  - proposed a relative equation, s/C = m × (a/A) × (d/D), and tested it on 30 tanks.
- **What they found:**
  - Capacity follows area × depth almost in a straight line (R² about 0.97).
  - In the relative equation, m averaged 1.0137, and the fit was 0.99 or better for all 30 tanks.
  - Four or five area readings were enough for a good curve.
  - For tanks with m between 0.98 and 1.04, the error stayed under 5% at depths above one-third of full depth.
- **Did it work?** Yes. A simple equation fitted most tanks well, especially in the middle and lower parts of the basins.
- **Weak points:**
  - Each curve is specific to its tank, and upper-basin tanks gave mixed results.
  - Only three river basins were studied.
  - Dead storage below the sluice was not covered.
- **Why it matters for us:** it gives a cheap way to turn area and depth into volume for small tanks. A few area–height pairs from our water maps and SWOT can build each tank's curve. The proposal uses this equation as the fallback for plant-covered tanks. Two of its basins, Malwathu Oya and Yan Oya, are in our area.

## B. Water height from space

SWOT measures water levels far more often than the older laser satellite, ICESat-2.

### 3. ICESat-2 water levels for Sri Lankan reservoirs (Madumalsha et al., 2025)

*Estimation of water elevation using ICESat-2 satellite.* Engineer: Journal of the Institution of Engineers, Sri Lanka, 58(1), 15–25. https://doi.org/10.4038/engineer.v58i1.7674

- **Problem:** many reservoirs lack good gauge data. Can the ICESat-2 laser satellite measure Sri Lankan reservoir levels accurately?
- **What they did:**
  - Used ICESat-2 ATL13 data (December 2018 to 2023) for nine reservoirs: Udawalawe, Parakrama Samudraya, Kala Wewa, Victoria, Randenigala, Nachchaduwa, Nuwara Wewa, Kotmale and Rantambe.
  - Removed bad points with quality flags and an outlier test.
  - Compared the results with daily gauge levels from the Mahaweli Authority and the Irrigation Department.
- **What they found:**
  - Correlation with the gauges was 0.99 for all nine reservoirs.
  - ICESat-2 always read too high: by 0.37 m on average, from 0.16 m (Victoria) to 0.43 m (Udawalawe).
  - Two of our study reservoirs are in it: Nachchaduwa (0.415 m, 6 readings) and Nuwara Wewa (0.382 m, 7 readings).
  - Each reservoir had only 3 to 8 usable passes in five years, because ICESat-2 repeats every 91 days.
- **Did it work?** Partly. The heights are accurate, but there are far too few of them for regular monitoring.
- **Weak points:**
  - Very few observations.
  - A steady bias that was not corrected.
  - No error in metres for each reservoir (only a relative RMSE).
- **Why it matters for us:**
  - It uses our reservoirs and gauge sources, with a clear recipe for checking satellite heights against gauges.
  - We should expect a similar fixed offset between SWOT and the gauges. So we either match each gauge's zero first, or compare changes in level.
  - SWOT sees a reservoir about every 10 days (paper 8), and the authors themselves suggest adding SWOT.

### 4. The SWOT Prior Lake Database (Wang et al., 2025)

*The Surface Water and Ocean Topography (SWOT) Prior Lake Database (PLD): lake mask and operational auxiliaries.* Water Resources Research, 61(3), e2023WR036896. https://doi.org/10.1029/2023WR036896

- **Problem:** SWOT needs a map of known lakes to decide which water pixels belong to which lake. It also needs reference values to work out storage change.
- **What they did:** they built the Prior Lake Database (PLD), mainly from the UCLA Circa-2015 lake map. They added reservoir outlines, dam IDs, river links and SWOT orbit tracks. Each lake gets an ID, an outline, a reference area and height, and its pass coverage.
- **What they found:**
  - The PLD has 5,897,941 lakes of at least 1 ha, and 76% of them are smaller than 10 ha.
  - About 97% of lakes are fully seen at least once per 21-day cycle.
- **Did it work?** Yes. It is the lake list that SWOT's lake products use today.
- **Weak points:**
  - Small lakes are slightly shrunk in the mask, and many are missing.
  - Checked against lake censuses in Canada, Sweden and the USA, it holds 78.2% of lakes of 6.25 ha or more, and 63.1% of lakes of 1 ha or more. So about 22% and 37% are missing.
- **Why it matters for us:**
  - We need each tank's PLD lake ID to get its SWOT heights from Hydrocron.
  - Each outline must be checked against the real tank.
  - Our feasibility check found lake IDs for all 13 study tanks. But only 58% of the mapped tanks of 6.25 ha or more in our area have one.

### 5. How well SWOT works on small lakes (Köhn and Nielsen, 2026)

*Evaluation of SWOT HR PIXC version D water level time series of small lakes.* Hydrology and Earth System Sciences, 30(20), 6443–6458 (published 22 October 2026). https://doi.org/10.5194/hess-30-6443-2026

- **Problem:** small lakes have been hard to measure from space. Can SWOT's pixel cloud give accurate water-level time series for them?
- **What they did:**
  - Studied 37 gauged lakes in Denmark (0.25 to 40 km²) with SWOT pixel cloud data (version D), July 2023 to May 2026.
  - Tested six rules for choosing good pixels.
  - Compared the result with the simpler LakeSP product from Hydrocron.
- **What they found (final version):**
  - The best rule (drop geo-flagged points, keep passes with at least 200 points) gave a median error of 5.57 cm and a correlation of 0.91.
  - Hydrocron's LakeSP data, after quality filtering, was slightly better (4.92 cm, 0.94). But it kept only about half the dates: a median of 60 per lake, against 125.
  - About 18% of passes showed roll errors.
- **Did it work?** Yes, for tracking how lake levels change. Absolute height was not tested.
- **Weak points:**
  - Offsets were removed using the gauges, so absolute accuracy is unknown.
  - Only Danish lakes were studied.
  - Lakes whose level hardly moves gave poor correlation, because the change is inside SWOT's error. Danish lakes move less than 1 m a year.
- **Why it matters for us:**
  - It is a tested recipe for cleaning SWOT data of small lakes.
  - The simple Hydrocron product we plan to use did about as well, with fewer dates.
  - Our tanks swing by metres, so their changes stand out. For example, Hurulu Wewa rose about 4.6 m in late 2025.
  - Its smallest lakes (25 ha) are still 1.5 to 4 times the size of our village tanks (6.5–16 ha).
  - Our "best quality" rule comes from its Section 2.1: quality_f = 0, xovr_cal_q ≤ 1 and dark_frac ≤ 0.1.
- **Note:** the PDF in our folder is the earlier preprint, and its numbers differ: 5.34 cm and 0.91 for the pixel rule, 6.05 cm and 0.89 for Hydrocron. Quote the final version.

## C. Mapping water area

Water maps from Sentinel-1 (radar, which sees through cloud) and Sentinel-2 (optical) give the area of each tank.

### 6. S1S2-Water: training data for water maps (Wieland et al., 2024)

*S1S2-Water: a global dataset for semantic segmentation of water bodies from Sentinel-1 and Sentinel-2 satellite images.* IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 17, 1084–1099. https://doi.org/10.1109/JSTARS.2023.3333969 · Dataset: https://doi.org/10.5281/zenodo.8314175

- **Problem:** radar confuses water with other dark, smooth ground, and optical images have clouds and shadows. Good global training labels for deep learning were missing.
- **What they did:**
  - Built a free dataset of 65 areas (each 100 × 100 km, about 650,000 km² in all) in 29 countries.
  - Each area has matching Sentinel-1 and Sentinel-2 images, expert-checked water masks and slope.
  - Trained U-Net models with four encoders and tested which bands help.
- **What they found:**
  - Overlap with the true mask (IoU, where 1 is perfect) was 0.965 with Sentinel-2 and 0.845 with Sentinel-1.
  - For Sentinel-1, using both VV and VH added 0.095, and adding slope added 0.072.
  - On flood images, IoU dropped to 0.75 (Sentinel-1) and 0.82 (Sentinel-2).
  - They chose a U-Net with an EfficientNet-B0 encoder: about as accurate as the bigger ones, and faster.
- **Did it work?** Yes. Very good with Sentinel-2, and good with Sentinel-1, which also works under cloud.
- **Weak points:**
  - Few cloudy scenes, and normal water only, not floods.
  - Sentinel-1 mixed up sand and bare soil with water.
  - Sentinel-2 mixed up cloud shadow with water.
- **Why it matters for us:** it gives free training data and a tested recipe:
  - a U-Net with an EfficientNet-B0 encoder;
  - VV + VH + slope for Sentinel-1;
  - RGB + NIR + slope for Sentinel-2.

  We still need to fine-tune and test it on Sri Lankan tanks. It has no "water under plants" class.

### 7. Sentinel-1 water area of Sri Lankan reservoirs (Rajakaruna et al., 2023)

*Analysis of rainfall variability and water extent of selected hydropower reservoir using Google Earth Engine (GEE): a case study from two tropical countries, Sri Lanka and Vietnam.* arXiv:2310.05682. https://doi.org/10.48550/arXiv.2310.05682

- **Problem:** how does rainfall change the water area of hydropower reservoirs in monsoon countries? The authors, from the International Water Management Institute (IWMI), wanted a cheap way to watch this from space.
- **What they did:**
  - Used Sentinel-1 radar images (10 m, 2017–2022) in Google Earth Engine, and split water from land with Otsu thresholding.
  - Compared water area with CHIRPS rainfall for Victoria, Kotmale and Samanalawewa, plus three reservoirs in Vietnam.
- **What they found:** in 2022, the water area changed through the year:
  - Victoria: 10.85 km² in April to 21.23 km² in November;
  - Kotmale: 3.63 to 5.41 km²;
  - Samanalawewa: 4.55 to 6.00 km².

  Area usually rose during and after the monsoon.
- **Did it work?** Partly. The maps show clear seasonal change, but the authors call the results preliminary, and they never checked them against ground truth.
- **Weak points:**
  - No accuracy numbers.
  - The rainfall–area link is only described in words.
  - Dam operations also change the area.
- **Why it matters for us:** it is a simple, free Sentinel-1 baseline for Sri Lankan reservoirs. Our deep-learning maps should beat it, and we must add the accuracy check it lacks.

## D. From height to storage

Storage comes from water height plus a height–area–volume curve. Both papers here find that the curve, not SWOT, causes most of the error.

### 8. Storage change in the SWOT era (Das and Hossain, 2025)

*Multi-satellite tracking of surface water storage change in the era of Surface Water and Ocean Topography (SWOT) satellite mission.* Earth and Space Science, 12, e2024EA004178. https://doi.org/10.1029/2024EA004178

- **Problem:** before SWOT, storage came from mixing satellites that measure area with altimeters that measure height, and they rarely looked at the same reservoir at the same time. How much better is SWOT?
- **What they did:**
  - Studied 245 reservoirs in the USA, Spain, South Africa, Thailand and India (July 2023 to October 2024).
  - Combined SWOT with Sentinel-1, Sentinel-2, Landsat 8/9 and ICESat-2.
  - Built area–elevation–volume curves from SRTM elevation data, and compared storage with ground records.
- **What they found:**
  - SWOT water levels matched gauges with a median R² of 0.91 and an RMSE of 0.92 m.
  - SWOT storage reached a median R² above 0.8, with curves from SRTM or from ground data (98 reservoirs had ground curves).
  - SWOT saw each reservoir about every 10 days, against 59 days for ICESat-2.
- **Did it work?** Partly. SWOT tracks storage well when the curve is good. But their machine-learning link to the other satellites did not work yet (RMSE 377 million m³).
- **Weak points:**
  - The SWOT record is short (under two years), and SWOT sometimes mislabels water pixels.
  - Curves built from SRTM can be off by several metres.
  - The area-based method did badly in monsoon regions such as Thailand.
- **Why it matters for us:**
  - It is the closest study to our whole pipeline: SWOT height, plus radar and optical area, plus a curve, gives storage.
  - It warns that the curve is the main error, so we should use surveyed curves (paper 2) and gauges.
  - Their reservoirs are mostly large, so our small tanks will be harder.
  - It comes from the group behind RAT, the Reservoir Assessment Tool (see "Code we downloaded").

### 9. From SWOT water level to storage (Li et al., 2026)

*From water level to storage: evaluating vertical alignment and hypsometric constraints for SWOT-based reservoir storage.* Water, 18(19), 2462. https://doi.org/10.3390/w18192462

- **Problem:** SWOT gives water height. To get storage you need a height–storage curve, and the right zero point linking SWOT height to it. If that alignment is wrong, the storage is wrong.
- **What they did:**
  - Used 14 reservoirs, mostly in the USA and Mexico, plus one each in India and Australia.
  - Had 888 daily SWOT LakeSP records (August 2023 to March 2026).
  - Tested two global curve datasets (GRDL and REGEOM), and ways to align and extend the curves.
- **What they found:**
  - Assuming the highest SWOT level means a full reservoir gave a median error of 14.9% of capacity.
  - Calibrating each reservoir with its own records cut this to 2.6–2.9%.
  - One alignment value could not be carried over to reservoirs without records.
- **Did it work?** Partly. It works well when local storage records exist, but not yet for ungauged reservoirs.
- **Weak points:**
  - Only 14 reservoirs, mostly in North America, and a short SWOT record.
  - The best numbers were tested on the same data used to calibrate.
- **Why it matters for us:**
  - SWOT heights alone are not enough. Each tank's curve must be tied to Irrigation Department or Mahaweli gauge data.
  - A 10 cm height error was about 0.6 million m³ at their median reservoir, which is a much bigger share of a small tank.

## E. Forecasting

### 10. Predicting reservoir water levels with deep learning (Meyen, 2018)

*Predicting reservoir water levels using deep learning.* MSc thesis, University of Moratuwa. https://dl.lib.uom.lk/items/dfe8a16d-a36f-406e-89f4-6454f7b6e6b3

- **Problem:** Sri Lankan reservoir managers share water between farming and hydropower, but use simple forecasts. Can deep learning predict water levels better?
- **What they did:**
  - Used 546 days of daily water levels at Kotmale (July 2015 to January 2017), plus six weather variables for 10 nearby sites.
  - Trained LSTM and GRU networks.
  - Compared them with linear regression, six SVR models and XGBoost.
- **What they found:**
  - The best model, an LSTM with 150 neurons, had a test error (RMSE) of 2.14.
  - Most other models scored between 11.4 and 15.2, and two SVR models were far worse (40–48).
  - XGBoost had almost zero training error, a sign that it over-fitted.
- **Did it work?** Yes, on this small dataset. The LSTM error was about five times lower than the best baseline.
- **Weak points:**
  - One reservoir and about 1.5 years of data.
  - The thesis contradicts itself in places: the design describes hourly data, but the results use daily data.
  - It forecast only one day ahead.
  - Only the LSTM was given the previous day's water level, while the other models saw weather only. That explains much of the gap.
- **Why it matters for us:**
  - It shows an LSTM can forecast a Sri Lankan reservoir's next-day level from past levels and weather. That is a starting point for our 2–4 week forecast.
  - We forecast storage for many tanks, so we need longer records and satellite inputs.

# Part 2: other studies in the proposal

These studies come from the proposal's literature table and references. Their PDFs are not in our folder.

| Study | Where | What it does | Gap for our project |
|---|---|---|---|
| [Ma et al., 2026](https://impacts.ucar.edu/en/publications/monitoring-reservoir-storage-using-swot-satellite-observations-an/), Water Resources Research 62(3) | 12 reservoirs, western USA | Uses SWOT height and area together, with a reservoir operation model, to estimate storage. Height error under 20 cm, storage error under 10% | Large US reservoirs; not small tanks, not Sri Lanka |
| [Wang and Tseng, 2025](https://acrs-aars.org/proceeding/ACRS2025/fp_abs-127.pdf), ACRS 2025 | 14 reservoirs and 15 ponds, Taiwan | Checks SWOT heights against gauges. A refined workflow cut errors to 0.15–0.53 m, and detected ponds of about 110 × 110 m | Heights only; no volume, no AI |
| [IFAD and ESA, 2026](https://www.ifad.org/en/w/publications/water-surface-maps-to-manage-ancient-tank-cascade-systems-in-sri-lanka) | Six districts, Sri Lanka | Monthly 10 m water-surface maps of more than 10,650 tanks, 2017 to August 2025 | Area only; no volume |
| [Wu et al., 2023](https://www.mdpi.com/2072-4292/15/24/5740), Remote Sensing 15(24), 5740 | Four small reservoirs, California | A ResNet34 model reads the water level from Sentinel-1 and Sentinel-2 images; error 0.19–0.77 m | Needs gauge records to train for every reservoir; no SWOT |
| [Pruekthikanee, 2026](https://meetingorganizer.copernicus.org/EGU26/EGU26-4154.html), EGU26-4154 | Chi River basin, Thailand | A U-Net on Sentinel-1, Sentinel-2, NDWI and NDVI separates open water from floating plants and flooded farmland, then builds a stage–area relationship checked against a gauge | The closest study to our AI part, but only a conference abstract with no results yet; no SWOT, no tanks |
| [Perera and Jayaweera, 2020](https://archive.cmb.ac.lk/items/84f64851-eaa9-4b8c-babd-5d18f130c1ee), University of Kelaniya | Kaudulla, Senanayake Samudraya and Lunugamwehera, Sri Lanka | Compares Sentinel-1 water area (Google Earth Engine) with reported volumes, 2017–2019: R² 0.73, 0.94 and 0.67. A Sentinel-2 water index failed under cloud | Area only; no water height, no curve |
| [Madhurshan et al., 2024](https://dl.lib.uom.lk/handle/123/23575) (UoM) | Victoria reservoir | Compares the water area from Landsat 8, Sentinel-1 and Sentinel-2 | One reservoir; area only |
| [Jayawardena et al., 2021](https://dl.lib.uom.lk/items/4bf9e544-7327-4919-8839-197733244865) (UoM) | Sri Lanka | Extracts water from Landsat 8 with classic machine-learning classifiers; kappa above 0.95 | Classic methods on Landsat; area only |
| [Dawalagala et al., 2023](http://dl.lib.uom.lk/items/c7fd94f4-b934-44a7-87ac-63f4e30cd985) (UoM) | North Bolgoda Lake | Tracks water hyacinth over time; it spread most from May to November | Shows that floating plants hide water, which is why we add a "water under plants" class |
| [ICITR 2024 abstracts](https://icitr.uom.lk/abstracts/abstract-2024.pdf) (UoM) | UoM IT final-year projects | Typical computer-vision projects fine-tune a pretrained model on about 600–1,250 images and add a web app | Shows that our project sits above a typical final-year project |

# What all this tells us

We found no study that uses SWOT for Sri Lankan irrigation tanks; we searched in October 2026 and checked again on 26 October. The pieces exist separately, but nobody has joined them for small tanks.

- **SWOT works on small water bodies.** It tracked level changes in small Danish lakes to about 5–6 cm (paper 5), and it detected Taiwanese ponds of about 110 × 110 m (Wang and Tseng). It sees a reservoir about every 10 days (paper 8).
- **SWOT storage works on big reservoirs** (Ma et al.; paper 8), **but the curve is the weak link.** Papers 8 and 9 both find that the height-to-volume curve causes most of the error. Paper 2 gives a cheap way to build curves for village tanks.
- **Small tanks need checking.** The SWOT lake list misses about a fifth of lakes over 6.25 ha, and over a third of lakes over 1 ha, and it shrinks small ones (paper 4). So each tank's outline must be checked.
- **Sri Lankan studies stop short:**
  - ICESat-2 is accurate but too sparse (paper 3);
  - Sentinel-1 reservoir maps were never checked for accuracy (paper 7);
  - the other Sri Lankan studies give area only (Perera and Jayaweera; IFAD and ESA; Madhurshan; Jayawardena).
- **Water maps have a ready recipe:** S1S2-Water gives training data and a U-Net design (paper 6). The Thai abstract adds floating plants, but without SWOT.
- **Forecasting is barely tested.** The Sri Lankan thesis here used one reservoir, 1.5 years of data and a one-day horizon (paper 10).
- **Existing tools target dams.** RAT's SWOT plugin is built for dam reservoirs, and we found no use of it on small tanks.

**The gap our project fills:** join SWOT heights, deep-learning water maps, tank curves and gauge checks for small Sri Lankan tanks, then add a 2–4 week storage forecast.

# Code we downloaded (GitHub folder)

| Folder | What it is | Licence | How we would use it |
|---|---|---|---|
| podaac-hydrocron (https://github.com/podaac/hydrocron) | Code for Hydrocron, NASA PO.DAAC's free API that returns SWOT height and area for each lake as CSV or GeoJSON | Apache-2.0 | We call the API rather than run this code: `feature=PriorLake` plus a tank's lake ID returns `wse` (height) and `area_total` |
| UW-SASWE-RAT (https://github.com/UW-SASWE/RAT) | Reservoir Assessment Tool 3.0, from the group behind paper 8. Its SWOT plugin gets Hydrocron data and estimates storage change from height, from area, or from both | GPL-3.0 | The closest existing tool to our idea. Run it on a large reservoir as a benchmark. Copying its code would make our app GPL too |
| podaac-cookbook-SWOT-notebooks (https://podaac.github.io/tutorials/) | 8 notebooks from the PO.DAAC Cookbook: searching and downloading SWOT, Hydrocron time series, pixel-cloud area, quality flags, shapefile to CSV, and HLS water detection | Apache-2.0 | Our starter kit for the first weeks. The full cookbook (about 1.2 GB) stays online |

# References (APA)

**The 10 papers in our folder**
- Das, P., & Hossain, F. (2025). Multi-satellite tracking of surface water storage change in the era of Surface Water and Ocean Topography (SWOT) satellite mission. *Earth and Space Science, 12*, e2024EA004178. https://doi.org/10.1029/2024EA004178
- Imbulana, K. A. U. S. (2022). Depth-area-storage capacity relationships of village tanks. *Engineer: Journal of the Institution of Engineers, Sri Lanka, 55*(2), 53–61. https://doi.org/10.4038/engineer.v55i2.7508
- Köhn, S. J., & Nielsen, K. (2026). Evaluation of SWOT HR PIXC version D water level time series of small lakes. *Hydrology and Earth System Sciences, 30*(20), 6443–6458. https://doi.org/10.5194/hess-30-6443-2026
- Li, Z., Fu, J., Lei, J., Gong, J., Wu, G., & Tang, L. (2026). From water level to storage: Evaluating vertical alignment and hypsometric constraints for SWOT-based reservoir storage. *Water, 18*(19), 2462. https://doi.org/10.3390/w18192462
- Lopane, F. D., Ayling, S., Hoo, Y. R., Shamsudduha, M., Wang, Q., Joseph, G., & Bahuguna, A. (2025). Optimising water storage for climate resilience: Geospatial targeting for small tanks rejuvenation in Sri Lanka. *Journal of Environmental Management, 374*, 124031. https://doi.org/10.1016/j.jenvman.2025.124031
- Madumalsha, C., Nawoda, I., Neluwala, P., & Revel, M. (2025). Estimation of water elevation using ICESat-2 satellite. *Engineer: Journal of the Institution of Engineers, Sri Lanka, 58*(1), 15–25. https://doi.org/10.4038/engineer.v58i1.7674
- Meyen, A. N. (2018). *Predicting reservoir water levels using deep learning* [Master's thesis, University of Moratuwa]. https://dl.lib.uom.lk/items/dfe8a16d-a36f-406e-89f4-6454f7b6e6b3
- Rajakaruna, P., Ghosh, S., & Holmatov, B. (2023). *Analysis of rainfall variability and water extent of selected hydropower reservoir using Google Earth Engine (GEE): A case study from two tropical countries, Sri Lanka and Vietnam* (arXiv:2310.05682). arXiv. https://doi.org/10.48550/arXiv.2310.05682
- Wang, J., Pottier, C., Cazals, C., Battude, M., Sheng, Y., Song, C., … Pavelsky, T. M. (2025). The Surface Water and Ocean Topography (SWOT) Prior Lake Database (PLD): Lake mask and operational auxiliaries. *Water Resources Research, 61*(3), e2023WR036896. https://doi.org/10.1029/2023WR036896
- Wieland, M., Fichtner, F., Martinis, S., Groth, S., Krullikowski, C., Plank, S., & Motagh, M. (2024). S1S2-Water: A global dataset for semantic segmentation of water bodies from Sentinel-1 and Sentinel-2 satellite images. *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 17*, 1084–1099. https://doi.org/10.1109/JSTARS.2023.3333969

**Other studies in the proposal**
- Dawalagala, H. S., Radampola, A., Gowsigan, P. T., Chaminda, S. P., Dassanayake, S. M., & Jayawardena, C. L. (2023). *Comparative analysis of water hyacinth dynamics in North Bolgoda Lake, Sri Lanka*. University of Moratuwa. http://dl.lib.uom.lk/items/c7fd94f4-b934-44a7-87ac-63f4e30cd985
- IFAD. (2026). *Water surface maps to manage ancient tank cascade systems in Sri Lanka*. https://www.ifad.org/en/w/publications/water-surface-maps-to-manage-ancient-tank-cascade-systems-in-sri-lanka
- Jayawardena, C. L., Brinthan, K., & Thanujan, T. (2021). *Evaluation of machine learning algorithms in classifying multispectral imagery on waterbody extraction*. University of Moratuwa. https://dl.lib.uom.lk/items/4bf9e544-7327-4919-8839-197733244865
- Ma, X., Wang, J., Gharari, S., Mizukami, N., & Lettenmaier, D. P. (2026). Monitoring reservoir storage using SWOT satellite observations and a reservoir operation model. *Water Resources Research, 62*(3). https://impacts.ucar.edu/en/publications/monitoring-reservoir-storage-using-swot-satellite-observations-an/
- Madhurshan, R., Dhananjaya, K. I. K., Keerthi, T., Dassanayake, S. M., & Chao, Z. (2024). *Multi sensor satellite remote sensing data based comparative assessment of the surface water level dynamics of the Victoria reservoir, Sri Lanka*. University of Moratuwa. https://dl.lib.uom.lk/handle/123/23575
- Perera, H. E., & Jayaweera, H. H. E. (2020). *Use of Google Earth Engine to monitor surface water: A case study in water tanks located in the dry zone of Sri Lanka*. Faculty of Science, University of Kelaniya. https://archive.cmb.ac.lk/items/84f64851-eaa9-4b8c-babd-5d18f130c1ee
- Pruekthikanee, P. (2026). *Multi-sensor deep learning fusion for spatiotemporal water surface monitoring in the Yasothon Province's Chi River Basin, Thailand* (EGU26-4154). EGU General Assembly 2026. https://meetingorganizer.copernicus.org/EGU26/EGU26-4154.html
- University of Moratuwa. (2024). *ICITR 2024 abstracts*. https://icitr.uom.lk/abstracts/abstract-2024.pdf
- Wang, C.-H., & Tseng, K.-H. (2025). *Potential of SWOT mission in monitoring Taiwan surface water dynamics*. ACRS 2025. https://acrs-aars.org/proceeding/ACRS2025/fp_abs-127.pdf
- Wu, J., et al. (2023). Combining satellite imagery and a deep learning algorithm to retrieve the water levels of small reservoirs. *Remote Sensing, 15*(24), 5740. https://www.mdpi.com/2072-4292/15/24/5740
