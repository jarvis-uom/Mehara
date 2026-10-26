# SWOT Tank Water Tracker: ideas to strengthen and expand the project

Research scan, 26 October 2026. Every item comes from a real source with its link.
- **(abstract only):** we read only the abstract or the slides.
- **(search result only):** the page itself was not opened.
- **Spot checks:** the key facts I re-checked myself are listed at the end.

**Effort** is for 3 students over the remaining ~8 months: **Low** = days, **Medium** = 2–4 weeks, **High** = more than a month or needs new skills or hardware.

---

## 1. Fix in the proposal now (small corrections)

1. **SWOT accuracy wording.**
   - *What the mission says:* the lake requirement is **< 10 cm (1σ) for lakes over 1 km²**. The **< 25 cm** figure for lakes from 250 m × 250 m to 1 km² is a **goal, not a requirement**.
   - *What 1σ means:* 68% of the errors are smaller than that number.
   - *Fix:* change "SWOT specification: 10 cm … 25 cm" in the evaluation plan to "requirement 10 cm (lakes > 1 km²), goal 25 cm (smaller lakes)".
   - Source: [SWOT lake product validation slides, SWOT Science Team 2025](https://swotst.aviso.altimetry.fr/fileadmin/user_upload/ST2025/Presentations/ST2025HS0_05_PottierFjortoft_LakeProductVal.pdf)

2. **"Suspect" readings cost only about 2 cm in the official validation.**
   - *Result:* with SWOT version D, using quality_f 0 and 1 ("suspect") gave **14 cm** error on lakes of 250 m to 1 km², against **12 cm** with quality_f 0 only. These are preliminary figures, with 3,551 against 2,228 comparisons.
   - *Why it matters:* this backs RQ2 and gives us a number to compare with. Version D also adds a new flag, `qual_f_b`, that says *why* a reading is suspect.
   - Source: same slides as item 1.

3. **Cite the official gauge source.**
   - *What to cite:* "Irrigation Department of Sri Lanka, Reservoir Status (daily water level and storage of major and medium reservoirs)" instead of "public sheet".
   - *Why it's safe:* we checked that the Department's dashboards read from the same sheet we save.
   - Source: [irrigation.gov.lk Reservoir Status](https://irrigation.gov.lk/web/index.php?option=com_content&view=article&id=84&Itemid=201&lang=en)

4. **Sentinel-1 is back to a 6-day repeat.** Sentinel-1C and Sentinel-1D now share the job, which means more radar dates per tank than the proposal assumed.
   - Source: [Copernicus Data Space Ecosystem news](https://dataspace.copernicus.eu/node/4010)

5. **Add the closest similar study to the literature review.**
   - *The study:* Pruekthikanee (Chulalongkorn University), EGU 2026 abstract EGU26-4154.
   - *What it did:* a U-Net on Sentinel-1 + Sentinel-2 + NDWI + NDVI separates open water from floating plants and flooded farmland in Thailand's Chi River basin. It builds a stage–area relationship checked against one gauge.
   - *What it lacks:* **no SWOT, no tanks and no results yet** (abstract only).
   - *Why it matters:* it shows our idea is timely and still new for Sri Lankan tanks with SWOT.
   - Source: [EGU26-4154](https://meetingorganizer.copernicus.org/EGU26/EGU26-4154.html)

6. **Add a Sri Lankan satellite water-level study.** A 2025 paper in *Engineer* (vol. 58, no. 1) studied water levels of **nine main Sri Lankan reservoirs** over five years with **ICESat-2**, NASA's laser satellite. It's a local precedent, and a possible validation partner.
   - Source: [Engineer 58(1), 2025](https://engineer.sljol.info/articles/10.4038/engineer.v58i1.7674)

---

## 2. Do now (time-sensitive)

- **Download the Global Water Watch archive.**
  - *What it is:* Global Water Watch (Deltares, WRI, WWF) gives water-area time series of reservoirs from Landsat and Sentinel-2. Its website says the service "will be made unavailable".
  - *The archive:* a **September 2026 archive** is on Zenodo (71,208 lakes and reservoirs, CC BY 4.0).
  - *Our tanks:* a helper's live query found **4 of our 5 reservoirs**, with 214–298 readings each from 2015 to 2026. Mahakanadarawa and the village tanks are missing.
  - *Use:* free area history to check our AI water maps.
  - **Effort: Low.** [Archive (Zenodo)](https://zenodo.org/records/22643338) · [Global Water Watch](https://www.globalwaterwatch.earth/)
- **Keep saving the daily reservoir sheet**, ideally automatically. Every missed day is lost.

---

## 3. Better methods for the core research questions

### RQ1: AI water maps (clouds, plants)

**Choosing the model**

| Idea | What it is and how it helps us | Effort | Source |
|---|---|---|---|
| Keep U-Net as a main model, not just a baseline | In the PANGAEA benchmark on Sen1Floods11, a plain U-Net scored **91.42 mIoU**, ahead of CROMA (90.89) and Prithvi (90.37). Copy their tests with 100%, 50% and 10% of the labels | Low | [PANGAEA](https://arxiv.org/html/2412.04204v2) |
| Compare the newer AI models fairly | Kaushik et al. (2025) compared Prithvi, Clay and DOFA with U-Net for flood mapping. Clay was best with very few labels (5 training images: Clay 0.64 vs Prithvi 0.24 mIoU) | Low | [arXiv 2511.01990](https://arxiv.org/abs/2511.01990) |
| TerraMind-base-Flood (IBM and ESA) | A ready flood model that takes **Sentinel-2 + Sentinel-1 + elevation over 4 dates**. Apache-2.0 licence. Closer to our needs than the Prithvi model, which takes only Sentinel-2 | Medium (GPU) | [Hugging Face](https://huggingface.co/ibm-esa-geospatial/TerraMind-base-Flood) |
| TerraTorch | One tool to fine-tune Prithvi, TerraMind, Clay and DOFA on the same labels, so the RQ1 comparison is fair | Low–Medium | [GitHub](https://github.com/IBM/terratorch) |

**Clouds**

| Idea | What it is and how it helps us | Effort | Source |
|---|---|---|---|
| Cloud Score+ (Google) | A clear-sky score for each pixel in every Sentinel-2 image, in Earth Engine. Partly cloudy days can still be used if the tank itself is clear | Low | [GEE catalog](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_CLOUD_SCORE_PLUS_V1_S2_HARMONIZED) |
| Separate outputs for water and cloud (WorldFloods design) | Use separate outputs: water/land, plant/open, cloud/clear. Mapping water under thin cloud then works better than with one combined output | Low | [Sci. Reports 2023](https://www.nature.com/articles/s41598-023-47595-7) |
| Cloud removal (AllClear dataset, UnCRtainTS model) | Fills cloudy pixels using several dates, with a measure of how sure each fill is. Only as a stretch goal | High | [AllClear](https://allclear.cs.cornell.edu/) · [UnCRtainTS](https://arxiv.org/abs/2304.05464) |

**Training data and labels**

| Idea | What it is and how it helps us | Effort | Source |
|---|---|---|---|
| More training data with radar and several dates | ImpactMesh-Flood: 24,578 samples from 206 flood events, with Sentinel-1, Sentinel-2 and elevation over 4 dates, CC BY 4.0, 73.4 GB. Kuro Siwo and MMFlood are radar flood datasets | Medium | [ImpactMesh](https://huggingface.co/datasets/ibm-esa-geospatial/ImpactMesh-Flood) · [Kuro Siwo](https://arxiv.org/abs/2311.12056) · [MMFlood](https://zenodo.org/records/6534637) |
| Fewer hand labels | The Sen1Floods11 paper found that **many automatic labels beat a few hand labels**. Label clear days automatically, hand-label only the hard cases (plants, glare), then teach a radar model for cloudy days | Medium | [Sen1Floods11](https://mlanthology.org/cvprw/2020/bonafilia2020cvprw-sen1floods11) · [Garg et al. 2023](https://arxiv.org/abs/2302.08180) |
| segment-geospatial (SAM) | A labelling helper: click on the water and it draws the outline. Speeds up our hand labels | Low | [samgeo](https://samgeo.gishub.org) |

**Water plants**

| Idea | What it is and how it helps us | Effort | Source |
|---|---|---|---|
| Water hyacinth in Sentinel-1 radar | Simpson et al. 2022 (India): radar change detection found hyacinth patches over about a quarter hectare, about 95% of the time. A Lake Victoria study reached 98.89% accuracy | Low–Medium | [Simpson 2022](https://shura.shu.ac.uk/30437/) · [Lake Victoria 2024](https://dspace.stir.ac.uk/handle/1893/36387) |
| Simple plant baselines from Sentinel-2 | Sentinel-2 + Random Forest or SVM works well for floating plants: India 2026, a **UoM study on Bolgoda Lake (2023)**, and Vietnam lotus mapping 2023. These are the baselines our AI must beat | Low | [India 2026](https://link.springer.com/article/10.1007/s11356-026-38245-2) · [UoM Bolgoda (search result only)](https://dl.lib.uom.lk/items/452e06e6-7b1e-4bac-bfc0-5278b5d59e02) · [Vietnam](https://doi.org/10.3390/w15050854) |
| L-band radar sees water under plants | Longer-wavelength L-band radar goes through plants better than Sentinel-1. ALOS-2 PALSAR-2 data (25 m, since 2014) is free in Earth Engine. **NISAR** (NASA–ISRO) L-band data became free in 2026. Test both on our 3 plant-covered tanks | Medium–High | [ALOS-2 in GEE](https://developers.google.com/earth-engine/datasets/catalog/JAXA_ALOS_PALSAR-2_Level2_2_ScanSAR) · [NISAR data notice](https://asf.alaska.edu/notices/nisar-l-band-data-now-publicly-available/) |

**Benchmarks and small tanks**

| Idea | What it is and how it helps us | Effort | Source |
|---|---|---|---|
| Ready-made water maps to compare with | NASA OPERA DSWx-HLS (30 m water map, from April 2023) and Copernicus Global Flood Monitoring (radar flood maps) | Low | [OPERA in GEE](https://developers.google.com/earth-engine/datasets/catalog/OPERA_DSWX_L3_V1_HLS) |
| Pixels that are part water, part land | The edges of a 6–25 ha tank are many 10 m pixels. Mapping a "water share" per pixel, or a sharper map from Sentinel-1 + Sentinel-2, can cut the area error | Medium–High | [Yin et al. 2024](https://doi.org/10.1016/j.jag.2024.104176) |

### RQ2: SWOT heights

| Idea | What it is and how it helps us | Effort | Source |
|---|---|---|---|
| A smarter "suspect" filter | Levenson et al. (2025) tested a custom filter (wse_u ≤ 0.1, wse_std ≤ 0.35, xovr_cal_q < 2, plus a median check) on 411 gauged lakes. It kept **82.7%** of the good readings, against **56.3%** for "quality_f = 0 only", which means a reading every **18.3 days instead of 25.2**. Test it on our 5 gauged reservoirs | Low | [Slides](https://swotst.aviso.altimetry.fr/fileadmin/user_upload/ST2025/Presentations/ST2025HS2-Improved_SWOT_surface_water_storage_monitoring_through_multi-sensor_harmonization.pdf) |
| Use SWOT's raw "pixel cloud" for small tanks | Studies in the Sahel, Brazil, Australia and Taiwan found pixel-cloud heights better than the lake product for small reservoirs: within 0.06–0.11 m in the Sahel, and under 10 cm in Taiwan. NASA has a step-by-step guide (notebook) for it | Medium | [Girard 2025 (abstract only)](https://esploro.umontpellier.fr/esploro/outputs/journalArticle/Performance-of-the-Surface-Water-and/99151831409311) · [Taiwan EGU26-5144](https://meetingorganizer.copernicus.org/EGU26/EGU26-5144.html) · [PO.DAAC tutorials](https://podaac.github.io/tutorials/quarto_text/SWOT.html) |
| Numbers to compare with from India | 52 Indian reservoirs: 13.64 cm relative error, and slightly better in the monsoon (GRL 2025) | Low | [GRL 2025](https://doi.org/10.1029/2025GL115804) |
| ICESat-2 to fix the gauge "zero" | NASA's ICESat-2 laser heights can tie a gauge's zero mark to sea level, so we can report SWOT's **true** error. 124 of 136 US gauges matched within 10 cm. It can also spot-check village tanks that have no gauge | Medium | [CNES/CLS slides](https://swotst.aviso.altimetry.fr/fileadmin/user_upload/ST2025/Presentations/ST2025HS1-Performance_assessment_of_the_water_surface_elevation_of_HR_SWOT_products_based_on_comparisons_with_in-situ_networks_and_in-flight_nadir_altimetry_missions.pdf) · [ATL13](https://nsidc.org/data/atl13/versions/7) |
| Clear validation rules | Match gauge and SWOT readings on the same or next day. Report the 68% error, bias and RMSE, plus "days per usable reading". Watch the height reference: pixel-cloud heights are on the ellipsoid and lake-product heights on the geoid | Low | Same slides as above |

### RQ3: storage curves

| Idea | What it is and how it helps us | Effort | Source |
|---|---|---|---|
| 3D-LAKES ready-made curves | Height–area curves for **510,530 lakes** from ICESat-2 + Landsat (Scientific Data 2025), searchable by SWOT lake ID. They extend below SWOT's lowest level, so we can check them against the Irrigation Department capacity tables for our 5 reservoirs. Small lakes under 1 km² mostly have none | Low | [Zenodo](https://zenodo.org/records/13107867) · [Paper](https://doi.org/10.1038/s41597-025-05911-y) |
| Ready scripts for "active storage" | Ragettli and Kreiner (2025) match Sentinel-2 water edges with ground heights and ICESat-2 to build a reservoir's curve. A template for village tanks with no survey | Medium | [Zenodo](https://zenodo.org/records/14979466) |
| Elevation maps of the dry tank bed | Free elevation maps: Copernicus GLO-30, FABDEM, AW3D30, NASADEM. Avisse et al. 2017 used DEMs + Landsat on 0.6–3.8 km² reservoirs (R² 0.69–0.84) | Medium | [HESS 2017](https://hess.copernicus.org/articles/21/6445/2017/) |
| One drone survey of a dry village tank | In Tamil Nadu, a stereo satellite elevation map taken at low water was within 2–8% of a drone survey's volumes. One dry-season drone survey of 1–2 village tanks would give the real bed shape and test Imbulana's equation | High (drone, permits) | [Vanthof et al. 2024](https://isprs-archives.copernicus.org/articles/XLVIII-3-2024/565/2024/) |
| Curve shapes and error bands | Compare power-law and "only-rising" curve fits. Use Sentinel-2 area rather than SWOT's own area: it was better linked to SWOT height (r = 0.56 vs 0.39) | Low | Levenson slides (RQ2 table) |

---

## 4. New things we could add, ranked by value for effort

1. **Check SWOT's own storage-change numbers (new contribution, Low).**
   - SWOT's lake product gives storage change (ds1_l, ds1_q) for **91% of its lakes**, but the SWOT team says the data "have not been thoroughly validated yet! Please use with caution!" and asks for ground data.
   - We have gauges, so we can be among the first to test them, on Sri Lankan tanks.
   - [Storage-change slides](https://swotst.aviso.altimetry.fr/fileadmin/user_upload/ST2025/Presentations/ST2025HS3_10_deFleuryPottierFjortoft_StorageChange.pdf)
2. **Cyclone Ditwah case study (Low–Medium).**
   - Our data already shows Hurulu Wewa rising about 4.6 m in four weeks (25 Nov – 23 Dec 2025).
   - Add the storage jump for every tank and all 402 SWOT lakes, check Sentinel-1 under the cyclone's clouds, and compare with Copernicus flood maps (EMSR851).
   - [EMSR851](https://mapping.emergency.copernicus.eu/news/flood-in-sri-lanka-emsr851/)
3. **Curves down to the tank bed, for "% full" (Low–Medium).** Use 3D-LAKES, the elevation maps and the Ragettli scripts in section 3, checked against the official capacity tables.
4. **Storage forecast, RQ4 (Medium).**
   - A UoM–Department of Meteorology team built **SLRainGrid-D05**, a 0.05° daily rainfall grid for Sri Lanka. IMERG did better daily and CHIRPS better monthly.
   - An Indian study forecast 79 reservoirs with a Temporal Fusion Transformer and beat LSTM/GRU.
   - It needs older gauge records.
   - [SLRainGrid EGU26-20641](https://meetingorganizer.copernicus.org/EGU26/EGU26-20641.html) · [EGU26-19119](https://meetingorganizer.copernicus.org/EGU26/EGU26-19119.html)
5. **Evaporation loss per tank (Low).**
   - Earth Engine's ERA5-Land daily data has evaporation bands, and the RAT tool uses the Penman method.
   - A global study found reservoirs hold "5%" of lake water but cause "16%" of lake evaporation.
   - This tells farmers how much water is lost to the sun.
   - [GLEV, Nature Comms 2022](https://doi.org/10.1038/s41467-022-31125-6)
6. **Paddy area vs storage, small add-on (Medium).**
   - In Anuradhapura, pre-season (kanna) meetings decide the area to cultivate from the water in the tank.
   - Rice-field maps of Sri Lanka exist (Özdoğan et al. 2025).
   - Compare season-start storage with the paddy area below our 5 reservoirs.
   - [Oka et al.](https://www.managementjournal.usamv.ro/pdf/vol.15_4/Art28.pdf) · [Özdoğan 2025](https://doi.org/10.3390/rs17173065)
7. **Silt and lost capacity (Medium).**
   - In the Palugaswewa cascade, "40 to 50 % of the tank storage capacity has been filled with sediments" (Kowshayini et al. 2023).
   - Compare our satellite curves with the original capacity tables.
   - [Kowshayini 2023](https://doi.org/10.4038/tar.v34i4.8671)
8. **Spill and breach watch for cascades (Medium–High).**
   - A JIRCAS study tested a water-level logger and level forecasts in a cascade near Anuradhapura (Izumi and Hori).
   - Ali (2025) shows upstream tank breaches affect downstream tanks.
   - A World Bank-backed Rs 5.8bn project will rehabilitate 296 irrigation tanks (September 2026).
   - The add-on would flag tanks near full before heavy rain is forecast.
   - [JARQ](https://www.jircas.go.jp/en/publication/jarq/2018s14) · [Ali 2025](https://engineer.sljol.info/articles/10.4038/engineer.v58i3.7702) · [Newswire](https://www.newswire.lk/2026/09/01/rs-5-8bn-world-bank-backed-project-to-rehabilitate-296-irrigation-tanks/)
9. **"Wide run" over all 402 SWOT lakes in the study area (Medium).** A Chinese Academy of Sciences team used SWOT lake data for 1,596 lakes, with "most errors within 10%".
   - [CAS news](https://english.cas.cn/newsroom/research-news/202603/t20260327_1154061.shtml)
10. **Compare with existing services (Low).** Put our 5 reservoirs side by side with GloLakes, Global Water Watch and RAT 3.0. The Reservoir Assessment Tool (University of Washington, GPL v3) has a SWOT plugin, but it is heavy to run.
    - [RAT](https://github.com/UW-SASWE/RAT)
11. **Drought indicator (Low–Medium).** Compare each tank's storage with its own history. IWMI's South Asia drought monitor has no storage layer.
12. **Voice or SMS bulletin for farmer organisations (Medium).** Dialog's Govi Mithuru already sends voice advice to "over 800,000 farmers".
    - [Govi Mithuru](https://www.dialog.lk/govi-mithuru-farmers-friend)
13. **Low-cost water-level loggers at 1–2 village tanks (Medium).** They would give continuous ground truth for RQ2, following the JIRCAS precedent.
14. **AI that cleans SWOT's raw pixel cloud (High).** Tourian et al. (2025) re-classify pixel-cloud points with a CNN. This is a research project in its own right.

---

## 5. Sri Lankan data, studies and partners

**Free data we can use now**

- **Irrigation Department live river gauges and flood layers (ArcGIS).**
  - *Layers:* `gauges_2_view` (40 gauges, water level and rain), `hydrostations` (42 stations with alert levels) and `Flood__Nov2025` (Ditwah flood outlines, including Yan Oya and Kala Oya).
  - *Catch:* the gauge layer keeps only about a week, so it must be saved regularly.
  - *Use:* check releases and spills downstream, and test our water maps on the Ditwah floods.
  - [gauges_2_view](https://services3.arcgis.com/J7ZFXmR8rSmQ3FGf/ArcGIS/rest/services/gauges_2_view/FeatureServer/0) · [Flood__Nov2025](https://services3.arcgis.com/J7ZFXmR8rSmQ3FGf/ArcGIS/rest/services/Flood__Nov2025/FeatureServer/0)
- **Hydrological Annual** (Irrigation Department, every year since 1960).
  - *Contents:* monthly rainfall, pan evaporation (one station is at Huruluwewa) and river flow at Horowpothana and Thanthirimale.
  - *Use:* the rain and evaporation parts of a water balance.
  - [2021/22 edition](https://irrigation.gov.lk/web/images/Publications/Hydrologyical_Annual_2021-22.pdf)
- **Administration Report 2022, Part II** (Irrigation Department).
  - *Official capacities:* Huruluwewa 67.84, Nachchaduwa 55.69, Nuwarawewa 44.5, Mahakanadarawa 44.7 and Manankattiya 6.02 MCM.
  - *Water released each season:* for example, Mahakanadarawa released 21.20 MCM in Maha 2021/22.
  - *Drinking water:* the water board (NWSDB) takes 13,563 m³ a day from Nuwara Wewa.
  - *Use:* outflow figures and capacity checks.
  - [Report](https://www.irrigation.gov.lk/web/images/Downloads/Publications/Administration_Report_2022_Part_II.pdf)
- **Scheme maps (PDF).**
  - *Nachchaduwa:* names the nearby small tanks.
  - *Manankattiya:* full supply level 136.30 m MSL, 4,880 acre-feet, 735 acres of water when full.
  - *Use:* one fixed point on each curve, and local tank names.
  - [Nachchaduwa](https://www.irrigation.gov.lk/web/images/Scheme-Maps/Anuradhapura/Nachchaduwa/Nachchaduwa_Scheme.pdf) · [Manankattiya](https://www.irrigation.gov.lk/web/images/Scheme-Maps/Anuradhapura/HuruluWewa/Manankattiya_Map.pdf)
- **A 1970s Mahakanadarawa study (JIRCAS, Gomez).**
  - *Contents:* capacity–depth and area–depth equations, seepage and pan evaporation, with full supply level 94.87 m and maximum capacity 43.5 million m³.
  - *Use:* compare this old curve with our SWOT curve to see **how much storage silt has taken in 50 years**, which would be a new result.
  - *Note:* the text was scanned and converted (OCR), so check the numbers against the scan.
  - [JIRCAS PDF](https://www.jircas.go.jp/sites/default/files/publication/tars/tars9-_209-224.pdf)
- **JICA/NWSDB 2013 report.**
  - *Contents:* Mahakanadarawa also supplies drinking water, 6,700 m³ a day in 2016, planned to rise to 18,800 m³ a day by 2034.
  - *Use:* another outflow figure, and NWSDB as a user of our results.
  - [JICA report](https://openjicareport.jica.go.jp/pdf/12111100.pdf)

**A risk and an opportunity: Mahaweli water transfers**
- **The canal.** An ADB report (2015) says the Upper Elahera Canal "connects the Moragahakanda Reservoir to the existing reservoirs: Huruluwewa, Manankattiya, Eruwewa and Mahakanadarawa". Its tunnel was finished in August 2026.
  - [ADB report](https://ewsdata.rightsindevelopment.org/files/documents/01/ADB-47381-001_sX7uqyc.pdf)
- **Diversions already happening.** The Mahaweli Authority's weekly meeting notes list diversions to Huruluwewa and the Nachchaduwa feeder canal.
  - [Weekly notes](https://mahaweli.gov.lk/WMS%20DATA/Weekly%20Operational%20Meeting.pdf)
- **What it means for us:** some storage changes come from **transfers, not rain**. The forecast (RQ4) and any water balance must allow for this. The project could also record the first big transfers through the new canal.

**Studies to cite**

| Study | What it found | Why it matters to us | Link |
|---|---|---|---|
| **ICESat-2 on nine Sri Lankan reservoirs** (*Engineer* 58(1), 2025; first author from the University of Peradeniya) | It includes Nachchaduwa and Nuwara Wewa. ICESat-2 read **about 0.4 m above the gauges** (0.415 m at Nachchaduwa, 0.382 m at Nuwara Wewa), with correlation above 0.99 | Expect a similar fixed offset between SWOT and the gauges. For RQ2, compare **changes**, or fix the offset first | [Paper](https://engineer.sljol.info/articles/10.4038/engineer.v58i1.7674) |
| **Perera and Jayaweera, 2020** | Sentinel-1/2 water area in Google Earth Engine compared with Irrigation Department volumes: R² 0.73 (Kaudulla), 0.94 (Senanayake Samudra), 0.67 (Lunugamwehera). Optical images failed under cloud | A Sri Lankan "area only" baseline that our method should beat | [Paper](https://archive.cmb.ac.lk/items/84f64851-eaa9-4b8c-babd-5d18f130c1ee) |
| **Kowshayini et al., 2023** | In the Palugaswewa cascade, 40–50% of tank capacity is filled with silt | Village-tank curves must be measured on today's silted beds | [Paper](https://tar.sljol.info/articles/8671) |
| **Izumi and Hori, 2021** (JARQ) | The Kiulekada cascade (10 small tanks, 50 km north-east of Anuradhapura) has an online water-level logger running every hour since September 2017. Its server is at the Provincial Irrigation Department office in Anuradhapura | Ask for the logger records, to check our village-tank results | [Paper](https://www.jircas.go.jp/sites/default/files/publication/jarq/jarq55-1_-_5.pdf) |
| **IWMI and IIMI cascade studies** | More than 90% of small tanks lie in cascades. They cite a map of 240 cascades in Anuradhapura District, and a daily water-balance model for a cascade (IWMI Research Report 48) | Cascade IDs, tank positions and a model template | [Panabokke 2002](https://hdl.handle.net/10568/36415) · [IIMI 1997](https://hdl.handle.net/10568/39590) · [RR 48](https://hdl.handle.net/10568/39835) |
| **UoM theses** (search result only) | Catchment models for Huruluwewa and Nachchaduwa, and "Hydrology of Mahakanadarawa scheme" (1985) | Inflow models from our own university | [UoM library](https://dl.lib.uom.lk/items/066c1666-648b-47f2-8a8a-52612fc51efb) |

**To finish the "cascade and local names" task**
1. **Ask the Department of Agrarian Development for its tank cascade boundary layer.** FAO's 2017 map of the Palugaswewa heritage site shows the layer exists.
   - [FAO map](http://www.fao.org/fileadmin/templates/giahs_assets/GIAHS_test/02_GIAHS_around_the_world/01_Designated_sites/02_Asia_and_the_Pacific/07_Sri_Lanka/Mar_6th_rev_GIAHS_Hydrological_Connectivity_of_the_Tank_Cascades.pdf)
2. **Buy the Survey Department's 1:10,000 map sheet** (LKR 250) for the official tank names.
   - [Map portal](https://app.survey.gov.lk/mapportal/product.php?id=430)
3. **Look at the Copernicus Cyclone Ditwah map for "Mihintale".** It's a very-high-resolution image from 21 Dec 2025, just south of Mahakanadarawa, near some of our village tanks. It shows tank bunds and the links between tanks.
   - [EMSR851](https://rapidmapping.emergency.copernicus.eu/backend/dashboard-api/public-activations/?code=EMSR851)

**Possible partners**
- **UMCSAWM, UoM's water centre (UNESCO Madanjeet Singh Centre), and the Department of Meteorology:** they made SLRainGrid, and they're inside our own university.
- **The Provincial Irrigation Department, Anuradhapura:** they hold the Kiulekada logger records.
- **NWSDB, the water board:** it takes drinking water from Mahakanadarawa and Nuwara Wewa.
- **The University of Peradeniya ICESat-2 team.**
- **The Department of Agrarian Development:** it holds the cascade layer and manages 15,958 small tanks.

**Cyclone Ditwah facts for the problem statement**
- The Department of Agrarian Development reported 1,160 of its 15,958 small tanks damaged, 125 of them severely. Anuradhapura was among the three worst districts.
  - [News 1st, 5 Dec 2025](https://newsfirst.lk/2025/12/05/1-160-small-tanks-damaged-after-ditwah)
- The World Bank estimated US$4.1 billion of damage.
  - [World Bank](https://www.worldbank.org/en/news/press-release/2025/12/22/damage-from-cyclone-ditwah-in-sri-lanka-estimated-at-4-1-billion)

---

## 6. Ready-made tools and datasets

**Earth Engine collections**
- `COPERNICUS/S1_GRD`
- `COPERNICUS/S2_SR_HARMONIZED`
- `GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED`
- `GOOGLE/DYNAMICWORLD/V1`
- `JRC/GSW1_4/MonthlyHistory`
- `OPERA/DSWX/L3_V1/HLS`
- `UCSB-CHG/CHIRPS/DAILY`
- `NASA/GPM_L3/IMERG_V07`
- `ECMWF/ERA5_LAND/DAILY_AGGR`
- `COPERNICUS/DEM/GLO30_2024_1`
- `JAXA/ALOS/AW3D30/V4_1`
- `NASA/NASADEM_HGT/001`

**Global lake datasets**
- GloLakes
- Global Water Watch (archive)
- GRSAD / GRS / GRBD (Texas A&M): [lab page](https://hgao.engr.tamu.edu/research/)
- ReaLSAT: lakes over 0.1 km², 1984–2020
- HydroLAKES + GLOBathy: modelled depths only
- SWOT Prior Lake Database (Light)

**SWOT tools**
- PO.DAAC tutorials, including pixel-cloud area
- pixcdust: downloads and clips the pixel cloud
- SWODLR: custom SWOT rasters
- CNES SWOT Hydrology Toolbox (old Python)

**AI and labelling**
- TerraTorch
- TorchGeo
- segmentation-models-pytorch
- segment-geospatial

**Dashboard**
- TiTiler: map tiles for FastAPI
- leafmap / geemap: quick demo apps

---

## 7. My suggestion

**Add to the proposal now** (cheap and strong):
- the corrections in section 1;
- the smarter suspect filter and SWOT's new `qual_f_b` flag for RQ2;
- a U-Net vs TerraMind vs Prithvi comparison with 100/50/10% labels for RQ1;
- Cloud Score+;
- the 3D-LAKES check;
- two add-ons: **checking SWOT's own storage change** and the **Cyclone Ditwah case study**. Both use data we already have.

**Data to collect now** (free, and some of it disappears):
- the Global Water Watch archive;
- the daily reservoir sheet;
- the live river-gauge layer, which keeps only a week;
- the Hydrological Annual and the Administration Reports;
- the 1970s Mahakanadarawa curve.

Also request the Agrarian Development cascade layer to finish the cascade and names task.

**Note as a known risk:** water transfers from the Upper Elahera Canal and Mahaweli diversions.

**Keep as stretch goals:**
- the forecast (RQ4) with SLRainGrid/IMERG;
- evaporation;
- paddy vs storage;
- silt;
- the L-band test;
- the wide run over 402 lakes;
- the bulletin;
- the loggers.

**Don't add:**
- a full RAT run;
- cloud removal;
- the pixel-cloud CNN.

All three are big projects on their own.

---

## Spot checks (re-opened on 26 Oct 2026)

**SWOT team slides**
- The 10 cm requirement and 25 cm goal.
- 14 vs 12 cm with and without suspect readings.
- The custom filter (82.7% vs 56.3%; 18.3 vs 25.2 days).
- Storage change for 91% of lakes, "not thoroughly validated".

**Sri Lankan sources**
- The ICESat-2 nine-reservoir study (*Engineer* 58(1)).
- Palugaswewa "40 to 50 %" silt.
- The JIRCAS cascade water-level paper.
- Ali 2025 on cascade breaches.
- The World Bank 296-tank project.
- SLRainGrid (EGU26-20641).

**Datasets and services**
- The Global Water Watch shutdown notice and its Zenodo archive (September 2026, CC BY 4.0).
- 3D-LAKES on Zenodo.
- The Ragettli scripts.

**Satellites and models**
- The Sentinel-1 six-day repeat.
- The NISAR data notice.
- TerraMind-base-Flood.
- The Thai abstract EGU26-4154: no SWOT, no numbers.
- EGU26-19119.

**Section 5 (Sri Lanka)**
- The ADB quote about the Upper Elahera Canal.
- The Mahaweli weekly notes on diversions.
- The 1,160 damaged tanks.
- The Perera and Jayaweera paper.
- The EMSR851 "Mihintale" map area.

**One claim dropped:** a helper said "73.3% of SWOT values are suspect". I couldn't find this in the slides, so it's not used.
