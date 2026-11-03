# SWOT Tank Water Tracker: working files

Work for the IS4990 project "SWOT Tank Water Tracker for Sri Lanka" (Group 6, JARVIS), up to 3 November 2026.

**The idea:** estimate how much water each Sri Lankan tank holds from free satellite data.
- SWOT gives the water height.
- AI water maps from Sentinel-1 and Sentinel-2 give the water area.
- Together they give each tank a height–area–volume curve, and then its storage every 5–12 days.

## What is here

| Folder | Contents |
|---|---|
| `module-a/` | Module A (RQ1), 1 to 3 Nov 2026: water maps under cloud at 117 reservoirs, the retraining for tank beds, and the comparison with the last clear photo. Start with `module-a/README.md` |
| `proposal-drafts/` | The official 4-page proposal as submitted on 30 Oct 2026 (`Project Proposal - submitted 30 Oct 2026 (public copy).pdf`, without the title page), the earlier full proposal and drafts, and their text in `export/` |
| `feasibility-check/` | The October 2026 data checks: results in `results/` and working files in `work/`. Start with `results/SUMMARY.md` |
| `model-check/` | The 28 October test of the models of Members B and C on real data (trust screen, storage model, forecast): result tables. Start with `model-check/README.md` |
| `cascade-check/` | Do the eight village tanks share one cascade? (No.) A flow trace on the elevation map, checked against the department's catchments, and a real five-tank cascade south of Nachchaduwa |
| `data-checks/` | First data access checks |
| `presentation/` | The 15-slide proposal talk submitted on 30 Oct 2026, as PowerPoint and PDF (without the speaker notes, which we keep private, and with the owners shown as Member A, B and C), the pictures of the earlier 21-slide deck, and `tank-volume-explainer.html` (an interactive model tank) |
| `research/` | A summary of all the past research we used, and ideas to strengthen and widen the scope (26 Oct 2026), each with source links |

**The code and the Irrigation Department's tables are kept private** (since 3 Nov 2026). Script names in the write-ups refer to those private scripts.

**Project website:** https://swot.mehara.work (code in the private repo [jarvis-uom/swot-tank-tracker-site](https://github.com/jarvis-uom/swot-tank-tracker-site)).

## Main results so far

**Feasibility check: GO, with changes.**
- Sentinel-1 radar and a water-plant class are now core parts of the plan.

**Study area: 13 tanks east of Anuradhapura.**
- 5 major reservoirs with daily gauges: Hurulu Wewa, Mahakanadarawa, Manankattiya, Nachchaduwa and Nuwara Wewa.
- 8 village tanks near Mahakanadarawa.
- SWOT passes 273 and 092 cover all 13 tanks.

**Sri Lanka-wide check:** `feasibility-check/results/A7_srilanka_lakes.csv`.
- It ranks the 187 lakes of 1 km² or more by best-quality SWOT readings.
- Hurulu Wewa is 6th, and all 5 study reservoirs are in the top 22.

**How to check every SWOT number in a browser:** `feasibility-check/results/HOW_TO_CHECK_IN_BROWSER.md`.

**Proposal update, 26 October 2026:**
- It cites the Irrigation Department's official Reservoir Status page for the daily gauge data.
- It separates SWOT's 10 cm requirement (lakes over 1 km²) from its 25 cm goal (smaller lakes).
- It adds three related studies: a 2026 Thai study without SWOT, a Sri Lankan Sentinel-1 study, and a Sri Lankan ICESat-2 study that found gauges read about 0.4 m below the satellite.
- Team roles: each member owns one core research question. A: AI water maps (RQ1). B: SWOT and gauges (RQ2). C: storage curves and dashboard (RQ3), plus the forecast (RQ4) if time allows.

**Deck update, 26 October 2026:** new slides on why the project matters, the problem we solve, and who does what.

**Deck update, 28 October 2026:** one slide per member (the data analysis and the technical work) and a "Datasets and links" slide with every dataset's link.

**Plan update, 29 October 2026:** the three models now form one chain.
- A: the water map (RQ1).
- B: SWOT and storage (RQ2 and RQ3). A trust screen cleans the SWOT readings, and a gradient-boosting model trained on about 100 gauged reservoirs turns SWOT height and water area into % full.
- C: the forecast and the system (RQ4).
- The proposal draft and the talk (slides 14, 15, 17, 18 and 19) follow the new plan.

**Approved, 29 October 2026:** both supervisors approved the proposal. The final proposal is `proposal-drafts/SWOT Tank Water Tracker for Sri Lanka.pdf` (the older drafts are kept for reference). On 30 October it got 7 small factual fixes before the upload.
- RQ3 now compares the storage model with one area–storage formula for all tanks.
- New baselines: the Brazil SWOT filter (Lappicy et al., 2026) for the trust screen, and NASA's OPERA water maps for the water map.
- IMERG for daily rain, CHIRPS for monthly checks.
- The eight village tanks are not one cascade (`cascade-check/`).
- The slides in `presentation/` match the final proposal.

**Submitted, 30 October 2026:** the official 4-page proposal (`proposal-drafts/Project Proposal - submitted 30 Oct 2026 (public copy).pdf`) and the 15-slide talk (`presentation/`). The presentation date is still to be announced.

**Module A, 1 to 3 November 2026** (`module-a/README.md`):
- Under cloud, only radar works. At about 100 reservoirs, the radar-only U-Net reads storage within 11 to 12% of capacity, against 20 to 22% with no model.
- Retrained on our own reservoirs (a trial run), it reads storage within 8.9%, and it beats the last clear photo at 76% of reservoirs.
- Our two reservoir image datasets are public on Kaggle.

## Not included

These are left out. Each item says how to get it again.
- **SWOT lake files** (about 125 MB). They need a free NASA Earthdata login.
- **SWOT series from Hydrocron** (about 10 MB). Hydrocron serves them free.
- **Sentinel images.** They are free from Earth Search and Microsoft Planetary Computer. Module A's chips are on Kaggle (`module-a/README.md`).
- **The code and the Irrigation Department's tables.** They are kept private.
- **Global Water Watch archive** (about 530 MB). The Global Water Watch site is going offline; its September 2026 archive is on Zenodo: https://zenodo.org/records/22658942 (CC BY 4.0).

## Daily reservoir sheet

The private repo `jarvis-uom/water-data-logger` saves the department's sheet three times a day with GitHub Actions, along with the river gauges in our three basins.
- The sheet shows only the current day, so a day that is not saved is lost.
- It is the data behind the Irrigation Department's official "Reservoir Status" dashboards: https://irrigation.gov.lk/web/index.php?option=com_content&view=article&id=84&Itemid=201&lang=en

## Data sources and credits

- **SWOT:** NASA/JPL and CNES, Lake Single-Pass product version D, from NASA PO.DAAC (Hydrocron).
- **Sentinel-1 and Sentinel-2:** contains modified Copernicus Sentinel data 2020–2026.
- **Irrigation Department of Sri Lanka:** reservoir layer and daily reservoir status.
- **Tank outlines:** © OpenStreetMap contributors. Sri Lanka boundary: geoBoundaries.
- **SWOT illustration:** NASA/JPL-Caltech.
