# SWOT Tank Water Tracker: working files

Work for the IS4990 project "SWOT Tank Water Tracker for Sri Lanka" (Group 6, JARVIS), up to 26 October 2026.

**The idea:** estimate how much water each Sri Lankan tank holds from free satellite data.
- SWOT gives the water height.
- AI water maps from Sentinel-1 and Sentinel-2 give the water area.
- Together they give each tank a height–area–volume curve, and then its storage every 5–12 days.

## What is here

| Folder | Contents |
|---|---|
| `proposal-drafts/` | Proposal drafts (PDF; the latest is 28 Oct 2026), their text in `export/`, and the scripts that build the PDF |
| `feasibility-check/` | The October 2026 data checks: scripts in `work/`, results in `results/`. Start with `results/SUMMARY.md` |
| `data-checks/` | First data access checks, and the saved Irrigation Department reservoir sheets |
| `presentation/` | 21-slide deck for the supervisor meeting and a PDF of the slides (both without the speaker notes, which we keep private), the scripts and pictures that build the deck, and `tank-volume-explainer.html` (an interactive model tank) |
| `research/` | A summary of all the past research we used, and ideas to strengthen and widen the scope (26 Oct 2026), each with source links |

**Project website:** https://swot.mehara.work (code in [jarvis-uom/swot-tank-tracker-site](https://github.com/jarvis-uom/swot-tank-tracker-site)).

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

## Not included

These are left out. Each item says how to get it again.
- **SWOT lake files** (about 125 MB). They need a free NASA Earthdata login.
- **SWOT series from Hydrocron** (about 10 MB). Run `feasibility-check/work/a2_fetch.py`.
- **Sentinel images.** The presentation scripts read them straight from Earth Search and Microsoft Planetary Computer.
- **The Python environment and the Node build tools.** Run `npm install` in `presentation/build/`.
- **Global Water Watch archive** (about 530 MB). The Global Water Watch site is going offline; its September 2026 archive is on Zenodo: https://zenodo.org/records/22658942 (CC BY 4.0).

## Rebuild

1. **Python 3.13 packages:**
   - earthaccess, geopandas, rasterio, pystac-client, planetary-computer
   - scipy, matplotlib, pymupdf, markdown, python-pptx, qrcode, markitdown[pptx]
2. **Feasibility check:** run the scripts in `feasibility-check/work/` in order, from `a1_lake_ids.py` to `a7_srilanka.py`.
3. **Deck:**
   - Run `p1_maps.py` to `p6_study_map.py` in `presentation/build/`.
   - Then run `node build_deck.js`.
   - `render.ps1` exports the slides to pictures with PowerPoint.
   - `p7_phone_notes.py` prints each slide's picture with its speaker notes, one A5 page per slide, for reading on a phone.
   - The speaker notes are kept private, so they are not in this repo: `build_deck.js` here builds the deck without them.
4. **Daily reservoir sheet:** saved automatically three times a day by the private repo `jarvis-uom/water-data-logger` (GitHub Actions), which also saves the river gauges in our three basins.
   - The sheet shows only the current day, so a day that is not saved is lost.
   - It is the data behind the Irrigation Department's official "Reservoir Status" dashboards: https://irrigation.gov.lk/web/index.php?option=com_content&view=article&id=84&Itemid=201&lang=en
   - `data-checks/check_access.py` can still save one copy by hand.
5. **Proposal PDF:** run `proposal-drafts/build/build_pdf.py`. It reads the proposal's Markdown from `export/` and prints the PDF with Microsoft Edge.

## Data sources and credits

- **SWOT:** NASA/JPL and CNES, Lake Single-Pass product version D, from NASA PO.DAAC (Hydrocron).
- **Sentinel-1 and Sentinel-2:** contains modified Copernicus Sentinel data 2023–2026.
- **Irrigation Department of Sri Lanka:** reservoir layer and daily reservoir status.
- **Tank outlines:** © OpenStreetMap contributors. Sri Lanka boundary: geoBoundaries.
- **SWOT illustration:** NASA/JPL-Caltech.
