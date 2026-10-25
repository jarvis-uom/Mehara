# How to check the SWOT numbers yourself (in a browser)

Every SWOT reading count in `A2_heights_by_lake.csv` and in the proposal comes from NASA's public **Hydrocron** service. You can open it in Chrome or Edge. No login is needed.

Checked on 25 Oct 2026: NASA's answer gives exactly the counts in the table below for all 13 tanks.

## Quick way: see it on a map website

The **SWOT Lake and River Explorer** shows SWOT lakes on a map: https://mibrechb.github.io/swot-data-explorer/. Click a lake and it draws NASA's SWOT water heights over time. It was made by SOS-Water, a research project funded by the EU (Horizon Europe), not by us. It takes the lake outlines from Hydroweb.next (the French space agency's SWOT portal) and the readings, live, from NASA's Hydrocron service, version D: the same data we used. No login. Checked on 25 Oct 2026: all 13 study tanks are on its map and their charts load.

1. Open the site and close the welcome box. Keep **Lake** selected.
2. Click the search (magnifier) button on the map and type the tank's name: Hurulu Wewa, Nuwara Wewa, Nachchaduwa, Mahakanadarawa or Manankattiya. Pick the first result.
3. Click the tank's outline. The panel shows its lake ID and a chart of its water height since July 2023.
4. Open **Quality settings** and turn off **Show suspect observations**. Now only NASA's "good" readings (quality flag 0) are left.
5. **Download** saves the readings as a CSV file.

What the charts show (readings up to 25 Oct 2026; later passes will add points):

| Tank | Good (flag 0) | Suspect (flag 1) | Our best quality |
|---|---|---|---|
| Nachchaduwa | 58 | 27 | 55 |
| Hurulu Wewa | 73 | 17 | 71 |
| Nuwara Wewa | 55 | 17 | 53 |
| Mahakanadarawa | 65 | 29 | 63 |
| Manankattiya | 67 | 21 | 57 |
| Each village tank | 0 | 47–90 | 0 |

Our "best quality" count is a little lower because we also drop readings that fail two more checks: crossover calibration and dark water. This site does not show those two flags. The NASA links in part 1 show all three flags.

## 1. Open NASA's data for a tank

Click a tank name in the table. Each link asks Hydrocron (NASA PO.DAAC) for every SWOT reading of that tank from 1 July 2023 to 25 October 2026, from the SWOT lake product, version D.

- If the page shows one long line of text, tick **Pretty-print** at the top.
- Do not change the end date (2026-10-25). With a later date, new passes are added and the counts grow.
- To show that this is NASA's own service, open Hydrocron's home page: https://podaac.github.io/hydrocron/

| Tank | NASA lake ID | Readings | Best quality | Usable | NASA's name for it |
|---|---|---|---|---|---|
| [Nachchaduwa](https://soto.podaac.earthdatacloud.nasa.gov/hydrocron/v1/timeseries?feature=PriorLake&feature_id=4580051323&start_time=2023-07-01T00:00:00Z&end_time=2026-10-25T00:00:00Z&output=geojson&collection_name=SWOT_L2_HR_LakeSP_D&fields=lake_id,lake_name,time_str,wse,quality_f,xovr_cal_q,dark_frac,p_lat,p_lon) | 4580051323 | 100 | 55 | 76 | none (see part 4) |
| [Hurulu Wewa](https://soto.podaac.earthdatacloud.nasa.gov/hydrocron/v1/timeseries?feature=PriorLake&feature_id=4580013372&start_time=2023-07-01T00:00:00Z&end_time=2026-10-25T00:00:00Z&output=geojson&collection_name=SWOT_L2_HR_LakeSP_D&fields=lake_id,lake_name,time_str,wse,quality_f,xovr_cal_q,dark_frac,p_lat,p_lon) | 4580013372 | 101 | 71 | 77 | HURULU WEWA;IBBIGE WEWA |
| [Nuwara Wewa](https://soto.podaac.earthdatacloud.nasa.gov/hydrocron/v1/timeseries?feature=PriorLake&feature_id=4580009523&start_time=2023-07-01T00:00:00Z&end_time=2026-10-25T00:00:00Z&output=geojson&collection_name=SWOT_L2_HR_LakeSP_D&fields=lake_id,lake_name,time_str,wse,quality_f,xovr_cal_q,dark_frac,p_lat,p_lon) | 4580009523 | 101 | 53 | 62 | NUWARA WEWA |
| [Mahakanadarawa](https://soto.podaac.earthdatacloud.nasa.gov/hydrocron/v1/timeseries?feature=PriorLake&feature_id=4580006192&start_time=2023-07-01T00:00:00Z&end_time=2026-10-25T00:00:00Z&output=geojson&collection_name=SWOT_L2_HR_LakeSP_D&fields=lake_id,lake_name,time_str,wse,quality_f,xovr_cal_q,dark_frac,p_lat,p_lon) | 4580006192 | 100 | 63 | 85 | MAHAKANADARAWA WEWA |
| [Manankattiya](https://soto.podaac.earthdatacloud.nasa.gov/hydrocron/v1/timeseries?feature=PriorLake&feature_id=4580013332&start_time=2023-07-01T00:00:00Z&end_time=2026-10-25T00:00:00Z&output=geojson&collection_name=SWOT_L2_HR_LakeSP_D&fields=lake_id,lake_name,time_str,wse,quality_f,xovr_cal_q,dark_frac,p_lat,p_lon) | 4580013332 | 100 | 57 | 71 | none (see part 4) |
| [Village 1](https://soto.podaac.earthdatacloud.nasa.gov/hydrocron/v1/timeseries?feature=PriorLake&feature_id=4580005172&start_time=2023-07-01T00:00:00Z&end_time=2026-10-25T00:00:00Z&output=geojson&collection_name=SWOT_L2_HR_LakeSP_D&fields=lake_id,lake_name,time_str,wse,quality_f,xovr_cal_q,dark_frac,p_lat,p_lon) | 4580005172 | 79 | 0 | 73 | none |
| [Village 2](https://soto.podaac.earthdatacloud.nasa.gov/hydrocron/v1/timeseries?feature=PriorLake&feature_id=4580010522&start_time=2023-07-01T00:00:00Z&end_time=2026-10-25T00:00:00Z&output=geojson&collection_name=SWOT_L2_HR_LakeSP_D&fields=lake_id,lake_name,time_str,wse,quality_f,xovr_cal_q,dark_frac,p_lat,p_lon) | 4580010522 | 79 | 0 | 73 | none |
| [Village 3](https://soto.podaac.earthdatacloud.nasa.gov/hydrocron/v1/timeseries?feature=PriorLake&feature_id=4580006282&start_time=2023-07-01T00:00:00Z&end_time=2026-10-25T00:00:00Z&output=geojson&collection_name=SWOT_L2_HR_LakeSP_D&fields=lake_id,lake_name,time_str,wse,quality_f,xovr_cal_q,dark_frac,p_lat,p_lon) | 4580006282 | 80 | 0 | 73 | none |
| [Village 4](https://soto.podaac.earthdatacloud.nasa.gov/hydrocron/v1/timeseries?feature=PriorLake&feature_id=4580010542&start_time=2023-07-01T00:00:00Z&end_time=2026-10-25T00:00:00Z&output=geojson&collection_name=SWOT_L2_HR_LakeSP_D&fields=lake_id,lake_name,time_str,wse,quality_f,xovr_cal_q,dark_frac,p_lat,p_lon) | 4580010542 | 94 | 0 | 86 | none |
| [Village 5](https://soto.podaac.earthdatacloud.nasa.gov/hydrocron/v1/timeseries?feature=PriorLake&feature_id=4580011762&start_time=2023-07-01T00:00:00Z&end_time=2026-10-25T00:00:00Z&output=geojson&collection_name=SWOT_L2_HR_LakeSP_D&fields=lake_id,lake_name,time_str,wse,quality_f,xovr_cal_q,dark_frac,p_lat,p_lon) | 4580011762 | 82 | 0 | 77 | none |
| [Village 6](https://soto.podaac.earthdatacloud.nasa.gov/hydrocron/v1/timeseries?feature=PriorLake&feature_id=4580011422&start_time=2023-07-01T00:00:00Z&end_time=2026-10-25T00:00:00Z&output=geojson&collection_name=SWOT_L2_HR_LakeSP_D&fields=lake_id,lake_name,time_str,wse,quality_f,xovr_cal_q,dark_frac,p_lat,p_lon) | 4580011422 | 96 | 0 | 90 | none |
| [Village 7 (Bulankulama)](https://soto.podaac.earthdatacloud.nasa.gov/hydrocron/v1/timeseries?feature=PriorLake&feature_id=4580007392&start_time=2023-07-01T00:00:00Z&end_time=2026-10-25T00:00:00Z&output=geojson&collection_name=SWOT_L2_HR_LakeSP_D&fields=lake_id,lake_name,time_str,wse,quality_f,xovr_cal_q,dark_frac,p_lat,p_lon) | 4580007392 | 50 | 0 | 47 | none |
| [Village 8](https://soto.podaac.earthdatacloud.nasa.gov/hydrocron/v1/timeseries?feature=PriorLake&feature_id=4580043692&start_time=2023-07-01T00:00:00Z&end_time=2026-10-25T00:00:00Z&output=geojson&collection_name=SWOT_L2_HR_LakeSP_D&fields=lake_id,lake_name,time_str,wse,quality_f,xovr_cal_q,dark_frac,p_lat,p_lon) | 4580043692 | 94 | 0 | 89 | none |

"Readings" leaves out the rows marked `no_data`. Those are passes where SWOT saw nothing, so NASA's `hits` number at the top of the page is a little higher.

## 2. Read one reading

Each reading on the page looks like this (Hurulu Wewa, first pass):

```
"lake_id": "4580013372",
"lake_name": "HURULU WEWA;IBBIGE WEWA",
"time_str": "2023-07-30T23:15:33Z",   date and time of the pass
"wse": "127.586",                     water height, metres above sea level
"quality_f": "0",                     NASA's quality flag
"xovr_cal_q": "0",                    calibration flag
"dark_frac": "0.064849",              share of the tank that was "dark water"
"p_lat": "8.208651",                  NASA's centre point of the lake
"p_lon": "80.72828",
```

## 3. Which readings count

- **Best quality:** `quality_f` = 0, `xovr_cal_q` = 0 or 1, and `dark_frac` 0.1 or less. This is the rule of Köhn and Nielsen (2026), at the end of Section 2.1 "SWOT": https://doi.org/10.5194/hess-30-6443-2026
- **Usable:** our own wider rule. It is the same, but `quality_f` may also be 1 ("suspect"). The paper does not use it. Testing it against the gauges is part of RQ2.
- What the flags mean is in NASA's product guide for this data, page 23: [SWOT LakeSP Product Description Document](https://archive.podaac.earthdata.nasa.gov/podaac-ops-cumulus-docs/web-misc/swot_mission_docs/pdd/SWOT-TN-CDM-0673-CNES_Product_Description_L2_HR_LakeSP_20250307_RevC_signed.pdf)
  - `quality_f`: 0 good, 1 suspect, 2 degraded, 3 bad
  - `xovr_cal_q`: 0 good, 1 suspect, 2 bad (missing) calibration
  - `dark_frac`: share of the lake's area that is "dark water" (weak radar return), from 0 to 1

The village tanks have no best-quality readings. That is why the proposal tests whether "suspect" readings can be trusted.

## 4. Show that each ID is the right tank

- Hurulu Wewa, Nuwara Wewa and Mahakanadarawa: NASA's own `lake_name` gives the name.
- Nachchaduwa and Manankattiya: NASA gives no name. The Irrigation Department's official reservoir map lists both: [open the two records](https://services3.arcgis.com/J7ZFXmR8rSmQ3FGf/arcgis/rest/services/Scheme_Data_2/FeatureServer/10/query?where=Res_Name+IN+%28%27Nachchaduwa%27%2C%27Manankattiya%27%29&outFields=Res_Name,Type,Rivr_Basin,Cap_FSL_MCM,Area_FSL_ha&returnGeometry=false&returnCentroid=true&outSR=4326&f=html). The map's custodian is the ICT/GIS Branch of the Irrigation Department of Sri Lanka.
  - Nachchaduwa: the department's centre point is 8.2481, 80.4902. NASA's centre point is 8.2486, 80.4872, about 0.3 km away.
  - Manankattiya: the department's centre point is 8.2072, 80.6650. NASA's centre point is 8.2103, 80.6641, about 0.4 km away.
- Any tank: paste NASA's centre point (`p_lat, p_lon`) into Google Maps, for example `8.208651, 80.72828` for Hurulu Wewa.
- The village tanks have no names yet. The team still has to find them (the cascade and name check).

## 5. Count the readings in Excel (no code)

1. Excel: **Data** > **From Web**. Paste a tank's link and click OK. If Excel asks how to connect, choose **Anonymous** > **Connect**.
2. In the Power Query window, click **Record** next to *results*, then **Record** next to *geojson*, then **List** next to *features*.
3. Click **To Table**, then OK.
4. Click the two-arrow button on the *Column1* header. Tick only **properties**, untick "Use original column name as prefix", and click OK. Do the same on the new *properties* column (keep all boxes ticked, untick the prefix box).
5. Filter *time_str*: untick **no_data**.
6. Select *quality_f*, *xovr_cal_q* and *dark_frac* (Ctrl+click the headers). Then **Transform** > **Data Type** > **Decimal Number**.
7. **Home** > **Close & Load**.
8. In the sheet, filter *quality_f* to 0, *xovr_cal_q* to 0 and 1, and *dark_frac* with **Number Filters** > **Less Than Or Equal To** > 0.1.
9. The status bar shows the count, for example "71 of 101 records found" for Hurulu Wewa. For "usable", also tick 1 in *quality_f*.

## 6. The raw NASA files (optional)

Hydrocron is a quick way to read NASA's lake files. The files themselves are in NASA Earthdata Search (free Earthdata login): https://search.earthdata.nasa.gov/search?q=SWOT_L2_HR_LakeSP_D. Each "Prior" file covers one pass (Sri Lanka: passes 273 and 092) and opens in QGIS. Each lake row has the same lake ID and flags.

## 7. How to cite it

- Data: SWOT (2025). *SWOT Level 2 Lake Single-Pass Vector Data Product, Version D.* Jet Propulsion Laboratory, NASA PO.DAAC. https://doi.org/10.5067/SWOT-LAKESP-D. Accessed through the Hydrocron API on 25 October 2026.
- Rule: Köhn, S. J., Nielsen, K. (2026). Evaluation of SWOT HR PIXC version D water level time series of small lakes. *Hydrology and Earth System Sciences*, 30(20), 6443–6458. https://doi.org/10.5194/hess-30-6443-2026

The feasibility scripts used the same service and the same rule (`work\a2_fetch.py`, `work\a2_summarize.py`).
