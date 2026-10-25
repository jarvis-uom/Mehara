# Data access checks

This script repeats, from your own internet connection, the live checks in the doc
"Data and Access Check for the Three Projects". It also does the steps that our
sandbox could not: Overpass counts, the full Irrigation Department tank list and the
daily reservoir sheet.

## Run it

```
pip install requests
python check_access.py
```

Results land in `results/` next to the script:

| File | What it is |
| --- | --- |
| `summary.md` | All counts in one page |
| `swot_lakesp_files_*.txt` | SWOT lake files over Sri Lanka and the Anuradhapura cascade |
| `hydrocron_<lake_id>.csv` | SWOT height and area per tank, cleaned |
| `irrigation_tanks.csv` | Every tank in the Irrigation Department layer, with full-supply level and capacity |
| `reservoir_status_*_<date>.csv` | Today's major and medium reservoir status; run daily to build a history |
| `asf_*.csv` | Sentinel-1 and NISAR scenes over the study areas |

## Getting real Sri Lankan tank IDs for Hydrocron

1. Create a free account on Hydroweb.next (Theia) and download the SWOT Prior Lake
   Database tile that covers Sri Lanka, or log in to NASA Earthdata and download one
   `SWOT_L2_HR_LakeSP_Prior` file for pass 092, 273, 301 or 398.
2. Open it in QGIS, select the cascade's tanks, and copy their `lake_id` values.
3. Put one ID per line in `lake_ids.txt` and run the script again.

## Steps that need your own login (do these yourself in a terminal)

- NASA Earthdata: download SWOT pixel-cloud tiles (about 500 MB each), NISAR GCOV and HLS files.
- Google Earth Engine: Sentinel-1/2 time series, AlphaEarth embeddings, Dynamic World, the Özdoğan rice-map assets.
