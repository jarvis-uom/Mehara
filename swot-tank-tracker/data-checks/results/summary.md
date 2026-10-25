# Data access check, 25 Oct 2026

## SWOT Lake Single-Pass (CMR, no login)
- Sri Lanka: 1098 files; passes 023, 092, 273, 301, 398, 551, 579
  first SWOT_L2_HR_LakeSP_Obs_001_273_AS_20230730T231508_20230730T232608_PGD0_01
  last  SWOT_L2_HR_LakeSP_Unassigned_057_398_AS_20261015T201340_20261015T202633_PID0_01
- Anuradhapura: 735 files; passes 023, 092, 273, 301, 398
  first SWOT_L2_HR_LakeSP_Obs_001_273_AS_20230730T231508_20230730T232608_PGD0_01
  last  SWOT_L2_HR_LakeSP_Unassigned_057_398_AS_20261015T201340_20261015T202633_PID0_01

## Hydrocron (no login)
- lake_ids.txt is empty, so this run uses the documentation's example lake.
- 6350036102: 111 rows returned, 106 after removing duplicates and fill values

## Irrigation Department tank layer (ArcGIS, no login)
- 354 tanks; by type {'Medium': 166, 'Minor': 107, 'Major': 81}; 258 have a capacity value

## Irrigation Department reservoir status sheet (public Google Sheet)
- major: 74 rows saved to reservoir_status_major_20261025.csv; keep running daily to build a history
- medium: 44 rows saved to reservoir_status_medium_20261025.csv; keep running daily to build a history

## Sentinel-2 L2A scenes (Earth Search, no login)
| Site | Season | All scenes | Under 20% cloud |
| --- | --- | --- | --- |
| Gampaha | Yala 2025 | 180 | 10 |
| Gampaha | Maha 2025/26 | 170 | 47 |
| Gampaha | Yala 2026 | 184 | 4 |
| Kalutara | Yala 2025 | 94 | 6 |
| Kalutara | Maha 2025/26 | 85 | 27 |
| Kalutara | Yala 2026 | 90 | 3 |
| Anuradhapura | Yala 2025 | 45 | 4 |
| Anuradhapura | Maha 2025/26 | 43 | 4 |
| Anuradhapura | Yala 2026 | 46 | 5 |

## Sentinel-1 GRD over Gampaha (ASF, no login)
- 18 scenes since 1 Apr 2026 on 18 dates; gaps in days: [6, 7, 12, 24]

## NISAR products (ASF search, no login; downloads need Earthdata)
- Gampaha: 100 products; by type {'SME2': 19, 'GUNW': 18, 'GSLC': 21, 'GCOV': 21, 'RSLC': 21}
- Anuradhapura: 99 products; by type {'SME2': 21, 'GUNW': 15, 'GSLC': 21, 'GCOV': 21, 'RSLC': 21}

## OpenStreetMap counts for Moratuwa (Overpass, no login)
- power lines (ways): 0
- power poles: 0
- trees: 0
- farmland areas: 0
