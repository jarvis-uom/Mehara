# SWOT feasibility check, Anuradhapura (25 Oct 2026)

Live checks of public data, no login. Real SWOT heights per tank still need a login (see the end).

## 1. How often SWOT sees the area
- Source: NASA CMR search of SWOT_L2_HR_PIXC_D (version D pixel cloud), July 2023 to Oct 2026.
- Four passes cross Anuradhapura District: 092 (day 4 of each 21-day cycle), 273 (day 10), 398 (day 15) and 551 (day 20). Nadir tracks from the AVISO science-orbit KML.
- Data exist in 50 to 54 of 56 cycles for each pass (about 90 to 96%).
- Around Anuradhapura city (80.30-80.50E, 8.20-8.40N) there were 215 tiles on 156 dates: usually 3 looks per cycle for the area as a whole.

## 2. Which tanks each pass covers (radar sees 10-60 km from the ground track; 0-10 km is blind)
- 95 Irrigation Department tanks in the Anuradhapura range (ArcGIS Scheme_Data_2 layer):
  - 72 seen by 2 passes per 21 days (mostly 273 + 092; looks 6 and 15 days apart)
  - 13 seen by 1 pass
  - 10 not seen: they sit in the blind strip of pass 273 (ground track near 80.30E at 8.33N), e.g. Tissa Wewa, Abhaya Wewa, Galkadawala, Maha Bulankulama
- Of the 14 reservoirs the department reports daily in Anuradhapura, 12 are seen (8 by 2 passes, 4 by 1). Not seen: Tissa Wewa, Abhaya Wewa.
- Table: swot_coverage_anuradhapura_tanks.csv (distance in km from each pass's ground track).

## 3. Are the tanks big enough
- OpenStreetMap still-water polygons, 7.85-8.85N, 79.95-81.05E: 4,984 mapped.
  - 2,839 are 1 ha or more (the SWOT lake list's minimum); median 7.8 ha
  - 1,588 are 6.25 ha or more (SWOT's accuracy target size); 977 of these get 2 looks per 21 days
- Irrigation Department layer (354 tanks nationally): median full-supply area 63 ha; 46 are under 6.25 ha.
- Areas are mapped extents. Tanks shrink a lot in the dry season, so many 6-12 ha tanks fall below 6.25 ha for part of the year.

## 4. Ground truth
- The Irrigation Department's public sheet gives daily depth, storage and % full for 10 major and 4 medium Anuradhapura reservoirs (checked 24 Oct 2026). It shows today only, so save it daily (check_access.py does) or ask for the history.

## 5. Verdict
- Coverage is not the problem: most tanks are seen about every 10 days on average (worst gap about 15 days).
- The cascade must avoid the pass-273 blind strip (about 80.21-80.39E at Anuradhapura's latitude; it shifts east going north).
- Large and medium tanks: workable. Small village tanks (under about 10 ha, about 2 m deep): SWOT heights will be few and noisy; use Sentinel area plus curves for them and SWOT on the bigger tanks.
- Still unknown: whether the chosen tanks are in the SWOT lake list, and how clean their real SWOT heights are.

## 6. What needs a login
- Lake IDs come from the SWOT Prior Lake Database or a LakeSP_Prior file (pass 273 Asia file is about 20 MB). Both need a free NASA Earthdata or Hydroweb.next account.
- With lake IDs, Hydrocron (no login) returns each tank's full height and area series since July 2023.
