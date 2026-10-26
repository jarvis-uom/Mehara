# More frequent data: what it costs (27 October 2026)

**Why we need more data:** to check each tank's height–area–volume curve (its "layers"), we need many pairs of water height and water area at different levels, or a direct map of the tank bed.
- **The 5 big reservoirs** already have free daily gauge heights from the Irrigation Department. Our logger repo saves them, so pairing them with every satellite water map costs nothing.
- **The 8 village tanks** have no gauges, so that is where money helps.

**Exchange rate:** 1 USD = Rs 332.17 and 1 GBP = Rs 444.20 ([ExchangeRate-API](https://open.er-api.com/v6/latest/USD), 26 Oct 2026). Prices were checked on 27 Oct 2026. "Checked twice" means the page was opened again for a second check.

## Free (Rs 0)

| Source | Gives | How often | Detail | Link |
|---|---|---|---|---|
| Planet Education & Research, Basic | Water area, 3 m | Near daily | Free for university staff and students: 3,000 km² a month, 30-day delay, university email, non-commercial licence (checked twice) | [planet.com](https://www.planet.com/markets/education-and-research/) |
| NISAR (L-band radar) | Water area through cloud | Every 12 days | Public since 20 Jul 2026, for data from 17 Jun 2026 (checked twice); 10–80 m pixels | [ASF notice](https://asf.alaska.edu/notices/nisar-l-band-data-now-publicly-available/) |
| NASA HLS (Landsat + Sentinel-2) | Water area, 30 m | Every 2–3 days | Same cloud problem as Sentinel-2; a 10 ha tank is only about 110 pixels | [NASA](https://science.nasa.gov/missions/landsat/data-in-harmony-nasas-harmonized-landsat-and-sentinel-2-project) |
| Landsat 8 + 9 | Water area, 30 m | Every 8 days together | | [NASA](https://science.nasa.gov/mission/landsat-9) |
| Irrigation Department gauges | Water height, 5 big reservoirs | Daily | Already saved by our water-data-logger repo | [Reservoir Status](https://irrigation.gov.lk/web/index.php?option=com_content&view=article&id=84&Itemid=201&lang=en) |

## Water height at the 8 village tanks

**Low-cost logger, parts bought in Sri Lanka**

| Part | Price | Shop |
|---|---|---|
| ESP32 board | Rs 1,570, in stock (checked twice) | [duino.lk](https://www.duino.lk/product/esp32-cp2102-type-c-development-board/) |
| Waterproof ultrasonic sensor (AJ-SR04T) | Rs 1,090, in stock | [duino.lk](https://www.duino.lk/product/waterproof-ultrasonic-sensor-module-aj-sr04t/) |
| 4G module (A7670C) | Rs 4,750, in stock | [alphatronic.lk](https://alphatronic.lk/product/sima7670c-fs-mcore-4g-lte-module/) |
| 20 W solar panel | Rs 4,690, in stock | [duino.lk](https://www.duino.lk/product/20w-17-87vdc-solar-panel-polycrystalline-solar-cell/) |
| Step-down board (MP1584) | Rs 170 | duino.lk |
| 18650 battery + charger board | Rs 890 + Rs 100 | [duino.lk](https://www.duino.lk/product/3500mah-3-7v-18650-button-top-samsung-battery/) |
| IP65 box | Rs 1,270, in stock | [duino.lk](https://www.duino.lk/product/160x110x90mm-abs-project-box/) |
| **Parts for one logger** | **Rs 14,530** (Rs 11,570 with a 2G SIM800L module, but Dialog keeps 2G only for voice, so 4G is safer) | |
| Mobile data, 30 days | Dialog Rs 265 (2.5 GB), Hutch Rs 285 (4 GB), SLT-Mobitel Rs 299 (3.5 GB); a logger uses only a few MB | [Dialog](https://www.dialog.lk/mobile-broadband/prepaid/plan/mobile-data-plans) |

The ultrasonic sensor needs a fixed arm above the water at every level. Each logger's zero must be matched to SWOT once.

**Ready-made loggers (reseller prices)**
- **HOBO U20L-04:** US$425 (≈ Rs 141,200) per logger. A full first site costs US$1,148: logger, a second logger for air pressure, the base station and the software. One air-pressure logger can serve several nearby tanks. ([Instrumart](https://www.instrumart.com/products/40734/hobo-u20l-series-water-level-loggers))
- **Solinst Levelogger 6 + Barologger 6:** US$1,237 (≈ Rs 410,900) per pair. ([PRM Filtration](https://shop.prmfiltration.com/products/solinst-model-3001-levelogger-6-5-meter-to-200-meter-length))
- **Enamel staff gauge plate:** £68.73 (≈ Rs 30,500) in the UK, plus shipping. No Sri Lankan price found. ([qualityenamelsigns.co.uk](https://www.qualityenamelsigns.co.uk/enamel-gauge-scale-custom-made))

## The tank bed: checks every layer directly

- **Deeper PRO+ 2 sonar:** US$159.99 on sale, regular US$179.99 (≈ Rs 53,100). It was sold out on the US store on 27 Oct (checked twice).
  - Its depth points download as a CSV file of latitude, longitude, depth and time ([Deeper support](https://support.deeper.eu/756931-Downloading-raw-bathymetric-data)). ([Deeper store](https://deepersonar.com/en-us/products/deeper-sonar-pro-2))
  - To cover a whole tank it needs a boat. Our safety rule says we never go into the water, so it would need a local fisherman's boat with life jackets. From the bank it maps only the edges.
- **Deeper CHIRP+ 3:** £289.99 (≈ Rs 128,800) in the UK. ([Deeper UK](https://deepersonar.com/products/deeper-chirp-3))
- **Drone for the dry bed in the dry season:**
  - DJI Mini 4 Pro Fly More Combo, LKR 309,500 in Sri Lanka ([LuxuryX](https://luxuryx.lk/dji-mini-4-pro-fly-more-combo)).
  - Flying it needs CAASL approval and Ministry of Defence clearance ([caa.lk](https://www.caa.lk/en/drones)), so borrow one rather than buy.

## Paid satellite images (not needed if Planet's free plan works)

| Product | Price | Source |
|---|---|---|
| Umbra radar, 1 m, 5 × 5 km scene | US$675 (≈ Rs 224,200) per image (checked twice) | [umbra.space](https://umbra.space/pricing) |
| Capella radar, archive Stripmap | From US$464 per order (indicative) | [SkyWatch](https://skywatch.com/geospatial-data-providers/capella-space/) |
| ICEYE radar, archive Strip | From US$690 per order (indicative) | [SkyWatch](https://skywatch.com/geospatial-data-providers/iceye/) |
| PlanetScope, bought | US$2.25 per km², minimum 250 km², so US$562.50 per order | [Apollo Mapping](https://apollomapping.com/planetscope-satellite-imagery) |
| Planet Research plan | US$15,000 a year | [planet.com](https://www.planet.com/markets/education-and-research/) |

The free Umbra and Capella archives have no images over our tanks.

## Packages

| Package | One-off cost | Each month |
|---|---|---|
| A. Free data only | Rs 0 | Rs 0 |
| B. Smallest: one Deeper sonar | about Rs 53,100, plus shipping and duty | Rs 0 |
| C. Pilot: loggers on the 3 plant-covered village tanks | about Rs 43,600 | about Rs 800 |
| D. Recommended: free data + 8 loggers + 1 Deeper sonar | about Rs 169,400 | about Rs 2,100 |

Not priced in any package: poles and mounts, SIM cards, shipping and import duty, boat hire, and surveying each sensor's zero.
