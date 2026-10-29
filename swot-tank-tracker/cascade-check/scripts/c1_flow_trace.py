"""Do the eight village tanks share one cascade? Follow the water downhill from each tank.

Elevation: Copernicus GLO-30 (already downloaded). Each cell drains to the cell that the
priority flood reached it from, so water crosses tank bunds and hollows at their spill point,
as in a real cascade. Water bodies: OpenStreetMap (25 Oct 2026) and the Irrigation Department's
reservoir layer. Also checks each tank against the department's reservoir catchments.

Run: python -I c1_flow_trace.py <cascade-check folder> <13 study tanks geojson> <out folder>
"""
import heapq
import json
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import from_bounds
from shapely.geometry import LineString, Polygon
from shapely.ops import linemerge, polygonize, unary_union

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
base, tanks_path, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
out.mkdir(exist_ok=True)
W, S, E, N = 80.38, 8.18, 80.72, 8.52          # 13 tanks plus their downstream reservoirs

# --- elevation window
with rasterio.open(base / "terrain/dem_download/Copernicus_DSM_COG_10_N08_00_E080_00_DEM.tif") as src:
    win = from_bounds(W, S, E, N, src.transform)
    dem = src.read(1, window=win).astype("float64")
    tr = src.window_transform(win)
rows, cols = dem.shape
print(f"elevation window {rows} x {cols} cells")

# --- priority flood: rec[i] = the cell that water from cell i flows into (-1 = leaves the window)
n = rows * cols
z = dem.ravel()
rec = np.full(n, -2, dtype=np.int64)
heap = []
for r in range(rows):
    for c in (0, cols - 1):
        i = r * cols + c
        rec[i] = -1
        heap.append((z[i], i))
for c in range(1, cols - 1):
    for r in (0, rows - 1):
        i = r * cols + c
        rec[i] = -1
        heap.append((z[i], i))
heapq.heapify(heap)
filled = z.copy()
NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
while heap:
    h, i = heapq.heappop(heap)
    r, c = divmod(i, cols)
    for dr, dc in NB:
        rr, cc = r + dr, c + dc
        if 0 <= rr < rows and 0 <= cc < cols:
            j = rr * cols + cc
            if rec[j] == -2:
                rec[j] = i
                filled[j] = max(z[j], h)
                heapq.heappush(heap, (filled[j], j))
print("flow directions done")

# --- water bodies: OpenStreetMap areas
osm = json.loads((base / "osm/download/overpass_water_areas.json").read_text(encoding="utf-8"))
polys = []
for el in osm["elements"]:
    tags = el.get("tags", {})
    try:
        if el["type"] == "way" and len(el.get("geometry", [])) >= 4:
            g = Polygon([(p["lon"], p["lat"]) for p in el["geometry"]])
        elif el["type"] == "relation":
            lines = [LineString([(p["lon"], p["lat"]) for p in m["geometry"]]) for m in el.get("members", [])
                     if m.get("role") == "outer" and len(m.get("geometry", [])) >= 2]
            g = unary_union(list(polygonize(linemerge(lines)))) if lines else None
        else:
            g = None
    except Exception:
        g = None
    if g is not None and g.is_valid and not g.is_empty:
        polys.append({"src": f"osm {el['type']} {el['id']}", "name": tags.get("name", ""), "geometry": g})
water = gpd.GeoDataFrame(polys, crs="EPSG:4326")
water = water[water.intersects(Polygon([(W, S), (E, S), (E, N), (W, N)]))]
water["ha"] = water.to_crs("EPSG:32644").area / 1e4

# department reservoirs (names and type), and our 13 study tanks
dept = gpd.read_file(base / "web-sources/irrigation_layers/reservoirs.geojson")
dept = dept[["Res_Name", "Type", "Rivr_Basin", "geometry"]]
tanks = gpd.read_file(tanks_path)
vil = tanks[tanks["kind"] != "big"].copy()
pts = {str(r.lake_id): r.key for r in
       gpd.pd.read_csv(base / "web-sources/analysis/points_used.csv", dtype={"lake_id": str}).itertuples()}
vil["key"] = vil["lake_id"].map(pts)
assert vil["key"].notna().all() and len(vil) == 8, vil[["lake_id", "key"]]

# one id raster: department reservoirs > our tanks > other OSM water
labels = []
shapes = []
for r in water.itertuples():
    labels.append((r.name or "unnamed", r.src, round(r.ha, 1), "osm"))
    shapes.append((r.geometry, len(labels)))
for r in tanks.itertuples():
    key = pts.get(str(r.lake_id), r.name)
    labels.append((key if r.kind != "big" else r.name, f"study tank {r.lake_id}", round(r.outline_ha, 1), "study"))
    shapes.append((r.geometry, len(labels)))
for r in dept.itertuples():
    labels.append((r.Res_Name, f"department {r.Type}", None, "dept"))
    shapes.append((r.geometry, len(labels)))
ids = rasterize(shapes, out_shape=dem.shape, transform=tr, fill=0, dtype="int32").ravel()


def cell(lon, lat):
    c, r = ~tr * (lon, lat)
    return int(r) * cols + int(c)


def trace(start, own_label):
    """Water bodies passed on the way down, until the window edge."""
    seq, i, steps = [], start, 0
    while i >= 0 and steps < 200000:
        k = ids[i]
        if k and labels[k - 1][0] != own_label and (not seq or seq[-1] != k):
            seq.append(int(k))
        i, steps = rec[i], steps + 1
    return seq, steps


catch = gpd.read_file(base / "web-sources/irrigation_layers/catchments.geojson")
report = []
for r in vil.sort_values("key").itertuples():
    p = r.geometry.representative_point()
    seq, steps = trace(cell(p.x, p.y), r.key)
    named = [labels[k - 1] for k in seq]
    first_dept = next((i for i, l in enumerate(named) if l[3] == "dept"), None)
    upto = named[: first_dept + 1] if first_dept is not None else named
    inside = catch[catch.contains(p)]["Reservoir_Name"].tolist()
    report.append({"tank": r.key, "lake_id": r.lake_id, "osm_name": r.osm_name or "", "outline_ha": r.outline_ha,
                   "path_km": round(steps * 30 / 1000, 1),
                   "drains_to": [f"{l[0]} ({l[1]}{', ' + str(l[2]) + ' ha' if l[2] else ''})" for l in upto],
                   "first_department_reservoir": named[first_dept][0] if first_dept is not None else None,
                   "village_tanks_passed": [l[0] for l in upto if l[3] == "study"],
                   "inside_department_catchments": inside})
    print(f"\n{r.key} ({r.lake_id}, {r.outline_ha} ha, {p.y:.4f} N {p.x:.4f} E)")
    print("  inside department catchment(s):", ", ".join(inside) or "none")
    for l in upto:
        print("   ->", l[0], "|", l[1], "|", f"{l[2]} ha" if l[2] else "")

(out / "flow_trace.json").write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
np.save(out / "receivers.npy", rec)
json.dump({"transform": list(tr)[:6], "shape": [rows, cols]}, open(out / "grid.json", "w"))
print("\nwrote", out / "flow_trace.json")
