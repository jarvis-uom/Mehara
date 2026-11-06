r"""All 117 reservoirs for Google Earth (3 Nov 2026): a KML file to check the chips on Google's own imagery.

For each reservoir:
- the department's point (red pin), with its capacity, outline area and checks;
- the chip's square (yellow line): the area cut out of every satellite image of that reservoir;
- the outline used to measure water (blue): the JRC water body the chip is centred on, widened by 30 m, as the fixed
  outline in temp/module-a-retrain/kernel/retrain.py.
Open it in Google Earth: on earth.google.com, Projects, New project, Import KML file from computer; or in Google Earth
Pro, File, Open. Writes out/reservoirs_117.kml.
Run from the project folder:
    temp\feasibility-check\.venv\Scripts\python.exe -I temp\module-a-reservoirs\scripts\make_kml.py
"""
import csv
import sys
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import rasterio
from rasterio import features
from rasterio.warp import transform, transform_geom
from shapely.geometry import mapping, shape
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO / "temp/module-a-tanks/scripts"))
sys.path.insert(0, str(REPO / "temp/module-a-retrain/kernel"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import reservoir_stats as S  # noqa: E402  (sheet, SHEET)
import retrain as K  # noqa: E402  (tank_mask: the fixed outline; safe)
import run_model as M  # noqa: E402  (load)

sys.stdout.reconfigure(encoding="utf-8")
STYLES = """<Style id="chip"><LineStyle><color>ff00ffff</color><width>2</width></LineStyle><PolyStyle><fill>0</fill></PolyStyle></Style>
<Style id="outline"><LineStyle><color>ffff7f00</color><width>2</width></LineStyle><PolyStyle><color>40ff7f00</color></PolyStyle></Style>
<Style id="point"><IconStyle><color>ff0000ff</color><Icon><href>http://maps.google.com/mapfiles/kml/paddle/red-circle.png</href></Icon></IconStyle></Style>"""


def ring(coords):
    return " ".join(f"{x:.6f},{y:.6f},0" for x, y in coords)


def polygon(geom):
    """A GeoJSON Polygon or MultiPolygon (longitude, latitude) as KML."""
    polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
    parts = []
    for p in polys:
        inner = "".join(f"<innerBoundaryIs><LinearRing><coordinates>{ring(r)}</coordinates></LinearRing>"
                        f"</innerBoundaryIs>" for r in p[1:])
        parts.append(f"<Polygon><tessellate>1</tessellate><outerBoundaryIs><LinearRing><coordinates>{ring(p[0])}"
                     f"</coordinates></LinearRing></outerBoundaryIs>{inner}</Polygon>")
    return parts[0] if len(parts) == 1 else "<MultiGeometry>" + "".join(parts) + "</MultiGeometry>"


def chip_shapes(path):
    """The outline (simplified to 10 m) and the chip's square, both in longitude and latitude."""
    _, occ = M.load(path)
    with rasterio.open(path) as src:
        crs, tr, w, h = src.crs, src.transform, src.width, src.height
    mask = K.tank_mask(occ)
    geoms = [shape(g) for g, v in features.shapes(mask.astype(np.uint8), mask=mask, transform=tr) if v == 1]
    outline = transform_geom(crs, "EPSG:4326", mapping(unary_union(geoms).simplify(10)))
    corners = [tr * c for c in ((0, 0), (w, 0), (w, h), (0, h), (0, 0))]
    xs, ys = transform(crs, "EPSG:4326", [c[0] for c in corners], [c[1] for c in corners])
    square = {"type": "Polygon", "coordinates": [list(zip(xs, ys))]}
    return outline, square, round(float(mask.sum()) * 0.01, 1), f"{w * 0.01:.1f} x {h * 0.01:.1f} km"


def main():
    info, _ = S.sheet()
    pts = {}
    with open(S.SHEET, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["reservoir"] and r["reservoir"] not in pts:
                pts[r["reservoir"]] = (float(r["lat"]), float(r["lon"]))
    with open(ROOT / "passes/out/pass_flags.csv", encoding="utf-8") as f:
        flags = {r["reservoir"]: r for r in csv.DictReader(f)}
    with open(ROOT / "passes/out/pass_scores.csv", encoding="utf-8") as f:
        rho = {r["reservoir"]: r["spearman_all"] for r in csv.DictReader(f) if r["method"] == "radar U-Net r2"}
    folders = []
    for name in sorted(info):
        with open(ROOT / "manifests" / f"{K.safe(name)}.csv", encoding="utf-8") as f:
            first = next(csv.DictReader(f))["file"]
        outline, square, ha, size = chip_shapes(ROOT / "chips" / first)
        lat, lon = pts[name]
        fl = flags.get(name, {})
        used = "used" if fl.get("used") == "True" else f"left out: {fl.get('why_left_out', 'not in the check')}"
        desc = (f"Capacity {info[name]['cap']:.2f} million m3 ({info[name]['kind']}). Chip {size}. Outline {ha} ha. "
                f"Every-pass check: {used}" + (f"; radar Spearman with storage {float(rho[name]):.2f}" if rho.get(name)
                                               else "") + ".")
        folders.append(
            f"<Folder><name>{escape(name)}</name>"
            f"<Placemark><name>{escape(name)}</name><styleUrl>#point</styleUrl>"
            f"<description>{escape(desc)}</description><Point><coordinates>{lon:.6f},{lat:.6f},0</coordinates></Point>"
            f"</Placemark>"
            f"<Placemark><name>{escape(name)}: outline</name><styleUrl>#outline</styleUrl>{polygon(outline)}</Placemark>"
            f"<Placemark><name>{escape(name)}: chip</name><styleUrl>#chip</styleUrl>{polygon(square)}</Placemark>"
            f"</Folder>")
    kml = (f'<?xml version="1.0" encoding="UTF-8"?>\n<kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
           f"<name>IS4990 Module A: 117 reservoirs</name><description>Red pin: the Irrigation Department's point. "
           f"Yellow: the chip (the area cut from every satellite image). Blue: the outline used to measure water "
           f"(JRC water body, widened by 30 m).</description>{STYLES}{''.join(folders)}</Document></kml>\n")
    out = ROOT / "out/reservoirs_117.kml"
    out.write_text(kml, encoding="utf-8")
    print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB, {len(folders)} reservoirs)")


if __name__ == "__main__":
    main()
