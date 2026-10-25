"""Satellite pictures for the deck (Sentinel-2 from Earth Search, Sentinel-1 RTC from Microsoft
Planetary Computer; both free, no login).

hero.jpg / hero_title.jpg   the tank cascade around Mahakanadarawa, clear day after the rains
problem.jpg                 the study box with every mapped tank (OpenStreetMap) in blue
hurulu_low/mid/full.jpg     Hurulu Wewa low, half-full and full, beside SWOT's height that week
hurulu_cloud.jpg            the same tank under monsoon cloud (Sentinel-2)
hurulu_radar.jpg            Sentinel-1 radar a few days from the cloudy picture
hurulu_water.png            a simple water map (NDWI > 0) of the full tank
Writes data/images_meta.json with the dates, SWOT heights and water areas used.
"""
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import planetary_computer as pc
import rasterio
from PIL import Image, ImageDraw, ImageEnhance
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds
from shapely.geometry import Point

ROOT = Path(__file__).resolve().parents[1]
FEAS = ROOT.parent / "feasibility-check"
sys.path.insert(0, str(FEAS / "work"))
import a2_summarize as a2  # noqa: E402

IMG, DATA = ROOT / "img", ROOT / "data"
UTM = "EPSG:32644"
CLOUD = [3, 8, 9, 10]
GDAL_ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff",
                GDAL_HTTP_MULTIRANGE="YES", GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES", VSI_CACHE="TRUE",
                GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="2", AWS_NO_SIGN_REQUEST="YES")
EARTH_SEARCH = "https://earth-search.aws.element84.com/v1"
PLANETARY = "https://planetarycomputer.microsoft.com/api/stac/v1"
HERO = (80.44, 8.30, 80.69, 8.44)
PROBLEM = (80.40, 8.16, 80.80, 8.60)
HURULU = (80.690, 8.168, 80.768, 8.250)
HURULU_ID = "4580013372"
HURULU_SCENES = {"low": "S2A_44PMQ_20230729_0_L2A", "mid": "S2A_44PMQ_20250730_1_L2A",
                 "full": "S2B_44PMQ_20260228_0_L2A", "cloud": "S2B_44PMQ_20251130_0_L2A"}
NAVY = (11, 34, 48)


def read(href, bbox_ll, scale=1.0, resampling=Resampling.bilinear):
    """Window of a COG for a lon/lat box; scale < 1 reads a coarser overview."""
    with rasterio.Env(**GDAL_ENV), rasterio.open(href) as src:
        b = transform_bounds("EPSG:4326", src.crs, *bbox_ll)
        win = from_bounds(*b, transform=src.transform).round_offsets().round_lengths()
        shape = (src.count, max(1, int(win.height * scale)), max(1, int(win.width * scale)))
        arr = src.read(window=win, out_shape=shape, resampling=resampling, boundless=True, fill_value=0)
        tr = src.window_transform(win) * src.window_transform(win).scale(win.width / shape[2], win.height / shape[1])
        return arr, tr, src.crs


def s2_items(bbox_ll, start, end, max_cloud=60):
    cat = Client.open(EARTH_SEARCH)
    items = cat.search(collections=["sentinel-2-l2a"], bbox=bbox_ll, datetime=f"{start}/{end}",
                       query={"eo:cloud_cover": {"lt": max_cloud}}).item_collection()
    return sorted(items, key=lambda it: it.properties["eo:cloud_cover"])


def cloud_share(item, bbox_ll):
    scl, _, _ = read(item.assets["scl"].href, bbox_ll, scale=0.5, resampling=Resampling.nearest)
    scl = scl[0]
    valid = scl > 0
    if valid.mean() < 0.98:
        return 1.0
    return float(np.isin(scl[valid], CLOUD).mean())


def clearest(bbox_ll, start, end, limit=8):
    best = None
    for it in s2_items(bbox_ll, start, end)[:limit]:
        c = cloud_share(it, bbox_ll)
        print(f"   {it.id}: {c:.1%} cloud in box")
        if best is None or c < best[1]:
            best = (it, c)
        if c < 0.005:
            break
    return best


def cloudiest(bbox_ll, start, end, limit=10):
    worst = None
    for it in s2_items(bbox_ll, start, end, max_cloud=100)[::-1][:limit]:
        c = cloud_share(it, bbox_ll)
        if 0.55 < c < 0.95 and (worst is None or c > worst[1]):  # heavy cloud, but some ground showing
            worst = (it, c)
    return worst


def true_colour(item, bbox_ll, scale=1.0, stretch=None, colour=1.12, contrast=1.0):
    """8-bit RGB from the visual (TCI) asset with a joint percentile stretch and a little colour."""
    rgb, tr, crs = read(item.assets["visual"].href, bbox_ll, scale)
    rgb = np.moveaxis(rgb[:3], 0, -1).astype("float32")
    lo, hi = stretch or np.percentile(rgb[rgb.sum(axis=2) > 0], [0.5, 99.6])
    out = np.clip((rgb - lo) / (hi - lo), 0, 1) ** 0.92
    im = Image.fromarray((out * 255).astype("uint8"))
    im = ImageEnhance.Contrast(ImageEnhance.Color(im).enhance(colour)).enhance(contrast)
    return im, tr, crs, (lo, hi)


def fade_left(im, start=0.0, end=0.62, strength=0.88):
    """Dark navy fade from the left edge for title text."""
    w, h = im.size
    ramp = np.clip((end - np.linspace(0, 1, w)) / (end - start), 0, 1) ** 1.3 * strength
    alpha = Image.fromarray((np.tile(ramp, (h, 1)) * 255).astype("uint8"))
    return Image.composite(Image.new("RGB", im.size, NAVY), im, alpha)


def hurulu_outline():
    osm = gpd.read_file(FEAS / "work" / "a1_osm_water.gpkg")
    p = gpd.GeoSeries([Point(80.72828, 8.208651)], crs="EPSG:4326").to_crs(osm.crs).iloc[0]
    inside = osm[osm.contains(p)].sort_values("area_ha")
    return inside.iloc[-1].geometry, osm.crs


def water_area(item, outline, outline_crs, buffer_m=250):
    g, tr, crs = read(item.assets["green"].href, HURULU)
    n, _, _ = read(item.assets["nir"].href, HURULU)
    g, n = g[0].astype("float32"), n[0].astype("float32")
    ndwi = (g - n) / np.maximum(g + n, 1)
    shape = gpd.GeoSeries([outline.buffer(buffer_m)], crs=outline_crs).to_crs(crs).iloc[0]
    inside = ~geometry_mask([shape], out_shape=ndwi.shape, transform=tr)
    water = (ndwi > 0) & inside
    return water.sum() * abs(tr.a * tr.e) / 1e6, ndwi, inside


def swot_near(day, days=10):
    d = a2.load(HURULU_ID)
    d = d[d.good]
    d["gap"] = (d.time.dt.tz_localize(None) - pd.Timestamp(day)).abs().dt.days
    r = d.sort_values("gap").iloc[0]
    return (r.time.date().isoformat(), round(float(r.wse), 2)) if r.gap <= days else (None, None)


def radar(bbox_ll, start, end, target=None):
    cat = Client.open(PLANETARY, modifier=pc.sign_inplace)
    items = list(cat.search(collections=["sentinel-1-rtc"], bbox=bbox_ll, datetime=f"{start}/{end}").items())
    print(f"   Sentinel-1 RTC scenes: {[it.datetime.date().isoformat() for it in items]}")
    if not items:
        return None, None
    if target is not None:
        items.sort(key=lambda i: abs((i.datetime.date() - target).days))
    day = items[0].datetime.date()
    vv = None
    for it in [i for i in items if i.datetime.date() == day]:  # the same day's frames fill each other's edges
        part = read(it.assets["vv"].href, bbox_ll)[0][0].astype("float32")
        if vv is not None and part.shape != vv.shape:
            part = np.array(Image.fromarray(part).resize(vv.shape[::-1]))
        vv = part if vv is None else np.where(vv > 0, vv, part)
    it = items[0]
    vv = vv[None]
    db = 10 * np.log10(np.clip(vv[0].astype("float32"), 1e-5, None))
    img = np.clip((db + 24) / 22, 0, 1) ** 1.05
    return Image.fromarray((img * 255).astype("uint8")).convert("RGB"), it.datetime.date().isoformat()


def by_id(item_id):
    return next(Client.open(EARTH_SEARCH).search(collections=["sentinel-2-l2a"], ids=[item_id]).items())


def hurulu(meta):
    """Hurulu Wewa low, half-full and full (clear days picked from all 2024-2026 scenes), then the
    same tank under monsoon cloud and Sentinel-1 radar from the same week."""
    outline, ocrs = hurulu_outline()
    full = by_id(HURULU_SCENES["full"])
    im_full, _, _, st = true_colour(full, HURULU)
    for key in ("low", "mid", "full"):
        it = full if key == "full" else by_id(HURULU_SCENES[key])
        im, _, _, _ = true_colour(it, HURULU, stretch=st)
        im.save(IMG / f"hurulu_{key}.jpg", quality=92)
        km2, ndwi, inside = water_area(it, outline, ocrs)
        day = it.datetime.date()
        swot_day, wse = swot_near(day, days=10)
        meta[f"hurulu_{key}"] = {"item": it.id, "date": day.isoformat(), "water_km2": round(float(km2), 2),
                                 "swot_date": swot_day, "swot_wse": wse}
        if key == "full":
            water = (ndwi > 0) & inside
            rgb = np.zeros(water.shape + (3,), "uint8")
            rgb[:] = (232, 237, 240)
            rgb[water] = (42, 120, 214)
            Image.fromarray(rgb).save(IMG / "hurulu_water.png")
    cl = by_id(HURULU_SCENES["cloud"])
    im_cl, _, _, _ = true_colour(cl, HURULU, stretch=st)
    im_cl.save(IMG / "hurulu_cloud.jpg", quality=92)
    day = cl.datetime.date()
    meta["hurulu_cloud"] = {"item": cl.id, "date": day.isoformat(), "cloud": round(cloud_share(cl, HURULU), 3)}
    rad, rday = radar(HURULU, (day - pd.Timedelta(days=6)).isoformat(), (day + pd.Timedelta(days=6)).isoformat(), target=day)
    if rad is not None:
        rad.resize(im_cl.size).save(IMG / "hurulu_radar.jpg", quality=92)
        meta["hurulu_radar"] = {"date": rday}


def main():
    IMG.mkdir(exist_ok=True)
    meta_path = DATA / "images_meta.json"
    if sys.argv[1:] == ["hurulu"]:
        meta = json.loads(meta_path.read_text())
        meta = {k: v for k, v in meta.items() if not k.startswith("hurulu")}
        hurulu(meta)
        meta_path.write_text(json.dumps(meta, indent=1))
        print(json.dumps(meta, indent=1))
        return
    meta = {}

    print("hero: clear day, Jan-Mar 2026")
    item, c = clearest(HERO, "2026-01-01", "2026-03-31")
    hero, _, _, _ = true_colour(item, HERO, colour=1.25, contrast=1.15)
    hero.save(IMG / "hero.jpg", quality=90)
    fade_left(hero).save(IMG / "hero_title.jpg", quality=90)
    meta["hero"] = {"item": item.id, "date": item.datetime.date().isoformat(), "cloud": round(c, 4)}

    print("problem: study box with mapped tanks")
    item_p, c = clearest(PROBLEM, "2026-01-01", "2026-03-31")
    im, tr, crs, _ = true_colour(item_p, PROBLEM, scale=0.5)
    base = ImageEnhance.Brightness(ImageEnhance.Color(im).enhance(0.35)).enhance(0.62)
    osm = gpd.read_file(FEAS / "work" / "a1_osm_water.gpkg").to_crs(crs)
    tanks = osm[osm.area_ha >= 6.25]
    mask = ~geometry_mask(list(tanks.geometry), out_shape=(im.height, im.width), transform=tr)
    arr = np.array(base)
    arr[mask] = (0.25 * arr[mask] + 0.75 * np.array([90, 166, 240])).astype("uint8")
    Image.fromarray(arr).save(IMG / "problem.jpg", quality=90)
    in_box = tanks.to_crs("EPSG:4326").cx[PROBLEM[0]:PROBLEM[2], PROBLEM[1]:PROBLEM[3]]
    meta["problem"] = {"item": item_p.id, "date": item_p.datetime.date().isoformat(), "tanks_6ha": int(len(in_box))}

    hurulu(meta)
    (DATA / "images_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
