r"""The "lake high enough" rule: labels without hand-labelling (2 Nov 2026; version 2 after the outside reviews).

A reservoir fills like a bowl. When its storage rises past a certain value, a spot on its bed goes under water, and
it stays under water at any higher storage. Each spot's value is learnt from the reservoir's clear photos and the
department's storage on their dates:
- evidence of water: the photo's two water tests agree (NDWI >= -0.06 and MNDWI >= 0) and the spot is dark in SWIR
  (B11 under 0.10). Thin cloud and haze can pass both tests, but they are bright in SWIR, and water is not;
- evidence of dry ground: both tests say "not water" and the spot is bare (NDVI under 0.2). Plants give no evidence
  either way, because plants can stand on water or on dry ground;
- the spot's cover value is the storage that best splits its water days from its dry days (fewest days on the wrong
  side; never between two days with the same storage). Equal splits give a range. Above the range the spot is under
  water; below it, dry. A spot needs at least 3 days of evidence with at most 1 in 5 on the wrong side.
For any date with storage on record, inside the reservoir's outline: storage above the range = water (open, or under
plants), storage below the range = land, in between = unsure. On a clear day the photo adds what it sees: open water
is open water (unsure if the rule says the spot is dry), bare ground is land (unsure if the rule says it is under
water), and plants are water under plants on a covered spot and land on a dry one.
Version 2 (3 Nov), after the outside reviews:
- more land to learn from: outside the outline, ground that JRC never saw as water and that lies over 30 m from any
  water it saw is land on every date (also on radar-only days); at low storage (15% of capacity or less on both
  dates), plants on the bed more than 30 m from the photo's open water are land (the lake is small then; floating
  mats sit next to open water);
- cloud shadow: outside the outline, the photo's open water counts only where JRC saw water in over 10% of
  observations (shadows are dark in SWIR and passed the tests);
- spots that may have changed between the photo and the radar dates (storage moved by over 0.5% of capacity) are
  unsure only where the spot has a cover value or bound (version 1 also dropped unknown spots, 7% of the photo's);
- Cloud Score+ 0 is cloud too (the export wrote 0 for both "no data" and the thickest cloud).
Labels: 0 land, 1 open water, 2 water under plants, 3 water of either kind (no usable photo), 255 unsure.
"""
import numpy as np
from scipy import ndimage

VERSION = "2.1 (3 Nov 2026)"
BANDS = ["VV", "VH", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]  # as run_model.BANDS
NDWI_MIN, MNDWI_MIN, BARE_MAX = -0.06, 0.0, 0.2
SWIR_MAX = 1000  # B11 reflectance x 10000: water is darker
MARGIN = 0.01  # share of capacity
CHANGED = 0.005  # storage change between the photo and the radar that can move the shoreline (share of capacity)
LOW = 0.15  # low storage, when plants on the bed away from the water are dry ground
NEAR_PX = 3  # 30 m
JRC_OUTSIDE_MIN = 10  # % of observations: photo water outside the outline needs this much JRC history
MIN_DAYS, MAX_WRONG = 3, 0.2
LAND, OPEN, PLANTS, ANY, UNSURE = 0, 1, 2, 3, 255
P_NONE, P_BARE, P_WATER, P_PLANTS = -1, 0, 1, 2  # what the photo shows


def photo_class(x, cs):
    """x: the chip's 12 bands (VV, VH, then Sentinel-2 as reflectance x 10000, NaN = no data); cs: Cloud Score+."""
    b3, b4, b8, b11 = (x[BANDS.index(b)] for b in ("B3", "B4", "B8", "B11"))
    with np.errstate(invalid="ignore", divide="ignore"):
        ndwi, mndwi, ndvi = (b3 - b8) / (b3 + b8), (b3 - b11) / (b3 + b11), (b8 - b4) / (b8 + b4)
        wet = (ndwi >= NDWI_MIN) & (mndwi >= MNDWI_MIN) & (b11 < SWIR_MAX)
        dry = (ndwi < NDWI_MIN) & (mndwi < MNDWI_MIN)
        out = np.full(ndwi.shape, P_NONE, np.int8)
        out[wet] = P_WATER
        out[dry & (ndvi < BARE_MAX)] = P_BARE
        out[dry & (ndvi >= BARE_MAX)] = P_PLANTS
    out[cs < 0.6] = P_NONE  # cloud or shadow (0 = the thickest cloud, or no data)
    return out


def context(occ, inside):
    """What the labels need about a reservoir's chip besides its cover values: the outline, JRC occurrence, and the
    ground outside the outline that JRC never saw as water, over 30 m from any water it saw."""
    seen = occ > 0
    far = ndimage.distance_transform_edt(~seen) > NEAR_PX if seen.any() else np.ones(occ.shape, bool)
    return {"inside": inside, "occ": occ, "land_out": ~inside & ~seen & far}


def fit_cover(photos, storage):
    """photos: (days, H, W) of photo_class; storage: (days,) as a share of capacity.
    Returns lo and hi (H, W): the spot is under water at storage above hi and dry below lo (-inf and inf if unknown),
    and the days of evidence for each spot."""
    order = np.argsort(storage, kind="stable")
    s, p = np.asarray(storage, float)[order], photos[order]
    zero = np.zeros((1,) + p.shape[1:], np.int32)
    wet = np.concatenate([zero, np.cumsum(p == P_WATER, axis=0, dtype=np.int32)])
    dry = np.concatenate([zero, np.cumsum(p == P_BARE, axis=0, dtype=np.int32)])
    wrong = wet + (dry[-1] - dry)  # split k: the k lowest days should be dry, the rest under water
    wrong[1:-1][s[1:] == s[:-1]] = np.iinfo(np.int32).max  # no split between two days with the same storage
    best = wrong.min(axis=0)
    tied = wrong == best
    k_lo, k_hi = tied.argmax(axis=0), len(s) - tied[::-1].argmax(axis=0)
    edge = np.concatenate([[-np.inf], s, [np.inf]])  # edge[k] = storage of the k-th lowest day
    lo, hi = edge[k_lo], edge[k_hi + 1]
    days = wet[-1] + dry[-1]
    sure = (days >= MIN_DAYS) & (best <= MAX_WRONG * days)
    lo[~sure], hi[~sure] = -np.inf, np.inf
    return lo, hi, days


def rule(lo, hi, inside, s_low, s_high=None):
    """1 under water, 0 dry, -1 unknown, for a date (or two dates) with storage s_low to s_high."""
    s_high = s_low if s_high is None else s_high
    r = np.full(lo.shape, -1, np.int8)
    r[inside & (s_low >= hi + MARGIN)] = 1
    r[inside & (s_high <= lo - MARGIN)] = 0
    return r


def clear_label(photo, lo, hi, ctx, s_photo, s_radar):
    """The label on a clear day (or a radar pass near one), for training the radar: the photo where it is sure, the
    rule for plants and gaps."""
    inside, occ, land_out = ctx["inside"], ctx["occ"], ctx["land_out"]
    s_min, s_max = min(s_photo, s_radar), max(s_photo, s_radar)
    r = rule(lo, hi, inside, s_min, s_max)
    water = photo == P_WATER
    y = np.full(photo.shape, UNSURE, np.uint8)
    y[water & (inside | (occ > JRC_OUTSIDE_MIN))] = OPEN
    y[water & (r == 0)] = UNSURE  # a rain pond, paddy or shadow on ground the rule says is dry
    y[(photo == P_BARE) & (r != 1)] = LAND
    y[(photo == P_PLANTS) & (r == 1)] = PLANTS
    y[(photo == P_PLANTS) & ((r == 0) | ~inside)] = LAND
    if s_max <= LOW:  # low water: plants on the bed away from the open water are dry ground (2.1: also when the
        # photo shows no open water at all, where 2 left them unsure; found by the 3 Nov review)
        far = ndimage.distance_transform_edt(~water) > NEAR_PX if water.any() else np.ones(photo.shape, bool)
        y[(photo == P_PLANTS) & inside & far & (r != 1)] = LAND
    y[(photo == P_NONE) & (r == 1)] = ANY
    y[(photo == P_NONE) & ((r == 0) | land_out)] = LAND
    if s_max - s_min > CHANGED:  # spots with a known cover range that may have changed between the two dates
        known = np.isfinite(lo) | np.isfinite(hi)
        y[inside & known & (lo - MARGIN < s_max) & (hi + MARGIN > s_min)] = UNSURE
    return y


def radar_label(lo, hi, ctx, s):
    """The label on a day without a usable photo: the rule inside the outline, and never-water ground outside it."""
    r = rule(lo, hi, ctx["inside"], s)
    y = np.full(r.shape, UNSURE, np.uint8)
    y[r == 1] = ANY
    y[(r == 0) | ctx["land_out"]] = LAND
    return y


def photo_label(photo, ctx):
    """The control's label: the photo alone, no storage (open water as in clear_label; bare ground and plants are
    land, as in the flood datasets)."""
    y = np.full(photo.shape, UNSURE, np.uint8)
    y[(photo == P_WATER) & (ctx["inside"] | (ctx["occ"] > JRC_OUTSIDE_MIN))] = OPEN
    y[(photo == P_BARE) | (photo == P_PLANTS)] = LAND
    return y
