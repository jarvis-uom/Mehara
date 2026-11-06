r"""Reads one S1S2-Water scene's radar image, mask and valid mask straight from the zip on Zenodo (3 Nov 2026).

GDAL reads only the parts it needs (/vsizip/ over /vsicurl/), so nothing big is downloaded.
    temp\feasibility-check\.venv\Scripts\python.exe -I temp\s1s2water-check\scripts\peek_scene.py 31 part2.zip
"""
import sys

import numpy as np
import rasterio
from rasterio.windows import Window

sid, part = sys.argv[1], sys.argv[2]
base = f"/vsizip/{{/vsicurl/https://zenodo.org/api/records/11278238/files/{part}/content}}/{sid}/sentinel12_s1_{sid}"
with rasterio.Env(GDAL_HTTP_MAX_RETRY=3, CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".zip,/content"):
    for kind in ("img", "msk", "valid"):
        with rasterio.open(f"{base}_{kind}.tif") as src:
            print(kind, src.count, src.dtypes, src.width, src.height, src.crs, src.res, "nodata", src.nodata,
                  "desc", src.descriptions, "scales", src.scales, "offsets", src.offsets)
            print("  tags", dict(list(src.tags().items())[:8]))
            w = Window(src.width // 2 - 512, src.height // 2 - 512, 1024, 1024)
            a = src.read(window=w).astype("float64")
            for b in range(a.shape[0]):
                v = a[b][np.isfinite(a[b])]
                print(f"  band {b + 1}: min {v.min():.3f} p5 {np.percentile(v, 5):.3f} median {np.median(v):.3f} "
                      f"p95 {np.percentile(v, 95):.3f} max {v.max():.3f}; values {np.unique(v)[:6] if kind != 'img' else ''}")
