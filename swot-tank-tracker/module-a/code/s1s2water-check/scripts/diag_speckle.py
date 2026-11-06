r"""Diagnostic (3 Nov 2026): r6 on Sen1Floods11's 15 Bolivia chips, as they are and after a 3 x 3 median, to see
whether the speckle check's large drop is real or a bug. CPU, light.
    temp\feasibility-check\.venv\Scripts\python.exe -I temp\s1s2water-check\scripts\diag_speckle.py
"""
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kernel"))
import score_s1s2 as S  # noqa: E402

root = Path("temp/module-a-kaggle/downloads/sen1floods11/v1.1")
w6 = Path("temp/module-a-tanks/weights/run-6")
dev = torch.device("cpu")
models = [S.load(w6 / n, dev)[0] for n in ("unet_radar_best.pt", "unet_radar_seed1_best.pt", "unet_radar_seed2_best.pt")]
ck = torch.load(w6 / "unet_radar_best.pt", map_location="cpu", weights_only=False)
mean, std = np.array(ck["mean"], np.float32), np.array(ck["std"], np.float32)
groups = [("r6", "", "water", models)]
ids = [ln.split(",")[0].strip().replace("_S1Hand.tif", "") for ln in
       open(root / "splits/flood_handlabeled/flood_bolivia_data.csv", encoding="utf-8") if ln.strip()]
hand = root / "data/flood_events/HandLabeled"
tot = {"as is": [0, 0, 0, 0], "median": [0, 0, 0, 0]}
for c in ids:
    with rasterio.open(hand / "S1Hand" / f"{c}_S1Hand.tif") as src:
        x = src.read([1, 2]).astype(np.float32)
    x[~np.isfinite(x)] = np.nan
    with rasterio.open(hand / "LabelHand" / f"{c}_LabelHand.tif") as src:
        y = src.read(1)
    ok, t = y >= 0, y == 1
    line = [c, f"truth water {t[ok].mean():.2f}"]
    for v, xv in (("as is", x), ("median", S.median3(x))):
        z = np.nan_to_num((xv - mean[:, None, None]) / std[:, None, None], nan=0.0)
        h, w = z.shape[1:]
        zp = np.pad(z, ((0, 0), (0, (-h) % 32), (0, (-w) % 32)), mode="reflect")
        m = S.water_maps(groups, torch.from_numpy(zp[None]))[("r6", "")][0, :h, :w].numpy()
        a = S.counts(m, t, ok)
        tot[v] = [p + q for p, q in zip(tot[v], a)]
        line.append(f"{v}: water {m[ok].mean():.2f} IoU {S.scores(*a)['iou']}; VV water median "
                    f"{np.nanmedian(xv[0][t & ok]) if (t & ok).any() else float('nan'):.1f} dB")
    print(" | ".join(line))
print({v: S.scores(*a) for v, a in tot.items()})
