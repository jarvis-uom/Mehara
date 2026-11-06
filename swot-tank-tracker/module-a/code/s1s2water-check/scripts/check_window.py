r"""Light check (3 Nov 2026): r6 and Otsu on the 1024 x 1024 window with the most expert water in one prepared scene,
to see that the Earth Engine radar and the registered mask line up before the upload. CPU.
    temp\feasibility-check\.venv\Scripts\python.exe -I temp\s1s2water-check\scripts\check_window.py 23
"""
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "kernel"))
import score_s1s2 as S  # noqa: E402

sid = sys.argv[1]
with rasterio.open(HERE / "gee" / f"{sid}_truth.tif") as src:
    tr = src.read(1)
H, W = tr.shape
best = max(((r, c) for r in range(0, H - 1024, 512) for c in range(0, W - 1024, 512)),
           key=lambda rc: int((tr[rc[0]:rc[0] + 1024, rc[1]:rc[1] + 1024] == 1).sum()))
w6 = HERE.parent / "module-a-tanks/weights/run-6"
dev = torch.device("cpu")
groups = [(S.R6, "", "water", [S.load(w6 / n, dev)[0] for n in ("unet_radar_best.pt", "unet_radar_seed1_best.pt",
                                                                  "unet_radar_seed2_best.pt")]),
          (S.NEW, 1, "not land", [S.load(w, dev)[0] for w in sorted((HERE.parent / "module-a-retrain/runs/"
                                  "2026-11-03-retrain-3/part-A/weights").glob("fold0_rule_seed*.pt"))])]
ck = torch.load(w6 / "unet_radar_best.pt", map_location="cpu", weights_only=False)
mean, std = np.array(ck["mean"], np.float32), np.array(ck["std"], np.float32)
res = S.score_scene(HERE / "gee" / f"{sid}_radar.tif", HERE / "gee" / f"{sid}_truth.tif", groups, mean, std, dev,
                    (best[0], best[1], 1024, 1024))
print("window", best, res["info"])
for k, a in res["acc"].items():
    print(k, S.scores(*a))
