r"""Makes colab/s1s2water_colab.ipynb: the S1S2-Water check on Google Colab's free T4 GPU (3 Nov 2026), for when
Kaggle has no GPU hours left.

It is the same code as the Kaggle notebook (score_s1s2.py, shipped inside the private dataset is4990-s1s2water-gee),
with the same inputs: the prepared scenes, Sen1Floods11, and the weights of runs 2, 6 and 3 from their Kaggle notebook
outputs. The notebook reads the team's Kaggle API token from a Colab secret, so no token is ever written into the
notebook or the project. Results go to Google Drive as they are made, then to the private Kaggle dataset
is4990-s1s2water-check-results for the laptop.
Run from the project folder:
    temp\feasibility-check\.venv\Scripts\python.exe -I temp\s1s2water-check\scripts\make_s1s2_colab.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CELLS = [
    ("markdown", """# IS4990 Module A: the S1S2-Water check (PROTOCOL-S1S2)

Runs the existing water models (no training) on S1S2-Water's expert-labelled scenes, then the smoothing (speckle) check on Sen1Floods11. It is the same code as the Kaggle notebook (`score_s1s2.py`, from the private dataset).

**Before you start**
1. Runtime > Change runtime type > **T4 GPU**.
2. Secrets (the key icon on the left): **`KAGGLE_API_TOKEN`** must be there, with notebook access on (it is the one from run 3).
3. Runtime > **Run all**. Keep this tab open until the last cell prints `done` (about 1.5 hours).

Results are saved to your Google Drive (`MyDrive/is4990-s1s2water-check`) as each scene finishes, then go to the private Kaggle dataset `mehararothila/is4990-s1s2water-check-results`."""),
    ("code", """# 1. A GPU
import subprocess
import torch
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], capture_output=True, text=True).stdout)
assert torch.cuda.is_available(), "No GPU: Runtime > Change runtime type > T4 GPU" """),
    ("code", """# 2. Google Drive for the results, and the Kaggle login from the Colab secret
import os
from google.colab import drive, userdata
drive.mount("/content/drive")
OUT = "/content/drive/MyDrive/is4990-s1s2water-check"
os.makedirs(OUT, exist_ok=True)
token = userdata.get("KAGGLE_API_TOKEN").strip()
os.makedirs("/root/.kaggle", exist_ok=True)
with open("/root/.kaggle/access_token", "w") as f:
    f.write(token)
os.chmod("/root/.kaggle/access_token", 0o600)
os.environ["KAGGLE_API_TOKEN"] = token
!pip install -q "kaggle==2.2.5" rasterio "segmentation-models-pytorch==0.6.0"
!kaggle --version"""),
    ("code", """# 3. The inputs from Kaggle: the prepared scenes (about 7 GB), Sen1Floods11 and the weights (about 10 minutes)
IN = "/content/input"
for d in ["is4990-s1s2water-gee", "is4990-sen1floods11-radar"]:
    !kaggle datasets download -d mehararothila/{d} -p {IN}/{d} --unzip -q
for k in ["is4990-module-a-run-2", "is4990-module-a-run-6", "is4990-module-a-retrain-3a",
          "is4990-module-a-retrain-3b1", "is4990-module-a-retrain-3b2"]:
    !kaggle kernels output mehararothila/{k} -p {IN}/{k} --file-pattern ".*[.]pt$" --page-size 200 -q
!find {IN} -name "*.pt" | wc -l
!du -sh {IN}/*
!df -h /content | tail -n 1"""),
    ("code", """# 4. The check (about 1 to 1.5 hours). Each scene is saved to Drive as it finishes.
!cp {IN}/is4990-s1s2water-gee/score_s1s2.py /content/score_s1s2.py
!python -u /content/score_s1s2.py --input {IN} --out {OUT}"""),
    ("code", """# 5. The results to a private Kaggle dataset, for the laptop
import json
import shutil
UP = "/content/upload"
shutil.rmtree(UP, ignore_errors=True)
shutil.copytree(OUT, UP)
with open(f"{UP}/dataset-metadata.json", "w") as f:
    json.dump({"id": "mehararothila/is4990-s1s2water-check-results", "title": "IS4990 S1S2Water check results",
               "subtitle": "Module A on S1S2-Water's expert labels, from Colab (private)", "isPrivate": True,
               "licenses": [{"name": "other"}]}, f)
!kaggle datasets create -p {UP} --dir-mode zip
print("done")"""),
]


def main():
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                       "kernelspec": {"display_name": "Python 3", "name": "python3"},
                       "language_info": {"name": "python"}},
          "cells": []}
    for i, (kind, text) in enumerate(CELLS):
        lines = text.strip().split("\n")
        source = [ln + "\n" for ln in lines[:-1]] + [lines[-1]]
        cell = {"cell_type": kind, "id": f"cell{i}", "metadata": {}, "source": source}
        if kind == "code":
            cell.update({"execution_count": None, "outputs": []})
        nb["cells"].append(cell)
    out = ROOT / "colab/s1s2water_colab.ipynb"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
