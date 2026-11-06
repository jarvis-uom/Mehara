r"""Makes series 2's development Kaggle files (5 Nov 2026, ../PROTOCOL-6.md part 1 section B, ../DEV-6.md):
- dev6-code/: the private dataset is4990-module-a-dev6-code (retrain6.py, run6_dev.py, split_dates.csv and the
  S1S2-Water check's score_s1s2.py, unchanged). Git ignores this folder: its sources are kernel6-dev/ and
  ../s1s2water-check/kernel/;
- kaggle-notebooks/kernel6-dev-control/, -long/, -lovasz/: three private notebooks, one variant each (folds 1
  and 2 at once on two T4s), run by a teammate from their own Kaggle account (shared with them as a collaborator).
  They are pushed without a GPU, so the first version only prints how to run them.
Run from the project folder:
    temp\feasibility-check\.venv\Scripts\python.exe -I temp\module-a-retrain\scripts\make_dev6_kaggle.py
"""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OWNER = "mehararothila"
CODE = "is4990-module-a-dev6-code"
VARIANTS = {"control": ("run 5's recipe, the comparison for the others", "about 1 hour"),
            "long": ("3 times the training steps, scoring the weights' moving average", "about 2.5 to 3 hours"),
            "lovasz": ("as long, plus a Lovász (IoU) loss term", "about 2.5 to 3 hours")}
SOURCES = [f"{OWNER}/{d}" for d in ("is4990-module-a-run4-inputs", "is4990-s1s2water-train",
                                    "is4990-sen1floods11-radar", "is4990-module-a-dynamic-world",
                                    "is4990-module-a-first-check-chips", "is4990-module-a-pass-chips",
                                    "is4990-s1s2water-gee", CODE)]


def cells(variant, what, hours):
    return [
        ("markdown", f"""# IS4990 Module A: series 2 development, variant "{variant}"

Trains {what}, on folds 1 and 2, then scores them on development data only (PROTOCOL-6 part 1, section B; DEV-6).
Nothing in the code needs changing.

**How to run it**
1. On the right, under the session options: **Accelerator: GPU T4 x2** (or GPU P100 if T4 is not available), and
   **Internet: On**.
2. Top right: **Save Version**, choose **Save & Run All (Commit)**, then **Save**.
3. Stop this editing session (the power button at the top), so that it does not use GPU hours while the saved
   version runs.

The saved version then runs on its own, {hours}. The browser can be closed."""),
        ("code", """# 1. Is there a GPU? Without one, this version only prints how to run it.
import subprocess
import sys
from pathlib import Path
try:
    GPUS = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip()
except OSError:
    GPUS = ""
print(GPUS or "No GPU in this session, so nothing will run. To run: Accelerator: GPU T4 x2, then Save Version > "
      "Save & Run All (Commit).")"""),
        ("code", f"""# 2. Train variant "{variant}" on folds 1 and 2, then score on development data. Everything goes to /kaggle/working.
VARIANT = "{variant}"
if GPUS:
    script = next(Path("/kaggle/input").rglob("run6_dev.py"))
    p = subprocess.Popen([sys.executable, "-u", str(script), "--variant", VARIANT, "--folds", "1,2"],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line, end="", flush=True)
    p.wait()
    print("exit code", p.returncode)"""),
    ]


def notebook(parts):
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "cells": []}
    for i, (kind, text) in enumerate(parts):
        lines = text.strip().split("\n")
        cell = {"cell_type": kind, "id": f"cell{i}", "metadata": {},
                "source": [ln + "\n" for ln in lines[:-1]] + [lines[-1]]}
        if kind == "code":
            cell.update({"execution_count": None, "outputs": []})
        nb["cells"].append(cell)
    return nb


def main():
    shutil.copyfile(ROOT.parent / "s1s2water-check/kernel/score_s1s2.py", ROOT / "kernel6-dev/score_s1s2.py")
    code = ROOT / "dev6-code"
    code.mkdir(exist_ok=True)
    for f in ("retrain6.py", "run6_dev.py", "score_s1s2.py", "split_dates.csv"):
        shutil.copyfile(ROOT / "kernel6-dev" / f, code / f)
    (code / "dataset-metadata.json").write_text(json.dumps({
        "title": "IS4990 Module A dev6 code", "id": f"{OWNER}/{CODE}",
        "subtitle": "Series 2 development runs (PROTOCOL-6, DEV-6): training and development scoring",
        "licenses": [{"name": "other"}]}, indent=2), encoding="utf-8")
    print("wrote", code)
    for variant, (what, hours) in VARIANTS.items():
        d = ROOT / "kaggle-notebooks" / f"kernel6-dev-{variant}"
        d.mkdir(exist_ok=True)
        name = f"dev6-{variant}.ipynb"
        (d / name).write_text(json.dumps(notebook(cells(variant, what, hours)), indent=1), encoding="utf-8")
        (d / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{OWNER}/is4990-module-a-dev6-{variant}", "title": f"IS4990 Module A dev6 {variant}",
            "code_file": name, "language": "python", "kernel_type": "notebook", "is_private": True,
            "enable_gpu": False, "enable_tpu": False, "enable_internet": True, "dataset_sources": SOURCES,
            "competition_sources": [], "kernel_sources": [], "model_sources": []}, indent=2), encoding="utf-8")
        print("wrote", d)


if __name__ == "__main__":
    main()
