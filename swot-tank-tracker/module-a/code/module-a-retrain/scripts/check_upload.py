r"""Checks that a Kaggle dataset holds exactly the files of its local folder: same names, same sizes (2 Nov 2026).
Used before a dataset is made public. Exit code 0 when they match.
Run from the project folder, for example:
    temp\feasibility-check\.venv\Scripts\python.exe -I temp\module-a-retrain\scripts\check_upload.py
        mehararothila/is4990-module-a-pass-chips temp\module-a-retrain\kaggle-data\passes
"""
import csv
import io
import re
import subprocess
import sys
from pathlib import Path

KAGGLE = Path(__file__).resolve().parents[3] / "temp/feasibility-check/.venv/Scripts/kaggle.exe"


def remote(slug):
    files, token = {}, None
    while True:
        cmd = [str(KAGGLE), "datasets", "files", slug, "--page-size", "200", "--csv"] + (["--page-token", token] if token else [])
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
        m = re.search(r"Next Page Token = (\S+)", out)
        token = m.group(1) if m else None
        rows = [ln for ln in out.splitlines() if ln and not ln.startswith(("Next Page Token", "name,size"))]
        for r in csv.reader(io.StringIO("\n".join(rows))):
            files[r[0].replace("\\", "/")] = int(r[1])
        if not token:
            return files


def main():
    slug, folder = sys.argv[1], Path(sys.argv[2])
    there = remote(slug)
    here = {p.relative_to(folder).as_posix(): p.stat().st_size for p in folder.rglob("*")
            if p.is_file() and p.name != "dataset-metadata.json"}
    missing, extra = sorted(set(here) - set(there)), sorted(set(there) - set(here))
    wrong = sorted(n for n in set(here) & set(there) if here[n] != there[n])
    print(f"{slug}: {len(there)} files on Kaggle, {len(here)} here; missing {len(missing)}, extra {len(extra)}, "
          f"different size {len(wrong)}; {sum(here.values()) / 1e9:.2f} GB")
    sys.exit(0 if not (missing or extra or wrong) else 1)


if __name__ == "__main__":
    main()
