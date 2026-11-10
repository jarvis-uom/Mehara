r"""Shares one private Kaggle dataset with one user as a reader (viewer), keeping it private, and checks it.

    temp\feasibility-check\.venv\Scripts\python.exe -I temp\module-a-retrain\scripts\share_dataset.py <dataset slug> <kaggle username>

Steps: download the current metadata, back it up (tmp/share/<slug>/, git-ignored), stop if the dataset is not
private, add the reader with "isPrivate": true written explicitly, update, download the metadata again and compare the
settings, then check from outside (no login) that the dataset is still private (HTTP 403).
"""
import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KAGGLE = ROOT.parent / "feasibility-check/.venv/Scripts/kaggle.exe"
OWNER = "mehararothila"
KEYS = ("title", "subtitle", "description", "isPrivate", "licenses", "expectedUpdateFrequency", "keywords")


def kaggle(*args, cwd=None):
    r = subprocess.run([str(KAGGLE), *args], capture_output=True, text=True, cwd=cwd)
    if r.returncode:
        raise SystemExit(f"kaggle {' '.join(args)} failed: {r.stdout} {r.stderr}")
    return r.stdout.strip()


def info(path):
    m = json.loads(Path(path).read_text(encoding="utf-8"))
    return m, m.get("info", m)


def outside_status(slug):
    try:
        return urllib.request.urlopen(f"https://www.kaggle.com/api/v1/datasets/view/{OWNER}/{slug}").status
    except urllib.error.HTTPError as e:
        return e.code


def main():
    slug, user = sys.argv[1], sys.argv[2]
    d = ROOT / "tmp/share" / slug
    (d / "after").mkdir(parents=True, exist_ok=True)
    kaggle("datasets", "metadata", f"{OWNER}/{slug}", "-p", str(d))
    meta = d / "dataset-metadata.json"
    (d / "before.json").write_text(meta.read_text(encoding="utf-8"), encoding="utf-8")
    m, i = info(meta)
    if i.get("isPrivate") is not True:
        raise SystemExit(f"{slug}: not private, so nothing was changed")
    i["isPrivate"] = True
    others = [c for c in i.get("collaborators") or [] if c.get("username") != user]  # keep everyone already shared
    i["collaborators"] = others + [{"username": user, "role": "reader"}]
    meta.write_text(json.dumps(m, indent=2), encoding="utf-8")
    print(kaggle("datasets", "metadata", f"{OWNER}/{slug}", "-p", ".", "--update", cwd=d))
    kaggle("datasets", "metadata", f"{OWNER}/{slug}", "-p", str(d / "after"))
    a, b = info(d / "before.json")[1], info(d / "after" / "dataset-metadata.json")[1]
    changed = {k: (a.get(k), b.get(k)) for k in KEYS if a.get(k) != b.get(k)}
    status = outside_status(slug)
    print(f"{slug}: settings {'unchanged' if not changed else 'CHANGED ' + str(changed)}; "
          f"from outside {status} ({'private' if status in (403, 404) else 'CHECK: may be public'})")


if __name__ == "__main__":
    main()
