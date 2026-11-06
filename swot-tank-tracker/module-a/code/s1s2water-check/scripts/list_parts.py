r"""Lists what is inside each S1S2-Water zip on Zenodo without downloading it (3 Nov 2026).

Reads each zip's central directory with HTTP range requests, then the small per-scene metadata JSON files.
Writes data/files.csv (every file in every part, with sizes) and data/scenes.json (each scene's metadata).
Run from the project folder:
    temp\feasibility-check\.venv\Scripts\python.exe -I temp\s1s2water-check\scripts\list_parts.py
"""
import csv
import io
import json
import urllib.request
import zipfile
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
RECORD = json.loads((DATA / "zenodo_record.json").read_text(encoding="utf-8"))


class HttpFile(io.RawIOBase):
    """A read-only, seekable file over HTTP range requests."""

    def __init__(self, url, size):
        self.url, self.size, self.pos = url, size, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(b), self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


def main():
    rows, scenes = [], {}
    for f in sorted(RECORD["files"], key=lambda f: f["key"]):
        if not f["key"].endswith(".zip"):
            continue
        raw = HttpFile(f["links"]["self"], f["size"])
        z = zipfile.ZipFile(io.BufferedReader(raw, buffer_size=1 << 20))
        for i in z.infolist():
            rows.append({"part": f["key"], "name": i.filename, "mb": round(i.file_size / 1e6, 1)})
            if i.filename.endswith("_meta.json"):
                scenes[i.filename] = {"part": f["key"], **json.loads(z.read(i))}
        print(f["key"], len(z.infolist()), "files")
    with open(DATA / "files.csv", "w", newline="", encoding="utf-8") as out:
        w = csv.DictWriter(out, fieldnames=["part", "name", "mb"])
        w.writeheader()
        w.writerows(rows)
    (DATA / "scenes.json").write_text(json.dumps(scenes, indent=1), encoding="utf-8")
    print(len(rows), "files,", len(scenes), "scenes")


if __name__ == "__main__":
    main()
