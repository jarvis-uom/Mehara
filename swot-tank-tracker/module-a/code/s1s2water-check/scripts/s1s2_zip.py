r"""Reads files out of S1S2-Water's zips on Zenodo without downloading the zips (3 Nov 2026).

Used by fetch_gee_radar.py: one range request per zip member, inflated on the fly, with the size and CRC checked.
"""
import io
import json
import struct
import time
import urllib.request
import zipfile
import zlib

import numpy as np

ZENODO = "https://zenodo.org/api/records/11278238/files/{}/content"
PART_BYTES = {"part1.zip": 29478583513, "part2.zip": 32717311258, "part3.zip": 30763336558, "part4.zip": 29709506508,
              "part5.zip": 19792863346, "part6.zip": 23079787372}  # from the Zenodo record (v1.0.1)
MAIN = {23: "part2.zip", 25: "part2.zip", 29: "part2.zip", 33: "part3.zip", 35: "part3.zip", 47: "part3.zip",
        53: "part4.zip", 57: "part4.zip", 75: "part5.zip", 77: "part5.zip", 78: "part5.zip", 80: "part5.zip",
        82: "part5.zip", 88: "part6.zip", 89: "part6.zip"}  # the test split without scene 31 (PROTOCOL-S1S2)
EXTRA = {6: "part1.zip", 7: "part1.zip", 11: "part1.zip", 12: "part1.zip", 52: "part4.zip"}  # South/Southeast Asia


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class HttpFile(io.RawIOBase):
    """A read-only, seekable file over HTTP range requests (for a zip's central directory)."""

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
        data = get_range(self.url, self.pos, min(self.pos + len(b), self.size) - 1)
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


def get_range(url, a, b):
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except OSError as e:
            log(f"   range {a}-{b} failed ({e}); retry {attempt + 1}")
            time.sleep(5 * (attempt + 1))
    raise OSError(f"range {a}-{b} of {url} failed 5 times")


def zip_index(part, cache):
    """The part's members by name, read once per part into the worker's own cache (a ZipFile is not thread-safe)."""
    if part not in cache:
        url = ZENODO.format(part)
        z = zipfile.ZipFile(io.BufferedReader(HttpFile(url, PART_BYTES[part]), buffer_size=1 << 16))
        cache[part] = (url, z, {i.filename: i for i in z.infolist()})
    return cache[part]


def fetch_member(part, name, dest, cache):
    """Streams one zip member to dest with a single range request, inflating it on the fly; checks size and CRC."""
    url, _, members = zip_index(part, cache)
    info = members[name]
    n, e = struct.unpack("<HH", get_range(url, info.header_offset, info.header_offset + 29)[26:30])
    start = info.header_offset + 30 + n + e
    for attempt in range(4):
        try:
            inflate = zlib.decompressobj(-15) if info.compress_type == zipfile.ZIP_DEFLATED else None
            crc, size = 0, 0
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{start + info.compress_size - 1}"})
            with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as out:
                while True:
                    chunk = r.read(8 << 20)
                    if not chunk:
                        break
                    data = inflate.decompress(chunk) if inflate else chunk
                    crc, size = zlib.crc32(data, crc), size + len(data)
                    out.write(data)
                if inflate:
                    data = inflate.flush()
                    crc, size = zlib.crc32(data, crc), size + len(data)
                    out.write(data)
            if size == info.file_size and crc == info.CRC:
                return dest
            log(f"   {name}: size or CRC wrong ({size} of {info.file_size}); retry {attempt + 1}")
        except OSError as err:
            log(f"   {name}: {err}; retry {attempt + 1}")
        time.sleep(10 * (attempt + 1))
    raise OSError(f"{name} could not be downloaded")


def fetch_scene(sid, part, folder, cache):
    """The scene's radar image, water mask and valid mask, plus its metadata."""
    _, z, _ = zip_index(part, cache)
    meta = json.loads(z.read(f"{sid}/sentinel12_{sid}_meta.json"))
    t0 = time.time()
    files = {k: fetch_member(part, f"{sid}/sentinel12_s1_{sid}_{k}.tif", folder / f"s1_{sid}_{k}.tif", cache)
             for k in ("img", "msk", "valid")}
    mb = sum(f.stat().st_size for f in files.values()) / 1e6
    log(f"   scene {sid}: {mb:.0f} MB in {time.time() - t0:.0f} s")
    return meta, files


def latlon(meta):
    x0, y0, x1, y1 = meta["bbox"]
    x, y = (x0 + x1) / 2, (y0 + y1) / 2
    return float(np.degrees(2 * np.arctan(np.exp(y / 6378137)) - np.pi / 2)), float(np.degrees(x / 6378137))
