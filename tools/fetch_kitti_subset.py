"""
Downloads part of one SemanticKITTI sequence without the 80 GB KITTI zip: the servers allow HTTP
range requests, so this reads each zip's directory (a few MB) and then fetches only the requested
files. The scans are stored uncompressed in the zip, so each one costs just its ~2 MB.

  /opt/miniconda3/bin/python fetch_kitti_subset.py <dataset dir> [sequence=08] [first=0] [count=500]

Files that sit next to each other in a zip are fetched together in one request (up to 64 MB), so
many small files (the labels, ~7 KB each) don't cost a slow round-trip each.

Writes <dataset dir>/sequences/<seq>/{velodyne/*.bin, labels/*.label, poses.txt, calib.txt}, the
layout AI/Kitti.h reads. Scans are fetched from `first` for `count`; keep first = 0 so scan numbers
line up with the pose lines. Files already on disk with the right size are skipped, so it can be
re-run (e.g. after switching networks) and just continues.
"""
import io
import struct
import sys
import time
import urllib.request
import zipfile
import zlib
from pathlib import Path

VELODYNE_ZIP = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_velodyne.zip"
CALIB_ZIP = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_calib.zip"
LABELS_ZIP = "https://www.semantic-kitti.org/assets/data_odometry_labels.zip"


class RangeFile(io.RawIOBase):
    """A remote file as a seekable file object; every read is an HTTP range request."""

    def __init__(self, url):
        self.url, self.pos, self.fetched = url, 0, 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as r:
            if r.headers.get("Accept-Ranges") != "bytes":
                sys.exit(f"{url} doesn't support range requests")
            self.size = int(r.headers["Content-Length"])

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size) - 1
        for attempt in range(5):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    data = r.read()
                break
            except OSError as e:          # network hiccup / switching Wi-Fi: wait and retry
                if attempt == 4:
                    raise
                print(f"  retrying after: {e}")
                time.sleep(5 * (attempt + 1))
        self.pos += len(data)
        self.fetched += len(data)
        return data

    def readinto(self, b):
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)

    def read_range(self, start, end):
        self.seek(start)
        return self.read(end - start)


def open_remote_zip(url):
    rf = RangeFile(url)
    return zipfile.ZipFile(io.BufferedReader(rf, buffer_size=1 << 20)), rf


def member_bytes(buf, base, info):
    """One member's contents, from `buf` (the zip's bytes starting at offset `base`)."""
    off = info.header_offset - base
    if buf[off:off + 4] != b"PK\x03\x04":
        raise ValueError(f"bad local header for {info.filename}")
    name_len, extra_len = struct.unpack_from("<HH", buf, off + 26)
    start = off + 30 + name_len + extra_len
    data = buf[start:start + info.compress_size]
    if info.compress_type == zipfile.ZIP_STORED:
        raw = data
    elif info.compress_type == zipfile.ZIP_DEFLATED:
        raw = zlib.decompress(data, -15)
    else:
        raise ValueError(f"unsupported compression for {info.filename}")
    if len(raw) != info.file_size or zlib.crc32(raw) != info.CRC:
        raise ValueError(f"checksum mismatch for {info.filename}")
    return raw


def batches(infos, max_bytes=64 << 20, max_gap=1 << 20):
    """Groups members that are close together in the zip into ranges fetched with one request."""
    run = []
    for info in sorted(infos, key=lambda i: i.header_offset):
        end = info.header_offset + 30 + len(info.filename.encode()) + 1024 + info.compress_size
        if run and (info.header_offset - run_end > max_gap or end - run_start > max_bytes):
            yield run_start, run_end, run
            run = []
        if not run:
            run_start = info.header_offset
        run.append(info)
        run_end = end
    if run:
        yield run_start, run_end, run


def write(dest, raw):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(raw)
    tmp.rename(dest)


def main():
    root = Path(sys.argv[1])
    seq = sys.argv[2] if len(sys.argv) > 2 else "08"
    first = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    count = int(sys.argv[4]) if len(sys.argv) > 4 else 500
    wanted = {f"{i:06d}" for i in range(first, first + count)}
    prefix = f"dataset/sequences/{seq}/"
    out = root / "sequences" / seq
    total = 0

    for url, keep in [
        (CALIB_ZIP, lambda n: n.endswith("calib.txt")),
        (LABELS_ZIP, lambda n: n.endswith("poses.txt") or (n.endswith(".label") and Path(n).stem in wanted)),
        (VELODYNE_ZIP, lambda n: n.endswith(".bin") and Path(n).stem in wanted),
    ]:
        z, rf = open_remote_zip(url)
        infos = [i for i in z.infolist() if i.filename.startswith(prefix) and keep(i.filename)]
        # Skip what's already on disk at the right size (a re-run just continues).
        todo = [i for i in infos
                if not ((d := out / i.filename[len(prefix):]).exists() and d.stat().st_size == i.file_size)]
        size = sum(i.compress_size for i in todo)
        print(f"{url.rsplit('/', 1)[1]}: {len(infos)} files, {len(infos) - len(todo)} already here, "
              f"{size / 1e6:.1f} MB to fetch")
        done = 0
        for start, end, run in batches(todo):
            buf = rf.read_range(start, min(end, rf.size))
            for info in run:
                write(out / info.filename[len(prefix):], member_bytes(buf, start, info))
            done += len(run)
            print(f"  {done}/{len(todo)}  ({rf.fetched / 1e6:.0f} MB downloaded)")
        total += rf.fetched
    print(f"done: {out} ({total / 1e6:.0f} MB downloaded in total)")


if __name__ == "__main__":
    main()
