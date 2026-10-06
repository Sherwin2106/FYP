#!/usr/bin/env python
"""Build the local English ConceptNet 5.7 SQLite database used by Module 3.

Downloads the official assertions dump (~475 MB, gzip) and keeps only edges whose
two ends are English concepts (~3.4 M edges, ~400 MB on disk).

    python scripts/build_conceptnet_db.py
    python scripts/build_conceptnet_db.py --dump /path/to/conceptnet-assertions-5.7.0.csv.gz
"""

from __future__ import annotations

import argparse
import gzip
import json
import sqlite3
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DUMP_URL = "https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz"
SKIP_RELATIONS = {"ExternalURL", "dbpedia"}


def download(url: str, dest: Path) -> None:
    from tqdm import tqdm
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "svcr/1.0"})
    with urllib.request.urlopen(req) as resp, open(tmp, "wb") as fh:
        total = int(resp.headers.get("Content-Length", 0))
        with tqdm(total=total, unit="B", unit_scale=True, desc="Downloading ConceptNet") as bar:
            while chunk := resp.read(1 << 20):
                fh.write(chunk)
                bar.update(len(chunk))
    tmp.rename(dest)


def term(uri: str) -> str | None:
    parts = uri.split("/")
    if len(parts) < 4 or parts[1] != "c" or parts[2] != "en":
        return None
    return parts[3]


def build(dump: Path, db_path: Path) -> None:
    from tqdm import tqdm
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_db = db_path.with_suffix(".building")
    tmp_db.unlink(missing_ok=True)
    conn = sqlite3.connect(str(tmp_db))
    conn.executescript("""
        PRAGMA journal_mode = OFF;
        PRAGMA synchronous = OFF;
        CREATE TABLE raw (rel TEXT, start TEXT, end TEXT, weight REAL, surface TEXT);
    """)
    batch, kept, t0 = [], 0, time.time()
    with gzip.open(dump, "rt", encoding="utf-8") as fh:
        for line in tqdm(fh, desc="Filtering English edges", unit=" lines", total=34_074_917):
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 5:
                continue
            s, e = term(cols[2]), term(cols[3])
            if not s or not e or s == e:
                continue
            rel = cols[1].split("/")[2]
            if rel in SKIP_RELATIONS:
                continue
            try:
                info = json.loads(cols[4])
            except json.JSONDecodeError:
                info = {}
            batch.append((rel, s, e, float(info.get("weight", 1.0)), info.get("surfaceText") or ""))
            if len(batch) >= 100_000:
                conn.executemany("INSERT INTO raw VALUES (?,?,?,?,?)", batch)
                kept += len(batch)
                batch.clear()
    if batch:
        conn.executemany("INSERT INTO raw VALUES (?,?,?,?,?)", batch)
        kept += len(batch)
    print(f"Kept {kept:,} English edges in {time.time() - t0:.0f}s. De-duplicating and indexing ...")
    conn.executescript("""
        CREATE TABLE edges AS
            SELECT rel, start, end, MAX(weight) AS weight, MAX(surface) AS surface
            FROM raw GROUP BY rel, start, end;
        DROP TABLE raw;
        CREATE INDEX idx_start ON edges(start);
        CREATE INDEX idx_end ON edges(end);
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
    """)
    n = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    conn.executemany("INSERT INTO meta VALUES (?, ?)", [
        ("source", DUMP_URL), ("version", "5.7.0"), ("edges", str(n)), ("built", time.strftime("%Y-%m-%d")),
    ])
    conn.commit()
    conn.execute("VACUUM")
    conn.close()
    tmp_db.replace(db_path)
    print(f"Done: {n:,} unique edges -> {db_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", type=Path, default=ROOT / "data/conceptnet/conceptnet-assertions-5.7.0.csv.gz")
    ap.add_argument("--db", type=Path, default=ROOT / "data/conceptnet/conceptnet_en.sqlite")
    ap.add_argument("--keep-dump", action="store_true", help="keep the downloaded .csv.gz afterwards")
    args = ap.parse_args()

    if args.db.exists():
        print(f"{args.db} already exists - delete it to rebuild.")
        return 0
    downloaded = False
    if not args.dump.exists():
        download(DUMP_URL, args.dump)
        downloaded = True
    build(args.dump, args.db)
    if downloaded and not args.keep_dump:
        args.dump.unlink()
    return 0


if __name__ == "__main__":
    sys.exit(main())
