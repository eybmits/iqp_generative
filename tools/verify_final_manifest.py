#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify artifact manifest for outputs/final_plots."""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path


DEFAULT_EXCLUDES = {"ARTIFACT_MANIFEST.csv", "ARTIFACT_MANIFEST.md", ".DS_Store"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        r = csv.DictReader(fh)
        needed = {"path", "size_bytes", "sha256"}
        missing = needed - set(r.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest missing required columns: {sorted(missing)}")
        for row in r:
            rows.append({"path": row["path"], "size_bytes": row["size_bytes"], "sha256": row["sha256"]})
    return rows


def infer_root_from_rows(rows: list[dict[str, str]]) -> Path:
    if not rows:
        return Path("outputs/final_plots").resolve()

    parts = rows[0]["path"].split("/")
    if len(parts) >= 2:
        candidate = Path(parts[0]) / parts[1]
        return candidate.resolve()
    return Path("outputs/final_plots").resolve()


def collect_actual(root: Path, excludes: set[str]) -> set[str]:
    repo_root = Path.cwd()
    out: set[str] = set()
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.name in excludes:
            continue
        out.add(p.relative_to(repo_root).as_posix())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify final artifact manifest.")
    ap.add_argument("--manifest", type=str, default="outputs/final_plots/ARTIFACT_MANIFEST.csv")
    ap.add_argument("--root", type=str, default="", help="Artifact root override (default: inferred).")
    ap.add_argument("--strict", type=int, default=0, help="If 1, fail on unexpected files in artifact root.")
    ap.add_argument(
        "--exclude",
        type=str,
        default=",".join(sorted(DEFAULT_EXCLUDES)),
        help="Comma-separated file names ignored for strict set comparison.",
    )
    args = ap.parse_args()

    repo_root = Path.cwd()
    manifest = Path(args.manifest)
    rows = load_manifest(manifest)

    root = Path(args.root).resolve() if str(args.root).strip() else infer_root_from_rows(rows)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Artifact root not found: {root}")

    ok = 0
    fail = 0
    for row in rows:
        p = repo_root / row["path"]
        if not p.exists():
            print(f"[missing] {row['path']}")
            fail += 1
            continue

        got_size = str(p.stat().st_size)
        if got_size != row["size_bytes"]:
            print(f"[size-mismatch] {row['path']} expected={row['size_bytes']} got={got_size}")
            fail += 1
            continue

        got_sha = sha256_file(p)
        if got_sha != row["sha256"]:
            print(f"[sha-mismatch] {row['path']} expected={row['sha256']} got={got_sha}")
            fail += 1
            continue

        ok += 1

    if int(args.strict) == 1:
        excludes = {x.strip() for x in str(args.exclude).split(",") if x.strip()}
        actual = collect_actual(root=root, excludes=excludes)
        expected = {r["path"] for r in rows}

        unexpected = sorted(actual - expected)
        missing_from_root = sorted(expected - actual)

        if unexpected:
            print("[strict] unexpected files:")
            for p in unexpected:
                print(f"  - {p}")
            fail += len(unexpected)

        if missing_from_root:
            print("[strict] manifest paths missing in root scan:")
            for p in missing_from_root:
                print(f"  - {p}")
            fail += len(missing_from_root)

    print(f"[summary] verified={ok} failed={fail} strict={int(args.strict)}")
    if fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
