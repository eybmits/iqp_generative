#!/usr/bin/env python3
"""Verify one or more artifact manifests (CSV with path,size_bytes,sha256)."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_manifest(manifest: Path, repo_root: Path) -> tuple[int, int]:
    checked = 0
    failed = 0

    with manifest.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    required = {"path", "size_bytes", "sha256"}
    if not rows:
        print(f"[warn] empty manifest: {manifest.as_posix()}")
        return 0, 0
    if not required.issubset(rows[0].keys()):
        missing = sorted(required.difference(rows[0].keys()))
        raise SystemExit(f"manifest {manifest.as_posix()} missing columns: {missing}")

    for row in rows:
        checked += 1
        path_value = str(row["path"]).strip()
        target = Path(path_value)
        if not target.is_absolute():
            target = (repo_root / target).resolve()

        if not target.exists():
            failed += 1
            print(f"[fail] missing: {path_value}")
            continue

        expected_size = int(row["size_bytes"])
        real_size = target.stat().st_size
        if real_size != expected_size:
            failed += 1
            print(f"[fail] size mismatch: {path_value} expected={expected_size} got={real_size}")
            continue

        expected_hash = str(row["sha256"]).strip().lower()
        real_hash = _sha256(target)
        if real_hash != expected_hash:
            failed += 1
            print(f"[fail] hash mismatch: {path_value}")
            continue

    print(f"[ok] {manifest.as_posix()} checked={checked} failed={failed}")
    return checked, failed


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify artifact checksums and file sizes from CSV manifests.")
    ap.add_argument("manifests", nargs="+", type=Path, help="Manifest CSV path(s)")
    ap.add_argument("--repo-root", type=Path, default=Path.cwd(), help="Base directory for relative paths")
    args = ap.parse_args()

    repo_root = args.repo_root.resolve()
    total_checked = 0
    total_failed = 0
    for manifest in args.manifests:
        m = manifest.resolve()
        if not m.exists():
            raise SystemExit(f"manifest not found: {m.as_posix()}")
        checked, failed = _verify_manifest(m, repo_root)
        total_checked += checked
        total_failed += failed

    if total_failed > 0:
        raise SystemExit(f"verification failed: {total_failed} mismatches across {total_checked} checks")
    print(f"[done] all manifests valid ({total_checked} files)")


if __name__ == "__main__":
    main()
