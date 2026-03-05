#!/usr/bin/env python3
"""Build a deterministic artifact manifest with file sizes and SHA256 hashes."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


SKIP_NAMES = {".DS_Store"}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _iter_files(root: Path, *, include_hidden: bool, output_abs: Path) -> list[Path]:
    files: list[Path] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if p.resolve() == output_abs:
            continue
        if p.name in SKIP_NAMES:
            continue
        rel = p.relative_to(root)
        if not include_hidden and any(part.startswith(".") for part in rel.parts):
            continue
        files.append(p)
    return files


def main() -> None:
    ap = argparse.ArgumentParser(description="Build CSV manifest for a directory tree.")
    ap.add_argument("root", type=Path, help="Directory to scan")
    ap.add_argument("--output", type=Path, default=None, help="Output CSV path (default: <root>/ARTIFACT_MANIFEST.csv)")
    ap.add_argument("--include-hidden", action="store_true", help="Include hidden files and hidden directories")
    args = ap.parse_args()

    root = args.root.resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Root must be an existing directory: {root}")

    output = args.output.resolve() if args.output else (root / "ARTIFACT_MANIFEST.csv").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    files = _iter_files(root, include_hidden=args.include_hidden, output_abs=output)

    repo_root = Path.cwd().resolve()

    with output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["path", "size_bytes", "sha256"])
        writer.writeheader()
        for p in files:
            try:
                path_str = p.resolve().relative_to(repo_root).as_posix()
            except ValueError:
                path_str = p.as_posix()
            writer.writerow(
                {
                    "path": path_str,
                    "size_bytes": p.stat().st_size,
                    "sha256": _sha256(p),
                }
            )

    print(f"[ok] wrote {len(files)} rows -> {output.as_posix()}")


if __name__ == "__main__":
    main()
