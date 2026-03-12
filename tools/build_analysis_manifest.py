#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build deterministic artifact manifests for outputs/analysis."""

from __future__ import annotations

import argparse
import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_EXCLUDES = {"ARTIFACT_MANIFEST.csv", "ARTIFACT_MANIFEST.md", ".DS_Store"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_files(root: Path, excludes: set[str]) -> list[Path]:
    files: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.name in excludes:
            continue
        files.append(p)
    files.sort(key=lambda p: p.as_posix())
    return files


def to_repo_rel(path: Path) -> str:
    return path.relative_to(Path.cwd()).as_posix()


def write_csv(out_csv: Path, rows: list[dict[str, str]]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["path", "size_bytes", "sha256"], lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_md(out_md: Path, rows: list[dict[str, str]]) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "# Analysis Artifact Manifest",
        "",
        f"Generated: {now}",
        "",
        "This manifest records SHA-256 checksums and file sizes for curated files in `outputs/analysis`.",
        "",
        "| path | size_bytes | sha256 |",
        "|---|---:|---|",
    ]
    for row in rows:
        lines.append(f"| {row['path']} | {row['size_bytes']} | `{row['sha256']}` |")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build deterministic analysis artifact manifests.")
    ap.add_argument("--root", type=str, default="outputs/analysis", help="Artifact root directory.")
    ap.add_argument("--output-csv", type=str, default="outputs/analysis/ARTIFACT_MANIFEST.csv")
    ap.add_argument("--output-md", type=str, default="outputs/analysis/ARTIFACT_MANIFEST.md")
    ap.add_argument(
        "--exclude",
        type=str,
        default=",".join(sorted(DEFAULT_EXCLUDES)),
        help="Comma-separated file names to exclude by basename.",
    )
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Artifact root not found or not a directory: {root}")

    excludes = {x.strip() for x in str(args.exclude).split(",") if x.strip()}
    rows: list[dict[str, str]] = []
    for path in collect_files(root=root, excludes=excludes):
        rows.append(
            {
                "path": to_repo_rel(path),
                "size_bytes": str(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )

    out_csv = Path(args.output_csv)
    out_md = Path(args.output_md)
    write_csv(out_csv=out_csv, rows=rows)
    write_md(out_md=out_md, rows=rows)

    print(f"[saved] {out_csv}")
    print(f"[saved] {out_md}")
    print(f"[info] files={len(rows)}")


if __name__ == "__main__":
    main()
