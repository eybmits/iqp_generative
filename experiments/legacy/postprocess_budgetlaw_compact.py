#!/usr/bin/env python3
"""Post-process Budgetlaw compact scatter PDFs (label + underlay alignment).

This utility applies a deterministic micro-layout fix for the "best: IQP parity"
annotation in claim-35 compact scatter figures:
1) move the label text by (dx, dy),
2) keep the white rounded underlay exactly aligned to the label baseline,
3) optionally apply a subtle global zoom,
4) optionally re-render PNGs from the updated PDFs.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Iterator, List, Sequence, Tuple

from pypdf import PdfReader, PdfWriter, Transformation
from pypdf.generic import ContentStream, FloatObject, NameObject


DEFAULT_PDFS = [
    Path(
        "outputs/paper_even_final/35_claim_budgetlaw_global_m200_custom/"
        "budgetlaw_scatter_dual_holdout_m200_beta06to14_all_experiments_compact.pdf"
    ),
    Path(
        "outputs/paper_even_final/35_claim_budgetlaw_global_m200_custom/"
        "budgetlaw_scatter_dual_holdout_m200_beta06to14_all_experiments_compact_annotated.pdf"
    ),
]


def _flatten_strings(obj: object) -> Iterator[str]:
    if isinstance(obj, str):
        yield obj
        return
    if isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _flatten_strings(item)


def _contains_label(
    operations: Sequence[Tuple[object, bytes]],
    start_idx: int,
    label_fragment: str,
    window: int = 30,
) -> bool:
    stop = min(len(operations), start_idx + window)
    for j in range(start_idx + 1, stop):
        operands, op = operations[j]
        if op != b"TJ":
            continue
        text = "".join(_flatten_strings(operands))
        if label_fragment in text:
            return True
    return False


def _find_label_cm(
    operations: Sequence[Tuple[object, bytes]],
    label_fragment: str,
) -> Tuple[int, float, float]:
    for i, (operands, op) in enumerate(operations):
        if op != b"cm" or len(operands) != 6:
            continue
        x = float(operands[4])
        y = float(operands[5])
        if _contains_label(operations, i, label_fragment):
            return i, x, y
    raise RuntimeError(f'Could not find label transform containing "{label_fragment}".')


def _path_start(operations: Sequence[Tuple[object, bytes]], f_idx: int) -> int:
    path_ops = {b"m", b"l", b"c", b"h"}
    i = f_idx - 1
    while i >= 0 and operations[i][1] in path_ops:
        i -= 1
    return i + 1


def _path_extents(
    operations: Sequence[Tuple[object, bytes]], start_idx: int, f_idx: int
) -> Tuple[float, float, float, float]:
    xs: List[float] = []
    ys: List[float] = []
    for i in range(start_idx, f_idx):
        operands, op = operations[i]
        if op not in (b"m", b"l", b"c"):
            continue
        for k in range(0, len(operands), 2):
            xs.append(float(operands[k]))
            ys.append(float(operands[k + 1]))
    if not xs or not ys:
        raise RuntimeError("Path extents could not be computed (empty coordinate list).")
    return min(xs), max(xs), min(ys), max(ys)


def _find_underlay_path(
    operations: Sequence[Tuple[object, bytes]],
    label_idx: int,
    target_left: float,
    target_bottom: float,
    lookback: int = 140,
) -> Tuple[int, int]:
    best = None
    best_score = float("inf")
    lo = max(0, label_idx - lookback)

    for i in range(label_idx - 1, lo - 1, -1):
        if operations[i][1] != b"f":
            continue
        start = _path_start(operations, i)
        left, right, bottom, top = _path_extents(operations, start, i)
        width = right - left
        height = top - bottom
        if width < 40.0 or height < 5.0:
            # Skip small filled glyph/marker paths; underlay is a large rounded box.
            continue
        score = abs(left - target_left) + abs(bottom - target_bottom)
        if score < best_score:
            best_score = score
            best = (start, i)

    if best is None:
        raise RuntimeError("Could not find underlay filled path near the label.")
    return best


def _shift_path_coords(
    operations: List[Tuple[object, bytes]],
    start_idx: int,
    f_idx: int,
    dx: float,
    dy: float,
) -> None:
    for i in range(start_idx, f_idx):
        operands, op = operations[i]
        if op not in (b"m", b"l", b"c"):
            continue
        for k in range(0, len(operands), 2):
            operands[k] = FloatObject(float(operands[k]) + dx)
            operands[k + 1] = FloatObject(float(operands[k + 1]) + dy)


def _patch_pdf(
    pdf_path: Path,
    label_fragment: str,
    text_dx: float,
    text_dy: float,
    align_underlay: bool,
    underlay_bottom_offset: float,
    zoom: float,
) -> Tuple[int, int]:
    reader = PdfReader(str(pdf_path))
    writer = PdfWriter()
    moved_text = 0
    moved_underlay = 0

    for page in reader.pages:
        content = ContentStream(page.get_contents(), reader)
        operations = content.operations

        label_idx, x0, y0 = _find_label_cm(operations, label_fragment)
        operands, _ = operations[label_idx]
        new_x = x0 + text_dx
        new_y = y0 + text_dy
        operands[4] = FloatObject(new_x)
        operands[5] = FloatObject(new_y)
        moved_text += 1

        if align_underlay:
            target_left = new_x
            target_bottom = new_y + underlay_bottom_offset
            underlay_start, underlay_f = _find_underlay_path(
                operations=operations,
                label_idx=label_idx,
                target_left=target_left,
                target_bottom=target_bottom,
            )
            left, _, bottom, _ = _path_extents(operations, underlay_start, underlay_f)
            _shift_path_coords(
                operations=operations,
                start_idx=underlay_start,
                f_idx=underlay_f,
                dx=target_left - left,
                dy=target_bottom - bottom,
            )
            moved_underlay += 1

        page[NameObject("/Contents")] = content

        if zoom != 1.0:
            w = float(page.mediabox.width)
            h = float(page.mediabox.height)
            tx = (1.0 - zoom) * w * 0.5
            ty = (1.0 - zoom) * h * 0.5
            page.add_transformation(
                Transformation().scale(zoom, zoom).translate(tx, ty)
            )

        writer.add_page(page)

    with pdf_path.open("wb") as f:
        writer.write(f)

    return moved_text, moved_underlay


def _render_png(pdf_path: Path, dpi: int, gs_bin: str) -> None:
    png_path = pdf_path.with_suffix(".png")
    cmd = [
        gs_bin,
        "-dSAFER",
        "-dBATCH",
        "-dNOPAUSE",
        "-sDEVICE=pngalpha",
        f"-r{int(dpi)}",
        f"-sOutputFile={png_path}",
        str(pdf_path),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Patch claim-35 compact budgetlaw scatter PDFs: move label text, "
            "re-align underlay box, optional zoom, optional PNG re-render."
        )
    )
    ap.add_argument(
        "--pdf",
        dest="pdfs",
        action="append",
        default=[],
        help="PDF path to patch (repeat to pass multiple files).",
    )
    ap.add_argument(
        "--label-fragment",
        type=str,
        default="best: IQP parity (",
        help="Substring used to identify the annotation text in PDF operations.",
    )
    ap.add_argument(
        "--text-dx",
        type=float,
        default=0.0,
        help="Horizontal text shift in PDF points (+right, -left).",
    )
    ap.add_argument(
        "--text-dy",
        type=float,
        default=0.0,
        help="Vertical text shift in PDF points (+up, -down).",
    )
    ap.add_argument(
        "--underlay-bottom-offset",
        type=float,
        default=-3.311875,
        help=(
            "Target underlay bottom minus label baseline, in points. "
            "Default matches this compact figure style."
        ),
    )
    ap.add_argument(
        "--no-align-underlay",
        action="store_true",
        help="Only move text; keep underlay path untouched.",
    )
    ap.add_argument(
        "--zoom",
        type=float,
        default=1.0,
        help="Global zoom factor applied to full page (1.00 = no zoom).",
    )
    ap.add_argument(
        "--render-png",
        action="store_true",
        help="Re-render sibling PNG from each updated PDF via Ghostscript.",
    )
    ap.add_argument(
        "--png-dpi",
        type=int,
        default=300,
        help="DPI for --render-png (default: 300).",
    )
    ap.add_argument(
        "--gs-bin",
        type=str,
        default="gs",
        help="Ghostscript binary name/path used by --render-png.",
    )
    args = ap.parse_args()

    pdfs = [Path(p) for p in args.pdfs] if args.pdfs else list(DEFAULT_PDFS)
    align_underlay = not bool(args.no_align_underlay)

    for pdf in pdfs:
        if not pdf.exists():
            raise FileNotFoundError(f"PDF not found: {pdf}")
        moved_text, moved_underlay = _patch_pdf(
            pdf_path=pdf,
            label_fragment=args.label_fragment,
            text_dx=float(args.text_dx),
            text_dy=float(args.text_dy),
            align_underlay=align_underlay,
            underlay_bottom_offset=float(args.underlay_bottom_offset),
            zoom=float(args.zoom),
        )
        print(
            f"[updated] {pdf} | text={moved_text} "
            f"underlay={moved_underlay} zoom={float(args.zoom):.4f}"
        )
        if args.render_png:
            _render_png(pdf, dpi=int(args.png_dpi), gs_bin=args.gs_bin)
            print(f"[rendered] {pdf.with_suffix('.png')}")


if __name__ == "__main__":
    main()
