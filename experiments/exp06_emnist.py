#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Experiment 6: EMNIST (real data) holdout-recovery

Experiment_1-style plots on a **real dataset**: **EMNIST** (handwriting),
preprocessed to **n=16** binary variables via 28x28 -> 4x4 average pooling +
thresholding. This yields a fully-enumerable state space 2^16=65,536.

Recovery overlay:
  - fig1_recovery_emnist_overlay.pdf

The overlay plot shows:
  - Target p* (black)
  - Best IQP-QCBM (red) across the (sigma,K) sweep  [best = minimal finite Q80_iqp]
  - Spectral completion q~ (gray dash-dot) at the SAME (sigma,K) as best IQP
  - Many Ising controls (blue -> white) across the (sigma,K) grid
  - Uniform (gray dashed)

Heatmaps (EMNIST target):
  - fig2a_emnist_qH_ratio_iqp.pdf      [RED]
  - fig2a_emnist_qH_ratio_class.pdf    [BLUE]
  - fig2c_emnist_Q80_iqp.pdf           [RED-BLACK]
  - fig2c_emnist_Q80_class.pdf         [BLUE-BLACK]

Notes:
  - Self-contained: no hero_* scripts, no PennyLane.
  - Holdouts H are selected once using Appendix-A heuristic + a visibility check
    in a *reference band* (holdout_sigma, holdout_K). The (sigma,K) sweep trains
    all models on the SAME fixed H.

Run (paper-like defaults):
  python3 experiments/exp06_emnist.py --split byclass --include-test --even-parity-only --include-fields

Run a single setting (convenience aliases):
  python3 experiments/exp06_emnist.py --sigma 1.0 --K 512 --split byclass --include-test --even-parity-only --include-fields
"""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import hashlib
import json
import math
import os
import random
import struct
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Torch is required for training; keep import local-friendly.
try:
    import torch
except Exception as e:  # pragma: no cover
    torch = None
    _TORCH_IMPORT_ERROR = e

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def ensure_outdir(outdir: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    return outdir


def set_style(base: int = 9) -> None:
    plt.rcParams.update(
        {
            "font.size": base,
            "axes.titlesize": base,
            "axes.labelsize": base,
            "legend.fontsize": base - 1,
            "xtick.labelsize": base - 1,
            "ytick.labelsize": base - 1,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def fig_size(which: str, height: float):
    """Figure sizes aligned with REVTeX single-column figures.

    Matches the common helper used in the paper scripts:
      - 'col'  : single-column width (~3.375 in)
      - 'full' : two-column width (~7.0 in)

    height is in inches.
    """
    widths = {"col": 3.375, "half": 3.375, "full": 7.0}
    w = widths.get(which, 3.375)
    return (float(w), float(height))


def q_grid(Qmax: int = 10000) -> np.ndarray:
    """Smooth sampling grid for recovery curves, including small Q."""
    Q = np.unique(
        np.concatenate(
            [
                np.unique(np.logspace(0, 3.5, 120).astype(int)),
                np.linspace(1000, Qmax, 160).astype(int),
                np.array([0, 1, 2, 3, 4, 5, 10, 20, 50, 100], dtype=int),
            ]
        )
    )
    Q = Q[(Q >= 0) & (Q <= Qmax)]
    return Q


# -----------------------------------------------------------------------------
# Fast Walsh-Hadamard transform (FWHT)
# -----------------------------------------------------------------------------


def fwht_inplace_np(a: np.ndarray) -> np.ndarray:
    """In-place unnormalized Walsh-Hadamard transform.

    a must be 1D, length power of two.
    Returns the same array (transformed).
    """
    n = a.shape[0]
    h = 1
    while h < n:
        a = a.reshape(-1, 2 * h)
        x = a[:, :h]
        y = a[:, h : 2 * h]
        t0 = x + y
        t1 = x - y
        a[:, :h] = t0
        a[:, h : 2 * h] = t1
        h *= 2
    return a.reshape(n)


def fwht_np(a: np.ndarray) -> np.ndarray:
    out = np.array(a, copy=True)
    return fwht_inplace_np(out)


def fwht_torch(a: "torch.Tensor") -> "torch.Tensor":
    """Unnormalized FWHT for torch tensors (autograd-friendly)."""
    n = a.shape[0]
    h = 1
    out = a
    while h < n:
        out = out.view(-1, 2 * h)
        x = out[:, :h]
        y = out[:, h : 2 * h]
        t0 = x + y
        t1 = x - y
        out = torch.cat([t0, t1], dim=1)
        h *= 2
    return out.view(n)


# -----------------------------------------------------------------------------
# Bit / popcount helpers
# -----------------------------------------------------------------------------


def make_byte_popcount_table() -> np.ndarray:
    """popcount for 0..255."""
    tbl = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        tbl[i] = bin(i).count("1")
    return tbl


_BYTE_POP = make_byte_popcount_table()


def popcount_uint16(x: np.ndarray) -> np.ndarray:
    """Vectorized popcount for uint16/uint32 arrays using byte table."""
    x = x.astype(np.uint32, copy=False)
    lo = x & 0xFF
    hi = (x >> 8) & 0xFF
    return _BYTE_POP[lo] + _BYTE_POP[hi]


def hamming_distance_u16(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise Hamming distance between arrays of uint16 indices.

    Returns popcount(a xor b).
    """
    return popcount_uint16(np.bitwise_xor(a.astype(np.uint16), b.astype(np.uint16)))


def parity_sign(alpha: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Compute (-1)^{popcount(alpha & x)} for uint16 arrays alpha,x.

    alpha: (K,) uint16
    x: (M,) uint16
    returns: (M,K) int8 in {+1,-1}
    """
    ax = (x[:, None].astype(np.uint16) & alpha[None, :].astype(np.uint16)).astype(np.uint16)
    pc = popcount_uint16(ax.astype(np.uint32))
    return (1 - 2 * (pc & 1)).astype(np.int8)


# -----------------------------------------------------------------------------
# EMNIST download + IDX parsing
# -----------------------------------------------------------------------------


EMNIST_MD5 = "58c8d27c78d21e728a6bc7b3cc06412e"  # torchvision reference
EMNIST_URLS = [
    "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip",
    "https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip",
]


def md5_file(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def download_with_fallback(urls: Sequence[str], outpath: str, timeout: int = 60) -> None:
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    last_err: Optional[Exception] = None
    for url in urls:
        try:
            print(f"[Download] {url}")
            with urllib.request.urlopen(url, timeout=timeout) as r:
                total = r.length
                tmp = outpath + ".part"
                with open(tmp, "wb") as f:
                    downloaded = 0
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct = 100.0 * downloaded / total
                            print(
                                f"  {downloaded/1e6:7.1f} MB / {total/1e6:7.1f} MB  ({pct:5.1f}%)",
                                end="\r",
                            )
                print()
            os.replace(tmp, outpath)
            print(f"[Saved] {outpath}")
            return
        except Exception as e:  # pragma: no cover
            print(f"[Warn] Failed: {url} ({e})")
            last_err = e
    raise RuntimeError(f"All download URLs failed. Last error: {last_err}")


def ensure_emnist_download(root: str, verify_md5: bool = False) -> str:
    """Download and extract EMNIST gzip.zip. Returns extraction directory."""
    root = os.path.abspath(root)
    raw_dir = os.path.join(root, "emnist_raw")
    zip_path = os.path.join(raw_dir, "gzip.zip")
    os.makedirs(raw_dir, exist_ok=True)

    expected_any = any(
        os.path.exists(os.path.join(raw_dir, f))
        for f in [
            "emnist-byclass-train-images-idx3-ubyte.gz",
            "gzip/emnist-byclass-train-images-idx3-ubyte.gz",
        ]
    )

    if not os.path.exists(zip_path):
        download_with_fallback(EMNIST_URLS, zip_path)

    if verify_md5:
        print("[MD5] Verifying gzip.zip ...")
        h = md5_file(zip_path)
        if h != EMNIST_MD5:
            print(f"[Warn] MD5 mismatch: got {h}, expected {EMNIST_MD5}")
        else:
            print("[MD5] OK")

    if not expected_any:
        print("[Extract] Unzipping gzip.zip ...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(raw_dir)
        print("[Extract] Done")

    return raw_dir


def _find_emnist_gz(raw_dir: str, split: str, train: bool, kind: str) -> str:
    """Locate a gz file inside raw_dir.

    kind: "images" or "labels"
    """
    assert kind in ("images", "labels")
    part = "train" if train else "test"
    suffix = "images-idx3-ubyte.gz" if kind == "images" else "labels-idx1-ubyte.gz"
    name = f"emnist-{split}-{part}-{suffix}"

    candidates = [
        os.path.join(raw_dir, name),
        os.path.join(raw_dir, "gzip", name),
        os.path.join(raw_dir, "emnist-gzip", name),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p

    for root, _, files in os.walk(raw_dir):
        for f in files:
            if f == name:
                return os.path.join(root, f)

    raise FileNotFoundError(f"Could not find {name} under {raw_dir}")


def _read_be_u32(f) -> int:
    return struct.unpack(">I", f.read(4))[0]


def iter_emnist_batches(
    raw_dir: str,
    split: str,
    train: bool,
    batch_size: int = 8192,
    max_items: Optional[int] = None,
    label_allow: Optional[set] = None,
) -> Iterable[Tuple[np.ndarray, Optional[np.ndarray]]]:
    """Yield (images_uint8, labels_uint8 or None) batches from EMNIST gz files."""
    img_path = _find_emnist_gz(raw_dir, split, train=train, kind="images")
    lbl_path = _find_emnist_gz(raw_dir, split, train=train, kind="labels")

    with gzip.open(img_path, "rb") as fi, gzip.open(lbl_path, "rb") as fl:
        magic_i = _read_be_u32(fi)
        if magic_i != 2051:
            raise RuntimeError(f"Bad image magic: {magic_i} in {img_path}")
        n_items = _read_be_u32(fi)
        rows = _read_be_u32(fi)
        cols = _read_be_u32(fi)
        if rows != 28 or cols != 28:
            raise RuntimeError(f"Unexpected image size {rows}x{cols} in {img_path}")

        magic_l = _read_be_u32(fl)
        if magic_l != 2049:
            raise RuntimeError(f"Bad label magic: {magic_l} in {lbl_path}")
        n_lbl = _read_be_u32(fl)
        if n_lbl != n_items:
            raise RuntimeError(f"Label count mismatch: {n_lbl} vs {n_items}")

        total = n_items if max_items is None else min(n_items, int(max_items))

        for start in range(0, total, batch_size):
            bs = min(batch_size, total - start)
            buf_i = fi.read(bs * 28 * 28)
            if len(buf_i) != bs * 28 * 28:
                raise RuntimeError("Unexpected EOF while reading images")
            imgs = np.frombuffer(buf_i, dtype=np.uint8).reshape(bs, 28, 28)

            buf_l = fl.read(bs)
            if len(buf_l) != bs:
                raise RuntimeError("Unexpected EOF while reading labels")
            labels = np.frombuffer(buf_l, dtype=np.uint8)

            if label_allow is not None:
                mask = np.isin(labels, list(label_allow))
                if not np.any(mask):
                    continue
                imgs = imgs[mask]
                labels = labels[mask]

            yield imgs, labels


# -----------------------------------------------------------------------------
# EMNIST -> bitstring distribution p*(x)
# -----------------------------------------------------------------------------


@dataclass
class BuildConfig:
    split: str = "byclass"  # byclass, bymerge, balanced, letters, digits, mnist
    include_test: bool = True
    downsample: int = 4  # 4 => 4x4 => n=16
    threshold: float = 0.20
    fix_orientation: bool = True
    max_items: Optional[int] = None  # None => all
    label_allow: Optional[List[int]] = None
    temperature: float = 1.0
    eps: float = 1e-12
    bit_transform: str = "none"  # none|permute|gf2
    perm_seed: int = 0
    gf2_seed: int = 0


def _orientation_fix(imgs: np.ndarray) -> np.ndarray:
    """Fix EMNIST orientation to be more human-friendly."""
    imgs = np.transpose(imgs, (0, 2, 1))
    imgs = np.flip(imgs, axis=2)
    return imgs


def _avg_pool_28_to_4(imgs_f: np.ndarray) -> np.ndarray:
    """Average pool 28x28 -> 4x4 by 7x7 blocks."""
    bs = imgs_f.shape[0]
    return imgs_f.reshape(bs, 4, 7, 4, 7).mean(axis=(2, 4))


def _bits_to_index(bits: np.ndarray) -> np.ndarray:
    """bits: (B, n) uint8 -> indices (B,) uint16"""
    n = bits.shape[1]
    powers = (1 << np.arange(n, dtype=np.uint32)).astype(np.uint32)
    idx = (bits.astype(np.uint32) * powers[None, :]).sum(axis=1)
    return idx.astype(np.uint16)


def _random_invertible_gf2_matrix(n: int, seed: int) -> np.ndarray:
    """Return a random invertible n x n matrix over GF(2) as uint8."""
    rng = np.random.default_rng(seed)
    while True:
        A = rng.integers(0, 2, size=(n, n), dtype=np.uint8)
        for i in range(n):
            A[i, i] = 1
        M = A.copy()
        rank = 0
        col = 0
        for r in range(n):
            while col < n and M[r:, col].max() == 0:
                col += 1
            if col >= n:
                break
            piv = r + np.argmax(M[r:, col])
            if M[piv, col] == 0:
                continue
            if piv != r:
                M[[r, piv]] = M[[piv, r]]
            for rr in range(r + 1, n):
                if M[rr, col]:
                    M[rr] ^= M[r]
            rank += 1
            col += 1
        if rank == n:
            return A


def _bit_transform_table(n: int, mode: str, perm_seed: int = 0, gf2_seed: int = 0) -> Tuple[np.ndarray, Dict]:
    """Return mapping table map_idx of length 2^n applying an invertible transform."""
    N = 1 << n
    idx = np.arange(N, dtype=np.uint16)
    meta: Dict = {"mode": mode}

    if mode == "none":
        return idx.astype(np.uint16), meta

    if mode == "permute":
        rng = np.random.default_rng(perm_seed)
        perm = rng.permutation(n)
        meta.update({"perm_seed": perm_seed, "perm": perm.tolist()})
        out = np.zeros_like(idx, dtype=np.uint16)
        for new_pos in range(n):
            old_pos = int(perm[new_pos])
            bit = (idx >> old_pos) & 1
            out |= (bit.astype(np.uint16) << new_pos)
        return out, meta

    if mode == "gf2":
        A = _random_invertible_gf2_matrix(n, gf2_seed)
        meta.update({"gf2_seed": gf2_seed, "A": A.astype(int).tolist()})
        row_masks = np.zeros(n, dtype=np.uint16)
        for r in range(n):
            m = 0
            for c in range(n):
                if A[r, c]:
                    m |= 1 << c
            row_masks[r] = np.uint16(m)

        out = np.zeros_like(idx, dtype=np.uint16)
        for r in range(n):
            pc = popcount_uint16((idx & row_masks[r]).astype(np.uint16))
            bit = (pc & 1).astype(np.uint16)
            out |= (bit << r)
        return out, meta

    raise ValueError(f"Unknown bit_transform: {mode}")


def build_p_star_from_emnist(
    data_root: str,
    cfg: BuildConfig,
    cache_dir: str,
) -> Tuple[np.ndarray, Dict]:
    """Build p*(x) over n=downsample^2 bits from EMNIST."""
    assert cfg.downsample in (4,), "This script currently supports downsample=4 (n=16) only."
    n = cfg.downsample * cfg.downsample
    N = 1 << n

    raw_dir = ensure_emnist_download(data_root, verify_md5=False)

    cache_cfg = dataclasses.asdict(cfg)
    cache_cfg["n"] = n
    cache_bytes = json.dumps(cache_cfg, sort_keys=True).encode("utf-8")
    h = hashlib.md5(cache_bytes).hexdigest()[:12]
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"counts_emnist_{h}.npz")

    if os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=True)
        counts = z["counts"].astype(np.int64)
        meta = dict(z["meta"].item())
        print(f"[Cache] Loaded counts from {cache_path}")
    else:
        counts = np.zeros(N, dtype=np.int64)

        label_allow = set(cfg.label_allow) if cfg.label_allow else None

        map_idx, tmeta = _bit_transform_table(
            n=n,
            mode=cfg.bit_transform,
            perm_seed=cfg.perm_seed,
            gf2_seed=cfg.gf2_seed,
        )

        def process_stream(train: bool, max_items: Optional[int]) -> int:
            processed = 0
            for imgs_u8, _labels in iter_emnist_batches(
                raw_dir=raw_dir,
                split=cfg.split,
                train=train,
                batch_size=8192,
                max_items=max_items,
                label_allow=label_allow,
            ):
                if cfg.fix_orientation:
                    imgs_u8 = _orientation_fix(imgs_u8)
                x = imgs_u8.astype(np.float32) / 255.0
                x4 = _avg_pool_28_to_4(x)
                bits = (x4 > cfg.threshold).astype(np.uint8).reshape(-1, n)
                idx = _bits_to_index(bits)
                if cfg.bit_transform != "none":
                    idx = map_idx[idx]
                bc = np.bincount(idx.astype(np.int32), minlength=N)
                counts[:] += bc.astype(np.int64)
                processed += imgs_u8.shape[0]
                if processed % 50000 < imgs_u8.shape[0]:
                    print(f"  processed {processed:,} images", end="\r")
            print()
            return processed

        t0 = time.time()
        n_train = process_stream(train=True, max_items=cfg.max_items)
        n_test = 0
        if cfg.include_test:
            max_test = None
            if cfg.max_items is not None:
                max_test = max(0, int(cfg.max_items) - int(n_train))
            n_test = process_stream(train=False, max_items=max_test)

        t1 = time.time()

        meta = {
            "split": cfg.split,
            "include_test": cfg.include_test,
            "downsample": cfg.downsample,
            "threshold": cfg.threshold,
            "fix_orientation": cfg.fix_orientation,
            "max_items": cfg.max_items,
            "label_allow": cfg.label_allow,
            "bit_transform": tmeta,
            "n": n,
            "N": N,
            "n_train": int(n_train),
            "n_test": int(n_test),
            "total_images": int(n_train + n_test),
            "build_seconds": float(t1 - t0),
        }

        np.savez_compressed(cache_path, counts=counts, meta=meta)
        print(f"[Cache] Saved counts to {cache_path}")

    total = counts.sum()
    if total <= 0:
        raise RuntimeError("No samples counted. Check download / label filter.")

    p_emp = counts.astype(np.float64) / float(total)

    T = float(cfg.temperature)
    if T <= 0:
        raise ValueError("temperature must be > 0")
    gamma = 1.0 / T
    p = np.power(p_emp + float(cfg.eps), gamma)
    p /= p.sum()

    nz = int((counts > 0).sum())
    meta["unique_states"] = nz
    meta["max_prob"] = float(p.max())
    meta["min_nonzero_prob"] = float(p[p > 0].min())

    return p.astype(np.float64), meta


# -----------------------------------------------------------------------------
# Paper-style objects: good set, holdouts, masking, visibility
# -----------------------------------------------------------------------------


@dataclass
class HoldoutConfig:
    good_frac: float = 0.05
    holdout_k: int = 20
    m_train: int = 1000
    holdout_pool: int = 400
    pool_size: int = 4000
    seed: int = 0
    visibility_tries: int = 40
    visibility_min_ratio: float = 2.0  # require q_lin(H)/q_unif(H) >= this


def make_ring_edges(n: int) -> List[Tuple[int, int]]:
    """NN + NNN ring edges as in the paper (i,i+1) and (i,i+2) mod n."""
    edges = []
    for i in range(n):
        edges.append((i, (i + 1) % n))
        edges.append((i, (i + 2) % n))
    return edges


def topk_good_mask(p_star: np.ndarray, support_mask: np.ndarray, frac: float) -> np.ndarray:
    """Good set G: top frac of support by log p*(x) == by p*(x)."""
    idx = np.where(support_mask)[0]
    if idx.size == 0:
        raise RuntimeError("Support empty")
    k = max(1, int(math.ceil(frac * idx.size)))
    vals = p_star[idx]
    top_idx_local = np.argpartition(vals, -k)[-k:]
    good_idx = idx[top_idx_local]
    good = np.zeros_like(p_star, dtype=bool)
    good[good_idx] = True
    return good


def probability_floor_candidates(p_star: np.ndarray, good_mask: np.ndarray, tau0: float, need: int) -> np.ndarray:
    """Select candidate indices in G above a probability floor tau."""
    tau = float(tau0)
    good_idx = np.where(good_mask)[0]
    if good_idx.size < need:
        raise RuntimeError("Good set smaller than holdout_k")

    while True:
        cand = good_idx[p_star[good_idx] >= tau]
        if cand.size >= need:
            return cand
        tau *= 0.5
        if tau < 1e-15:
            return good_idx


def farthest_point_holdout(p_star: np.ndarray, candidates: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Appendix-A Step 3-4: farthest-point sampling with p* tie-break."""
    cand = candidates[np.argsort(-p_star[candidates])]
    if cand.size < k:
        raise ValueError("Not enough candidates")

    selected = [int(cand[0])]
    cand_u16 = cand.astype(np.uint16)
    sel_u16 = np.array(selected, dtype=np.uint16)

    for _ in range(1, k):
        xor = np.bitwise_xor(cand_u16[:, None], sel_u16[None, :]).astype(np.uint16)
        d = popcount_uint16(xor.astype(np.uint32))
        min_d = d.min(axis=1).astype(np.int16, copy=False)
        min_d[np.isin(cand_u16, sel_u16)] = -1
        best = np.max(min_d)
        best_idx = np.where(min_d == best)[0]
        chosen_i = int(best_idx[0])
        selected.append(int(cand[chosen_i]))
        sel_u16 = np.array(selected, dtype=np.uint16)

    return np.array(selected, dtype=np.int32)


def sample_holdout_set(
    p_star: np.ndarray,
    good_mask: np.ndarray,
    hcfg: HoldoutConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """One randomized instance of the Appendix-A heuristic."""
    tau0 = 1.0 / float(max(1, hcfg.m_train))
    cand = probability_floor_candidates(p_star, good_mask, tau0=tau0, need=hcfg.holdout_k)

    cand = cand[np.argsort(-p_star[cand])]
    cand = cand[: max(hcfg.pool_size, hcfg.holdout_pool)]

    if cand.size > hcfg.holdout_pool:
        w = p_star[cand]
        w = w / w.sum()
        sub = rng.choice(cand, size=hcfg.holdout_pool, replace=False, p=w)
    else:
        sub = cand

    H = farthest_point_holdout(p_star, sub, k=hcfg.holdout_k, rng=rng)
    return H


def compute_visibility(
    H: np.ndarray,
    alphas: np.ndarray,
    z_target: np.ndarray,
    n: int,
) -> float:
    """Vis_B(H) = sum_k z_k * 1_hat_H(alpha_k), Definition 1."""
    N = 1 << n
    H_u16 = H.astype(np.uint16)
    alpha_u16 = alphas.astype(np.uint16)
    signs = parity_sign(alpha_u16, H_u16)
    h_hat = (signs.astype(np.float64).sum(axis=0) / float(N))
    return float(np.dot(z_target.astype(np.float64), h_hat))


def select_holdout_visible(
    p_star: np.ndarray,
    good_mask: np.ndarray,
    alphas: np.ndarray,
    n: int,
    hcfg: HoldoutConfig,
    seed: int,
) -> Tuple[np.ndarray, Dict]:
    """Select H using Appendix-A heuristic + visibility check."""
    rng = np.random.default_rng(seed)

    best = None
    best_info: Dict = {}

    N = 1 << n
    q_unif_H = float(hcfg.holdout_k) / float(N)

    def eval_H(H: np.ndarray) -> Tuple[float, float, float, np.ndarray]:
        p_train = p_star.copy()
        p_train[H] = 0.0
        p_train = p_train / p_train.sum()
        p_hat = fwht_np(p_train)
        z = p_hat[alphas]
        vis = compute_visibility(H, alphas, z, n=n)
        q_lin_H = q_unif_H + vis
        ratio = q_lin_H / q_unif_H if q_unif_H > 0 else float("inf")
        return ratio, q_lin_H, vis, z

    ratios = []
    for t in range(max(1, hcfg.visibility_tries)):
        H = sample_holdout_set(p_star, good_mask, hcfg, rng=rng)
        ratio, q_lin_H, vis, z = eval_H(H)
        ratios.append(ratio)
        if best is None or ratio > best_info.get("ratio", -1):
            best = H
            best_info = {
                "ratio": float(ratio),
                "q_lin_H": float(q_lin_H),
                "vis": float(vis),
                "z_target": z.astype(np.float64),
                "trial": int(t),
            }

    best_info["min_ratio_required"] = float(hcfg.visibility_min_ratio)
    best_info["ratios_tried"] = {
        "min": float(np.min(ratios)),
        "median": float(np.median(ratios)),
        "max": float(np.max(ratios)),
    }

    return best.astype(np.int32), best_info


# -----------------------------------------------------------------------------
# Parity-mask sampling (sigma, K)
# -----------------------------------------------------------------------------


def p_mask_from_sigma(sigma: float) -> float:
    return 0.5 * (1.0 - math.exp(-1.0 / (2.0 * sigma * sigma)))


def sample_parity_masks(n: int, K: int, sigma: float, seed: int, force_global_parity: bool = True) -> np.ndarray:
    """Sample K random parity masks α ∈ {0,1}^n as uint16 indices."""
    rng = np.random.default_rng(seed)
    p = p_mask_from_sigma(float(sigma))

    masks: List[int] = []
    seen = set()

    if force_global_parity:
        gp = (1 << n) - 1
        masks.append(gp)
        seen.add(gp)

    while len(masks) < K:
        bits = (rng.random(n) < p).astype(np.uint8)
        m = int(_bits_to_index(bits.reshape(1, -1))[0])
        if m == 0:
            continue
        if m in seen:
            continue
        masks.append(m)
        seen.add(m)

    return np.array(masks, dtype=np.int32)


# -----------------------------------------------------------------------------
# Recovery curves and Q80
# -----------------------------------------------------------------------------


def expected_unique_fraction(q: np.ndarray, H: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Expected fraction of distinct recovered states in H after Q samples."""
    qH = q[H].astype(np.float64)
    Q = Q.astype(np.int64)
    log_term = np.log1p(-qH[None, :])
    never = np.exp(Q[:, None] * log_term)
    rec = 1.0 - never
    return rec.mean(axis=1)


def compute_Q80(q: np.ndarray, H: np.ndarray, thr: float = 0.8, Qmax: int = 200000) -> int:
    """Smallest Q with R(Q) >= thr using monotone binary search."""
    qH = q[H].astype(np.float64)

    def R(Q: int) -> float:
        return float(np.mean(1.0 - np.exp(Q * np.log1p(-qH))))

    if R(0) >= thr:
        return 0
    if R(Qmax) < thr:
        return int(Qmax)

    lo, hi = 0, int(Qmax)
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if R(mid) >= thr:
            hi = mid
        else:
            lo = mid
    return int(hi)


# -----------------------------------------------------------------------------
# Torch models: Ising control and IQP-QCBM
# -----------------------------------------------------------------------------


@dataclass
class TrainConfig:
    n: int = 16
    iqp_depth: int = 3
    include_fields: bool = True
    lr: float = 0.05
    steps: int = 600
    seed: int = 0
    device: str = "cpu"


def make_spin_and_edge_features(n: int, edges: List[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
    """Precompute spins s_i(x) and pair features s_i s_j for all states x."""
    N = 1 << n
    idx = np.arange(N, dtype=np.uint32)
    bits = ((idx[:, None] >> np.arange(n, dtype=np.uint32)[None, :]) & 1).astype(np.int8)
    spins = (1 - 2 * bits).astype(np.float32)

    edge_feat = np.zeros((N, len(edges)), dtype=np.float32)
    for e_i, (i, j) in enumerate(edges):
        edge_feat[:, e_i] = spins[:, i] * spins[:, j]

    return spins, edge_feat


def iqp_distribution_torch(
    theta_edges: "torch.Tensor",
    theta_fields: Optional["torch.Tensor"],
    spins_t: "torch.Tensor",
    edge_feat_t: "torch.Tensor",
) -> "torch.Tensor":
    """Compute IQP Born distribution q(x) exactly via FWHT-based statevector sim."""
    device = edge_feat_t.device
    N = edge_feat_t.shape[0]
    L = theta_edges.shape[0]
    sqrtN = math.sqrt(N)

    state = torch.ones(N, dtype=torch.complex64, device=device) / sqrtN

    for l in range(L):
        phase = edge_feat_t @ theta_edges[l]
        if theta_fields is not None:
            phase = phase + (spins_t @ theta_fields[l])
        state = state * torch.exp(1j * phase.to(torch.complex64))
        state = fwht_torch(state) / sqrtN

    q = (state.real**2 + state.imag**2)
    q = q / q.sum()
    return q


def ising_distribution_torch(
    J: "torch.Tensor",
    edge_feat_t: "torch.Tensor",
) -> "torch.Tensor":
    """Pairwise Ising/Boltzmann model on the same edge set (fields h=0 by default)."""
    logits = edge_feat_t @ J
    logits = logits - logits.max()
    q = torch.exp(logits)
    q = q / q.sum()
    return q


def train_iqp(
    z_target_t: "torch.Tensor",
    alphas: np.ndarray,
    spins_t: "torch.Tensor",
    edge_feat_t: "torch.Tensor",
    tcfg: TrainConfig,
) -> Tuple[np.ndarray, Dict]:
    assert torch is not None

    torch.manual_seed(int(tcfg.seed))

    n = tcfg.n
    edges_E = edge_feat_t.shape[1]
    L = int(tcfg.iqp_depth)

    theta_edges = torch.nn.Parameter(0.01 * torch.randn(L, edges_E, device=tcfg.device))
    theta_fields = None
    if tcfg.include_fields:
        theta_fields = torch.nn.Parameter(0.01 * torch.randn(L, n, device=tcfg.device))

    params = [theta_edges] + ([theta_fields] if theta_fields is not None else [])
    opt = torch.optim.Adam(params, lr=float(tcfg.lr))

    alpha_idx = torch.tensor(alphas.astype(np.int64), device=tcfg.device)

    hist = {"loss": [], "time_sec": []}

    t_start = time.time()
    for step in range(1, int(tcfg.steps) + 1):
        opt.zero_grad(set_to_none=True)
        q = iqp_distribution_torch(theta_edges, theta_fields, spins_t, edge_feat_t)
        q_hat = fwht_torch(q)
        z = q_hat.index_select(0, alpha_idx)
        loss = torch.mean((z - z_target_t) ** 2)
        loss.backward()
        opt.step()

        if step == 1 or step % 50 == 0 or step == tcfg.steps:
            elapsed = time.time() - t_start
            hist["loss"].append(float(loss.detach().cpu().item()))
            hist["time_sec"].append(float(elapsed))
            print(f"[IQP] step {step:4d}/{tcfg.steps}  loss={hist['loss'][-1]:.6f}  t={elapsed:.1f}s")

    with torch.no_grad():
        q = iqp_distribution_torch(theta_edges, theta_fields, spins_t, edge_feat_t)
        q_np = q.detach().cpu().numpy().astype(np.float64)

    info = {
        "train_hist": hist,
        "theta_edges": theta_edges.detach().cpu().numpy().tolist(),
        "theta_fields": theta_fields.detach().cpu().numpy().tolist() if theta_fields is not None else None,
    }

    return q_np, info


def train_ising(
    z_target_t: "torch.Tensor",
    alphas: np.ndarray,
    edge_feat_t: "torch.Tensor",
    tcfg: TrainConfig,
) -> Tuple[np.ndarray, Dict]:
    assert torch is not None

    torch.manual_seed(int(tcfg.seed))
    E = edge_feat_t.shape[1]

    J = torch.nn.Parameter(0.01 * torch.randn(E, device=tcfg.device))
    opt = torch.optim.Adam([J], lr=float(tcfg.lr))

    alpha_idx = torch.tensor(alphas.astype(np.int64), device=tcfg.device)

    hist = {"loss": [], "time_sec": []}
    t_start = time.time()
    for step in range(1, int(tcfg.steps) + 1):
        opt.zero_grad(set_to_none=True)
        q = ising_distribution_torch(J, edge_feat_t)
        q_hat = fwht_torch(q)
        z = q_hat.index_select(0, alpha_idx)
        loss = torch.mean((z - z_target_t) ** 2)
        loss.backward()
        opt.step()

        if step == 1 or step % 50 == 0 or step == tcfg.steps:
            elapsed = time.time() - t_start
            hist["loss"].append(float(loss.detach().cpu().item()))
            hist["time_sec"].append(float(elapsed))
            print(f"[Ising] step {step:4d}/{tcfg.steps}  loss={hist['loss'][-1]:.6f}  t={elapsed:.1f}s")

    with torch.no_grad():
        q = ising_distribution_torch(J, edge_feat_t)
        q_np = q.detach().cpu().numpy().astype(np.float64)

    info = {"train_hist": hist, "J": J.detach().cpu().numpy().tolist()}

    return q_np, info


# -----------------------------------------------------------------------------
# Spectral completion q~ (Eq 14-15)
# -----------------------------------------------------------------------------


def spectral_completion(z_target: np.ndarray, alphas: np.ndarray, n: int) -> Tuple[np.ndarray, Dict]:
    """Compute q~ from enforced moments and simplex projection (clip+renorm)."""
    N = 1 << n
    r_hat = np.zeros(N, dtype=np.float64)
    r_hat[0] = 1.0
    r_hat[alphas] = z_target.astype(np.float64)
    q_lin = fwht_np(r_hat) / float(N)
    q_tilde = np.clip(q_lin, a_min=0.0, a_max=None)
    s = q_tilde.sum()
    if s <= 0:
        q_tilde = np.ones(N, dtype=np.float64) / float(N)
    else:
        q_tilde = q_tilde / s

    q_hat = fwht_np(q_tilde)
    z_post = q_hat[alphas]
    mse = float(np.mean((z_target - z_post) ** 2))

    info = {
        "q_lin_min": float(q_lin.min()),
        "q_lin_max": float(q_lin.max()),
        "clipped_mass": float(np.sum(q_lin < 0)),
        "postproj_mse": mse,
    }

    return q_tilde.astype(np.float64), info


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


def plot_recovery_curves(
    outpath: str,
    title: str,
    Q: np.ndarray,
    curves: Dict[str, np.ndarray],
    Q80s: Dict[str, int],
) -> None:
    set_style(base=9)
    fig, ax = plt.subplots(1, 1, figsize=fig_size("col", 2.6), constrained_layout=True)

    order = ["Target p*", "IQP-QCBM", "Ising control", "Spectral completion q~", "Uniform"]
    colors = {
        "Target p*": "black",
        "IQP-QCBM": "#d62728",
        "Ising control": "#1f77b4",
        "Spectral completion q~": "#555555",
        "Uniform": "#888888",
    }
    styles = {
        "Target p*": "-",
        "IQP-QCBM": "-",
        "Ising control": ":",
        "Spectral completion q~": "-.",
        "Uniform": "--",
    }

    for name in order:
        if name not in curves:
            continue
        ax.plot(
            Q,
            curves[name],
            label=f"{name} (Q80={Q80s.get(name,'?')})",
            color=colors.get(name),
            linestyle=styles.get(name),
            linewidth=2,
        )

    ax.set_title(title)
    ax.set_xlabel(r"$Q$ samples from model")
    ax.set_ylabel(r"Recovery $R(Q)$")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(0, int(Q.max()))
    ax.axhline(0.8, color="#aaaaaa", linestyle=":", linewidth=1)
    ax.legend(loc="center right", frameon=False, ncol=2)

    fig.savefig(outpath)
    plt.close(fig)
    print(f"[Saved] {outpath}")


# -----------------------------------------------------------------------------
# Experiment-1 style plots (overlay + paper heatmaps)
# -----------------------------------------------------------------------------

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D


def _ising_color(rank: int, m: int):
    """Dark blue for best (rank=0), fading to near-white for worst."""
    if m <= 1:
        t = 0.95
        r = 0.0
    else:
        r = rank / (m - 1)
        t = 0.95 - 0.90 * r
    color = plt.cm.Blues(t)
    alpha = 0.90 - 0.50 * r
    alpha = max(0.35, float(alpha))
    return color, alpha


def plot_heatmap_paperstyle(
    mat: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    cbar_label: str,
    outpath: str,
    *,
    log10: bool = False,
    fmt: str = "{:.1f}",
    cmap=None,
    mode: str = "col",
) -> None:
    """Heatmap similar to hero_full_validation plotting (no title)."""
    set_style(base=8)
    figsize = fig_size("col", 2.6)

    data = mat.astype(np.float64)
    if log10:
        data = np.log10(np.maximum(data, 1e-12))

    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
    im = ax.imshow(data, aspect="auto", cmap=cmap)

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel(r"$K$")
    ax.set_ylabel(r"$\sigma$")

    finite_vals = data[np.isfinite(data)]
    thresh = np.nanmedian(finite_vals) if finite_vals.size else 0.0

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v_raw = mat[i, j]
            v_plot = data[i, j]
            s = "" if not np.isfinite(v_raw) else fmt.format(v_raw)
            color = "white" if (np.isfinite(v_plot) and v_plot > thresh) else "black"
            ax.text(j, i, s, ha="center", va="center", fontsize=7.5, color=color)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    fig.savefig(outpath)
    plt.close(fig)
    print(f"[Saved] {outpath}")


def plot_overlay_best_iqp_vs_all_ising(
    *,
    outpath: str,
    Q: np.ndarray,
    Qmax: int,
    p_star: np.ndarray,
    holdout: np.ndarray,
    best_sigma: float,
    best_K: int,
    q_iqp_best: np.ndarray,
    q_spec_best: np.ndarray,
    ising_rows: List[Tuple[float, int, float, float, np.ndarray]],
) -> None:
    """Overlay plot: Target (black), best IQP (red), spec (gray), all Ising (blue->white), uniform (gray)."""

    set_style(base=8)

    SPEC_COLOR = "#555555"
    SPEC_LS = (0, (7, 2, 1.2, 2))

    N = p_star.size
    q_unif = np.ones(N, dtype=np.float64) / float(N)

    y_star = expected_unique_fraction(p_star, holdout, Q)
    y_unif = expected_unique_fraction(q_unif, holdout, Q)
    y_iqp = expected_unique_fraction(q_iqp_best, holdout, Q)
    y_spec = expected_unique_fraction(q_spec_best, holdout, Q)

    fig, ax = plt.subplots(figsize=fig_size("col", 2.6), constrained_layout=True)

    # many Ising curves (underneath)
    m = len(ising_rows)
    for idx, (sigma, K, _Q80c, _qHr, q_cl) in enumerate(ising_rows):
        y_cl = expected_unique_fraction(q_cl, holdout, Q)
        color, alpha = _ising_color(idx, m)
        ax.plot(Q, y_cl, color=color, alpha=alpha, linewidth=1.2, zorder=2)

    # uniform + spectral + target + best IQP
    ax.plot(Q, y_unif, color="#888888", linestyle="--", linewidth=1.5, alpha=0.9, zorder=1)
    ax.plot(Q, y_spec, color=SPEC_COLOR, linestyle=SPEC_LS, linewidth=1.9, zorder=4)
    ax.plot(Q, y_star, color="black", linewidth=1.9, zorder=5)
    ax.plot(Q, y_iqp, color="#d62728", linewidth=2.4, zorder=6)

    ax.axhline(1.0, color="#888888", linestyle=":", alpha=0.7)

    ax.set_xlim(0, int(Qmax))
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel(r"$Q$ samples from model")
    ax.set_ylabel(r"Recovery $R(Q)$")

    legend_handles = [
        Line2D([0], [0], color="black", lw=1.9, label=r"Target $p^*$"),
        Line2D(
            [0],
            [0],
            color="#d62728",
            lw=2.4,
            label=fr"Best IQP-QCBM, $\sigma$={best_sigma:g}, $K$={best_K:d}",
        ),
        Line2D([0], [0], color=SPEC_COLOR, lw=1.9, ls=SPEC_LS, label=r"Spectral completion $\~q$"),
        Line2D([0], [0], color=plt.cm.Blues(0.90), lw=1.6, label="Ising controls"),
        Line2D([0], [0], color="#888888", lw=1.5, ls="--", label="Uniform"),
    ]

    # --- FIX: smaller, tighter legend that sits in the white region (right side) ---
    ax.legend(
        handles=legend_handles,
        loc="center right",
        bbox_to_anchor=(0.985, 0.53),
        fontsize=7.0,
        frameon=True,
        framealpha=0.90,
        facecolor="white",
        edgecolor="none",
        handlelength=1.6,
        labelspacing=0.25,
        borderpad=0.25,
        handletextpad=0.5,
        borderaxespad=0.2,
    )

    fig.savefig(outpath)
    plt.close(fig)
    print(f"[Saved] {outpath}")


def plot_heatmap(
    outpath: str,
    mat: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    title: str,
    cbar_label: str,
    log10: bool = False,
    fmt: str = "{:.2f}",
) -> None:
    set_style(base=9)
    fig, ax = plt.subplots(1, 1, figsize=fig_size("col", 2.6), constrained_layout=True)

    data = mat.copy().astype(np.float64)
    if log10:
        data = np.log10(np.maximum(data, 1e-12))

    im = ax.imshow(data, aspect="auto")
    ax.set_title(title)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(
                j,
                i,
                fmt.format(mat[i, j]),
                ha="center",
                va="center",
                fontsize=8,
                color="white" if data[i, j] > np.nanmean(data) else "black",
            )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    fig.savefig(outpath)
    plt.close(fig)
    print(f"[Saved] {outpath}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="EMNIST (real data) holdout-recovery experiment with Experiment_1-style plots")

    # Data / build
    ap.add_argument("--data-root", type=str, default=str(ROOT / "data" / "emnist"),
                    help="Where to download/cache EMNIST")
    ap.add_argument("--cache-dir", type=str, default=str(ROOT / "data" / "emnist" / "cache"),
                    help="Cache directory for counts")
    ap.add_argument(
        "--split",
        type=str,
        default="byclass",
        choices=["byclass", "bymerge", "balanced", "letters", "digits", "mnist"],
        help="EMNIST split",
    )
    ap.add_argument("--include-test", action="store_true", help="Include EMNIST test split in p*(x)")
    ap.add_argument("--downsample", type=int, default=4, help="Downsample factor (28->4 only supported)")
    ap.add_argument("--threshold", type=float, default=0.20, help="Binarization threshold after pooling")
    ap.add_argument("--no-fix-orientation", action="store_true", help="Do not rotate/flip EMNIST")
    ap.add_argument("--max-items", type=int, default=None, help="Optional cap on total images processed")
    ap.add_argument("--label-allow", type=str, default=None, help="Comma-separated EMNIST label ids to include")
    ap.add_argument("--temperature", type=float, default=1.0, help="Temperature transform on p_emp")
    ap.add_argument("--eps", type=float, default=1e-12, help="Smoothing eps for temperature transform")

    ap.add_argument(
        "--bit-transform",
        type=str,
        default="none",
        choices=["none", "permute", "gf2"],
        help="Invertible bit transform",
    )
    ap.add_argument("--perm-seed", type=int, default=0)
    ap.add_argument("--gf2-seed", type=int, default=0)

    ap.add_argument("--even-parity-only", action="store_true", help="Restrict p* to even parity sector (paper-like)")

    # Holdouts
    ap.add_argument("--good-frac", type=float, default=0.05, help="Good set fraction (top by log p*)")
    ap.add_argument("--holdout-k", type=int, default=20, help="|H|")
    ap.add_argument("--m-train", type=int, default=1000, help="Training sample budget m")
    ap.add_argument("--holdout-pool", type=int, default=400)
    ap.add_argument("--pool-size", type=int, default=4000)
    ap.add_argument("--holdout-visibility-tries", type=int, default=40)
    ap.add_argument("--holdout-visibility-min-ratio", type=float, default=2.0)

    # Holdout reference band
    ap.add_argument("--holdout-sigma", type=float, default=1.0)
    ap.add_argument("--holdout-K", type=int, default=None)

    # Sweep grid
    ap.add_argument("--sigmas", type=str, default="0.5,1.0,2.0,3.0", help="Comma-separated sigmas")
    ap.add_argument("--Ks", type=str, default="128,256,512", help="Comma-separated Ks")
    ap.add_argument("--sigma", type=float, default=None, help="Alias: run a single sigma")
    ap.add_argument("--K", type=int, default=None, help="Alias: run a single K")

    # Training
    ap.add_argument("--iqp-depth", type=int, default=3)
    ap.add_argument("--include-fields", action="store_true")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default=None, help="cpu/cuda (default: auto)")

    # Evaluation
    ap.add_argument("--Qmax", type=int, default=10000)
    ap.add_argument("--Q80-thr", type=float, default=0.8)
    ap.add_argument("--Q80-max", type=int, default=200000)

    # Output
    ap.add_argument("--outdir", type=str, default=str(ROOT / "outputs" / "exp06_emnist"))

    # Skip flags
    ap.add_argument("--no-iqp", action="store_true")
    ap.add_argument("--no-ising", action="store_true")

    args = ap.parse_args()

    if torch is None:
        raise RuntimeError(f"PyTorch is required. Import error: {_TORCH_IMPORT_ERROR}")

    set_style(base=8)
    outdir = ensure_outdir(args.outdir)

    # Colormaps for heatmaps
    cmap_redblack = LinearSegmentedColormap.from_list("RedBlack", ["#FF0000", "#000000"])
    cmap_blueblack = LinearSegmentedColormap.from_list("BlueBlack", ["#DCEBFA", "#1F77B4", "#000000"])

    # ------------------------------
    # Build p*(x) from EMNIST
    # ------------------------------

    label_allow = None
    if args.label_allow:
        label_allow = [int(x.strip()) for x in args.label_allow.split(",") if x.strip()]

    bcfg = BuildConfig(
        split=args.split,
        include_test=bool(args.include_test),
        downsample=int(args.downsample),
        threshold=float(args.threshold),
        fix_orientation=not bool(args.no_fix_orientation),
        max_items=args.max_items,
        label_allow=label_allow,
        temperature=float(args.temperature),
        eps=float(args.eps),
        bit_transform=str(args.bit_transform),
        perm_seed=int(args.perm_seed),
        gf2_seed=int(args.gf2_seed),
    )

    p_star, meta = build_p_star_from_emnist(
        data_root=args.data_root,
        cfg=bcfg,
        cache_dir=args.cache_dir,
    )

    n = int(meta["n"])
    N = int(meta["N"])

    # Optional even-parity restriction
    if args.even_parity_only:
        idx = np.arange(N, dtype=np.uint32)
        bits = ((idx[:, None] >> np.arange(n, dtype=np.uint32)[None, :]) & 1).astype(np.uint8)
        parity = (bits.sum(axis=1) & 1)  # 0 even
        even_mask = (parity == 0)
        p_star = p_star * even_mask.astype(np.float64)
        p_star = p_star / p_star.sum()
        meta["even_parity_only"] = True
        meta["even_parity_mass"] = float(even_mask.mean())
    else:
        meta["even_parity_only"] = False

    support = p_star > 0

    # ------------------------------
    # Parse sweep grid
    # ------------------------------

    sigmas = [float(x) for x in args.sigmas.split(",") if x.strip()]
    Ks = [int(x) for x in args.Ks.split(",") if x.strip()]

    if args.sigma is not None:
        sigmas = [float(args.sigma)]
    if args.K is not None:
        Ks = [int(args.K)]
    if not sigmas or not Ks:
        raise ValueError("sigmas and Ks must be non-empty")

    holdout_sigma = float(args.holdout_sigma)
    holdout_K = int(args.holdout_K) if args.holdout_K is not None else int(max(Ks))

    # ------------------------------
    # Good set G and holdout set H (visibility-aware, using reference band)
    # ------------------------------

    good_mask = topk_good_mask(p_star, support_mask=support, frac=float(args.good_frac))

    hcfg = HoldoutConfig(
        good_frac=float(args.good_frac),
        holdout_k=int(args.holdout_k),
        m_train=int(args.m_train),
        holdout_pool=int(args.holdout_pool),
        pool_size=int(args.pool_size),
        seed=int(args.seed),
        visibility_tries=int(args.holdout_visibility_tries),
        visibility_min_ratio=float(args.holdout_visibility_min_ratio),
    )

    alphas_holdout = sample_parity_masks(
        n=n,
        K=holdout_K,
        sigma=holdout_sigma,
        seed=int(args.seed),
        force_global_parity=True,
    )

    H, H_info = select_holdout_visible(
        p_star=p_star,
        good_mask=good_mask,
        alphas=alphas_holdout,
        n=n,
        hcfg=hcfg,
        seed=int(args.seed) + 111,
    )

    p_train = p_star.copy()
    p_train[H] = 0.0
    p_train = p_train / p_train.sum()

    p_train_hat = fwht_np(p_train)

    z_hold = p_train_hat[alphas_holdout]
    Vis = compute_visibility(H, alphas_holdout, z_hold, n=n)
    q_unif_H = float(H.size) / float(N)
    q_lin_H = q_unif_H + Vis

    print("[Holdout] |H|=", H.size)
    print("[Holdout] reference band: sigma=", holdout_sigma, "K=", holdout_K)
    print("[Holdout] q_unif(H)=", q_unif_H)
    print("[Holdout] Vis_B(H)=", Vis)
    print("[Holdout] q_lin(H)=", q_lin_H, " ratio=", (q_lin_H / q_unif_H if q_unif_H > 0 else float("inf")))

    # ------------------------------
    # Torch features
    # ------------------------------

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = str(args.device)

    edges = make_ring_edges(n)
    spins_np, edge_feat_np = make_spin_and_edge_features(n=n, edges=edges)
    spins_t = torch.tensor(spins_np, dtype=torch.float32, device=device)
    edge_feat_t = torch.tensor(edge_feat_np, dtype=torch.float32, device=device)

    tcfg = TrainConfig(
        n=n,
        iqp_depth=int(args.iqp_depth),
        include_fields=bool(args.include_fields),
        lr=float(args.lr),
        steps=int(args.steps),
        seed=int(args.seed),
        device=device,
    )

    # ------------------------------
    # Sweep over (sigma,K)
    # ------------------------------

    results: List[Dict] = []

    for s in sigmas:
        for K in Ks:
            print(f"[Sweep] sigma={s} K={K}")
            alphas = sample_parity_masks(n=n, K=int(K), sigma=float(s), seed=int(args.seed), force_global_parity=True)
            z_target = p_train_hat[alphas].astype(np.float64)

            q_spec, spec_info = spectral_completion(z_target=z_target, alphas=alphas, n=n)
            qH_spec = float(q_spec[H].sum())
            qH_ratio_spec = qH_spec / q_unif_H
            Q80_spec = compute_Q80(q_spec, H, thr=float(args.Q80_thr), Qmax=int(args.Q80_max))

            z_target_t = torch.tensor(z_target.astype(np.float32), device=device)

            row: Dict = {
                "sigma": float(s),
                "K": int(K),
                "alphas": alphas,
                "q_spec": q_spec,
                "spec_info": spec_info,
                "qH_ratio_spec": float(qH_ratio_spec),
                "Q80_spec": int(Q80_spec),
            }

            if not args.no_ising:
                q_cl, info_cl = train_ising(
                    z_target_t=z_target_t,
                    alphas=alphas,
                    edge_feat_t=edge_feat_t,
                    tcfg=tcfg,
                )
                qH = float(q_cl[H].sum())
                row.update(
                    {
                        "q_class": q_cl,
                        "class_info": info_cl,
                        "qH_ratio_class": float(qH / q_unif_H),
                        "Q80_class": int(compute_Q80(q_cl, H, thr=float(args.Q80_thr), Qmax=int(args.Q80_max))),
                    }
                )

            if not args.no_iqp:
                q_iqp, info_iqp = train_iqp(
                    z_target_t=z_target_t,
                    alphas=alphas,
                    spins_t=spins_t,
                    edge_feat_t=edge_feat_t,
                    tcfg=tcfg,
                )
                qHq = float(q_iqp[H].sum())
                row.update(
                    {
                        "q_iqp": q_iqp,
                        "iqp_info": info_iqp,
                        "qH_ratio_iqp": float(qHq / q_unif_H),
                        "Q80_iqp": int(compute_Q80(q_iqp, H, thr=float(args.Q80_thr), Qmax=int(args.Q80_max))),
                    }
                )

            results.append(row)

    # ------------------------------
    # Build heatmap matrices
    # ------------------------------

    s2i = {float(s): i for i, s in enumerate(sigmas)}
    k2j = {int(k): j for j, k in enumerate(Ks)}

    def mat_from(key: str) -> np.ndarray:
        mat = np.full((len(sigmas), len(Ks)), np.nan, dtype=np.float64)
        for r in results:
            i = s2i[float(r["sigma"])]
            j = k2j[int(r["K"])]
            if key in r:
                mat[i, j] = float(r[key])
        return mat

    mat_ratio_iqp = mat_from("qH_ratio_iqp")
    mat_Q80_iqp = mat_from("Q80_iqp")
    mat_ratio_class = mat_from("qH_ratio_class")
    mat_Q80_class = mat_from("Q80_class")

    row_labels = [str(s) for s in sigmas]
    col_labels = [str(k) for k in Ks]

    if not args.no_iqp:
        plot_heatmap_paperstyle(
            mat_ratio_iqp,
            row_labels=row_labels,
            col_labels=col_labels,
            cbar_label=r"$q_\theta(H)/q_{\mathrm{unif}}(H)$",
            outpath=os.path.join(outdir, "fig2a_emnist_qH_ratio_iqp.pdf"),
            log10=False,
            fmt="{:.1f}",
            cmap="Reds",
            mode="col",
        )
        plot_heatmap_paperstyle(
            mat_Q80_iqp,
            row_labels=row_labels,
            col_labels=col_labels,
            cbar_label=r"$\log_{10} Q_{80}$",
            outpath=os.path.join(outdir, "fig2c_emnist_Q80_iqp.pdf"),
            log10=True,
            fmt="{:.0f}",
            cmap=cmap_redblack,
            mode="col",
        )

    if not args.no_ising:
        plot_heatmap_paperstyle(
            mat_ratio_class,
            row_labels=row_labels,
            col_labels=col_labels,
            cbar_label=r"$q_{\mathrm{cl}}(H)/q_{\mathrm{unif}}(H)$",
            outpath=os.path.join(outdir, "fig2a_emnist_qH_ratio_class.pdf"),
            log10=False,
            fmt="{:.1f}",
            cmap="Blues",
            mode="col",
        )
        plot_heatmap_paperstyle(
            mat_Q80_class,
            row_labels=row_labels,
            col_labels=col_labels,
            cbar_label=r"$\log_{10} Q_{80}$",
            outpath=os.path.join(outdir, "fig2c_emnist_Q80_class.pdf"),
            log10=True,
            fmt="{:.0f}",
            cmap=cmap_blueblack,
            mode="col",
        )

    # ------------------------------
    # Pick best IQP and plot overlay
    # ------------------------------

    best_sigma = None
    best_K = None
    best_row = None

    if not args.no_iqp:
        Qcap = int(args.Q80_max)
        candidates = [r for r in results if ("Q80_iqp" in r and int(r["Q80_iqp"]) < Qcap)]
        if candidates:
            best_row = min(candidates, key=lambda r: int(r["Q80_iqp"]))
        else:
            best_row = max([r for r in results if "qH_ratio_iqp" in r], key=lambda r: float(r["qH_ratio_iqp"]))

        best_sigma = float(best_row["sigma"])
        best_K = int(best_row["K"])

        ising_rows: List[Tuple[float, int, float, float, np.ndarray]] = []
        if not args.no_ising:
            for r in results:
                if "q_class" not in r:
                    continue
                sigma = float(r["sigma"])
                K = int(r["K"])
                Q80c = float(r.get("Q80_class", float("inf")))
                qHr = float(r.get("qH_ratio_class", float("nan")))
                ising_rows.append((sigma, K, Q80c, qHr, r["q_class"]))

            def ising_key(tup):
                _sigma, _K, Q80c, qHr, _q = tup
                finite = Q80c < Qcap
                return (0 if finite else 1, Q80c if finite else 1e99, -qHr)

            ising_rows.sort(key=ising_key)

        Q = q_grid(Qmax=int(args.Qmax))

        plot_overlay_best_iqp_vs_all_ising(
            outpath=os.path.join(outdir, "fig1_recovery_emnist_overlay.pdf"),
            Q=Q,
            Qmax=int(args.Qmax),
            p_star=p_star,
            holdout=H,
            best_sigma=best_sigma,
            best_K=best_K,
            q_iqp_best=best_row["q_iqp"],
            q_spec_best=best_row["q_spec"],
            ising_rows=ising_rows,
        )

    # ------------------------------
    # Save a compact summary.json (no full q arrays)
    # ------------------------------

    sweep_scalars = []
    for r in results:
        entry = {
            "sigma": float(r["sigma"]),
            "K": int(r["K"]),
            "qH_ratio_iqp": float(r.get("qH_ratio_iqp", float("nan"))),
            "Q80_iqp": int(r.get("Q80_iqp", int(args.Q80_max))),
            "qH_ratio_class": float(r.get("qH_ratio_class", float("nan"))),
            "Q80_class": int(r.get("Q80_class", int(args.Q80_max))),
            "qH_ratio_spec": float(r.get("qH_ratio_spec", float("nan"))),
            "Q80_spec": int(r.get("Q80_spec", int(args.Q80_max))),
        }
        sweep_scalars.append(entry)

    summary = {
        "meta": meta,
        "build_config": dataclasses.asdict(bcfg),
        "holdout_config": dataclasses.asdict(hcfg),
        "train_config": dataclasses.asdict(tcfg),
        "sweep_grid": {"sigmas": sigmas, "Ks": Ks},
        "holdout_ref_band": {"sigma": holdout_sigma, "K": holdout_K},
        "holdout": H.tolist(),
        "holdout_info": {
            **{
                k: (float(v) if isinstance(v, (np.floating, float)) else v)
                for k, v in H_info.items()
                if k != "z_target"
            },
            "VisB_H": float(Vis),
            "q_unif_H": float(q_unif_H),
            "q_lin_H": float(q_lin_H),
            "q_lin_ratio": float(q_lin_H / q_unif_H) if q_unif_H > 0 else float("inf"),
        },
        "matrices": {
            "qH_ratio_iqp": mat_ratio_iqp.tolist(),
            "Q80_iqp": mat_Q80_iqp.tolist(),
            "qH_ratio_class": mat_ratio_class.tolist(),
            "Q80_class": mat_Q80_class.tolist(),
        },
        "sweep_results": sweep_scalars,
        "best_iqp": {"sigma": best_sigma, "K": best_K} if best_sigma is not None else None,
    }

    with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[Saved] {os.path.join(outdir, 'summary.json')}")


if __name__ == "__main__":
    main()
