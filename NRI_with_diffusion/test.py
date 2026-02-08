#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def load_probs(path: Path) -> np.ndarray:
    """Load probs/logits from .npz (key 'probs') or .npy. Return [S,E,K]."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix == ".npz":
        d = np.load(path, allow_pickle=True)
        if "probs" not in d.files:
            raise ValueError(f"{path} missing key 'probs'. keys={d.files}")
        arr = d["probs"]
        # 兼容 object array（list of arrays）
        if isinstance(arr, np.ndarray) and arr.dtype == object:
            arr = np.concatenate(list(arr), axis=0)
        return arr

    if path.suffix == ".npy":
        return np.load(path, allow_pickle=True)

    raise ValueError(f"Unsupported file type: {path.suffix} (expect .npz or .npy)")


def softmax_stable(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x = np.asarray(x)
    x_max = np.nanmax(x, axis=axis, keepdims=True)
    ex = np.exp(x - x_max)
    denom = np.nansum(ex, axis=axis, keepdims=True)
    return ex / denom


def ensure_probabilities(arr: np.ndarray) -> np.ndarray:
    """
    如果 arr 看起来已经是概率（∈[0,1] 且 sum≈1），直接返回（并轻微归一化）。
    否则视作 logits/log-probs，做 softmax 得到概率。
    """
    arr = np.asarray(arr)

    # 处理非有限值：先保留，后面计算时会过滤；softmax 前先把 nan/inf 替换成很小的数避免爆炸
    if not np.isfinite(arr).all():
        arr = np.where(np.isfinite(arr), arr, -1e9)

    s = np.sum(arr, axis=-1)
    in01 = (np.min(arr) >= -1e-6) and (np.max(arr) <= 1.0 + 1e-6)
    sum1 = (np.min(s) >= 1.0 - 1e-3) and (np.max(s) <= 1.0 + 1e-3)

    if in01 and sum1:
        # 轻微修正数值误差
        arr = np.clip(arr, 0.0, 1.0)
        arr = arr / np.sum(arr, axis=-1, keepdims=True)
        return arr

    # 否则当成 logits / log-probs 处理
    return softmax_stable(arr, axis=-1)


def compute_confidence_from_probs(probs: np.ndarray) -> np.ndarray:
    """
    你的原定义（适用于二分类）：
      p = probs[...,1]
      c = |p-0.5| + 0.5  ∈ [0.5, 1]
    如果 K!=2，则退化为更通用的：c = max_k probs[...,k]
    """
    K = probs.shape[-1]
    if K == 2:
        p = probs[..., 1]
        c = np.abs(p - 0.5) + 0.5
    else:
        c = np.max(probs, axis=-1)  # 通用置信度
    return c


def ecdf_ccdf(x: np.ndarray):
    """Return sorted x and CCDF = 1 - ECDF."""
    x = np.asarray(x).reshape(-1)
    x = x[np.isfinite(x)]
    x = np.sort(x)
    n = x.size
    if n == 0:
        raise ValueError("No valid finite samples found after filtering.")
    y = np.arange(1, n + 1) / n
    return x, 1 - y


def get_npz_path(base_dir: Path, subfolder: str, filename: str) -> Path:
    return base_dir / subfolder / "results" / filename


def parse_title_from_base_dir(base_dir: Path) -> str:
    """
    从最后一级目录名解析标题：
    例：bn_ns_15r1_draw2  ->  BN_NS
      - 第一个 XX：第一个下划线前，转大写
      - 第二个 XX：第一个到第二个下划线之间，转大写
    """
    name = base_dir.name
    parts = name.split("_")
    if len(parts) < 2:
        raise ValueError(f"Cannot parse title from base_dir name: '{name}' (need at least two '_' parts)")
    xx1 = parts[0].upper()
    xx2 = parts[1].upper()
    return f"{xx1}_{xx2}"


def main():
    parser = argparse.ArgumentParser(
        description="Plot CCDF of confidence from posterior_test.npz in no_prior/prior/diff_prior_round1."
    )
    parser.add_argument("base_dir", type=str,
                        help="Base model directory, e.g. /.../models/bn_ns_15r1_draw2")
    parser.add_argument("--file", type=str, default="posterior_test.npz",
                        help="File name under each subfolder's results/ (default: posterior_test.npz)")
    parser.add_argument("--out", type=str, default=None,
                        help="Output figure path. Default: <base_dir>/probs_cdf.png")
    parser.add_argument("--dpi", type=int, default=300,
                        help="Output dpi (default: 300)")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).expanduser().resolve()
    title = parse_title_from_base_dir(base_dir)

    # 默认输出到 base_dir 下
    out_path = Path(args.out).expanduser().resolve() if args.out else (base_dir / "probs_cdf.png")

    items = [
        ("Uniform", "no_prior"),
        ("Fixed", "prior"),
        ("Diff-prior", "diff_prior_round1"),
    ]

    curves = []
    for label, sub in items:
        p = get_npz_path(base_dir, sub, args.file)
        raw = load_probs(p)                 # 可能是 probs，也可能是 logits
        probs = ensure_probabilities(raw)   # ✅ 强制 softmax/归一化到概率

        conf = compute_confidence_from_probs(probs)  # [S,E] -> confidence
        x, y = ecdf_ccdf(conf)
        curves.append((label, x, y))

    plt.figure(figsize=(7, 5))
    for label, x, y in curves:
        plt.plot(x, y, label=label)

    plt.title(title)
    plt.xlabel("Confidence c")
    plt.ylabel("Complementary CDF")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(out_path, dpi=args.dpi)
    print(f"[OK] Saved figure to: {out_path}")


if __name__ == "__main__":
    main()
