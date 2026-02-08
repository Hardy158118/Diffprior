#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# =========================
# 1) 根目录与固定配置
# =========================
MODELS_ROOT = Path(
    "/home/shaoqi/code2025/Benchmarking-Structural-Inference-Methods-for-Interacting-Dynamical-Systems/"
    "src/models/NRI/models"
).resolve()

DRAW_SUFFIX = "15r1_draw2"
POST_FILE = "posterior_test.npz"

CURVE_SPECS = [
    ("Uniform",    "no_prior"),
    ("Fixed",      "prior"),
    ("Diff-prior", "diff_prior_round1"),
]

COL_MODELS = ["bn", "fw", "grn", "vn", "crna"]
ROW_TYPES  = ["sp", "ns"]   # 第一行 sp，第二行 ns


# =========================
# 2) 读取 + softmax/归一化
# =========================
def load_probs(path: Path) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix == ".npz":
        d = np.load(path, allow_pickle=True)
        if "probs" not in d.files:
            raise ValueError(f"{path} missing key 'probs'. keys={d.files}")
        arr = d["probs"]
        if isinstance(arr, np.ndarray) and arr.dtype == object:
            arr = np.concatenate(list(arr), axis=0)
        return np.asarray(arr)

    if path.suffix == ".npy":
        return np.asarray(np.load(path, allow_pickle=True))

    raise ValueError(f"Unsupported file type: {path.suffix} (expect .npz or .npy)")


def softmax_stable(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x)
    x = np.where(np.isfinite(x), x, -1e9)
    m = np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x - m)
    denom = np.sum(ex, axis=axis, keepdims=True)
    return ex / np.clip(denom, 1e-30, None)


def ensure_probabilities(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    s = np.sum(arr, axis=-1)
    looks_like_prob = (
        np.nanmin(arr) >= -1e-6 and np.nanmax(arr) <= 1.0 + 1e-6
        and np.nanmin(s) >= 1.0 - 1e-3 and np.nanmax(s) <= 1.0 + 1e-3
    )
    if looks_like_prob:
        arr = np.clip(arr, 0.0, 1.0)
        arr = arr / np.clip(np.sum(arr, axis=-1, keepdims=True), 1e-30, None)
        return arr
    return softmax_stable(arr, axis=-1)


# =========================
# 3) confidence + CCDF
# =========================
def confidence_from_probs(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs)
    K = probs.shape[-1]
    if K == 2:
        p = probs[..., 1]
        return np.abs(p - 0.5) + 0.5
    return np.max(probs, axis=-1)


def ecdf_ccdf(x: np.ndarray):
    x = np.asarray(x).reshape(-1)
    x = x[np.isfinite(x)]
    x = np.sort(x)
    n = x.size
    if n == 0:
        raise ValueError("No valid finite samples found after filtering.")
    y = np.arange(1, n + 1) / n
    return x, 1 - y


def get_base_dir(model: str, typ: str) -> Path:
    return MODELS_ROOT / f"{model}_{typ}_{DRAW_SUFFIX}"


def get_npz_path(base_dir: Path, subfolder: str) -> Path:
    return base_dir / subfolder / "results" / POST_FILE


def make_title(model: str, typ: str) -> str:
    return f"{typ.upper()}_{model.upper()}_15"


# =========================
# 4) 主流程：2×5 拼图
# =========================
def main():
    plt.rcParams.update({
        "font.size": 16,
        "axes.labelsize": 17,
        "axes.labelweight": "normal",
        "axes.titlesize": 17,
        "axes.titleweight": "normal",
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 17,
        "axes.linewidth": 1.3,
    })

    if not MODELS_ROOT.exists():
        raise FileNotFoundError(f"MODELS_ROOT does not exist: {MODELS_ROOT}")

    fig, axes = plt.subplots(
        nrows=2, ncols=5,
        figsize=(22, 7.2),
        sharex=False, sharey=False
    )

    # ✅ 左边留白减小、右边留白增大（避免右侧刻度被裁）
    # ✅ top 调高一点让子图靠近 legend
    fig.subplots_adjust(
        left=0.040, right=0.998,
        bottom=0.09, top=0.85,
        wspace=0.16, hspace=0.48
    )

    legend_handles = None
    legend_labels = None

    X_MIN, X_MAX = 0.75, 1.002
    Y_MIN, Y_MAX = 0.50, 1.002

    x_ticks = [0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
    y_ticks = [0.50, 0.60, 0.70, 0.80, 0.90, 1.00]

    for r, typ in enumerate(ROW_TYPES):
        for c, model in enumerate(COL_MODELS):
            ax = axes[r, c]
            base_dir = get_base_dir(model, typ)

            ax.set_title(make_title(model, typ), pad=8)

            plotted_any = False
            if base_dir.exists():
                for label, sub in CURVE_SPECS:
                    f = get_npz_path(base_dir, sub)
                    if not f.exists():
                        continue

                    raw = load_probs(f)
                    probs = ensure_probabilities(raw)
                    conf = confidence_from_probs(probs)
                    x, y = ecdf_ccdf(conf)

                    # 线条粗细不变
                    ax.plot(x, y, linewidth=3.0, label=label)
                    plotted_any = True
            else:
                ax.text(0.5, 0.75, "MISSING\nDIR",
                        ha="center", va="center",
                        transform=ax.transAxes)

            if not plotted_any and base_dir.exists():
                ax.text(0.5, 0.75, "MISSING\nFILES",
                        ha="center", va="center",
                        transform=ax.transAxes)

            ax.set_xlim(X_MIN, X_MAX)
            ax.set_ylim(Y_MIN, Y_MAX)
            ax.set_xticks(x_ticks)
            ax.set_yticks(y_ticks)
            ax.tick_params(labelbottom=True, labelleft=True, width=1.2, length=4, pad=1)

            # 去掉网格
            ax.grid(False)

            # 每个子图都要 xlabel
            ax.set_xlabel("Confidence", labelpad=2)

            # ylabel 只放两个：每行最左边一个
            if c == 0:
                ax.set_ylabel("Complementary CDF", labelpad=2)
            else:
                ax.set_ylabel("")

            if legend_handles is None:
                handles, labels = ax.get_legend_handles_labels()
                if len(handles) > 0:
                    legend_handles, legend_labels = handles, labels

    # ✅ legend 距离子图更近：把 y 调低一点（同时 top 已经调高）
    if legend_handles is not None:
        fig.legend(
            legend_handles, legend_labels,
            loc="upper center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, 0.965),
            handlelength=2.4,
            handletextpad=0.8,
            columnspacing=1.6
        )

    out_path = MODELS_ROOT / "ccdf_2x5.png"
    # ✅ 关键：防止右侧刻度被截 + 顺便收紧空白
    fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
    print(f"[OK] Saved: {out_path}")


if __name__ == "__main__":
    main()
