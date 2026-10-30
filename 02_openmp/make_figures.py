import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from make_table import GRID, INK, MUTED, parse

HERE = Path(__file__).parent
OUT = HERE / "results"
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#104281", "#3987e5", "#f0efec", "#e34948", "#8f1d1d"])


def style(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.set_title(title, color=INK, fontsize=12, loc="left")
    ax.tick_params(colors=MUTED, length=0)
    for side in ax.spines.values():
        side.set_visible(False)


def load(dump):
    raw = Path(dump).read_bytes()
    M, N = np.frombuffer(raw[:8], dtype=np.int32)
    w = np.frombuffer(raw[8:]).reshape(M + 1, N + 1)
    X, Y = np.meshgrid(np.linspace(-1, 1, M + 1), np.linspace(-1, 1, N + 1), indexing="ij")
    return M, N, X, Y, np.ma.masked_where((X - 1) ** 2 + (Y - 1) ** 2 < 1, w)


def draw(ax, X, Y, field, title, labels=True):
    img = ax.pcolormesh(X, Y, field, cmap=DIVERGING, vmin=-1, vmax=1, shading="nearest", rasterized=True)
    lines = ax.contour(X, Y, field, levels=np.arange(-0.8, 0.81, 0.2), colors=INK, linewidths=0.6, alpha=0.55)
    if labels:
        ax.clabel(lines, fmt=lambda v: f"{v + 0.0:.1f}".replace("-0.0", "0.0"), fontsize=7, colors=INK)
    t = np.linspace(np.pi, 1.5 * np.pi, 200)
    ax.plot(1 + np.cos(t), 1 + np.sin(t), color=INK, linewidth=1.4)
    ax.set_aspect("equal")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    style(ax, "x", "y", title)
    return img


def colorbar(fig, img, axes):
    bar = fig.colorbar(img, ax=axes, fraction=0.046, pad=0.04)
    bar.outline.set_visible(False)
    bar.ax.tick_params(colors=MUTED, length=0)
    bar.set_ticks([-1, -0.5, 0, 0.5, 1])


def solution(dump):
    M, N, X, Y, field = load(dump)
    fig, ax = plt.subplots(figsize=(6.0, 5.2), dpi=150)
    img = draw(ax, X, Y, field, f"Приближённое решение, сетка {M}×{N}")
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.text(0.62, 0.62, "фиктивная\nобласть", ha="center", va="center", color=MUTED, fontsize=9)
    colorbar(fig, img, ax)
    fig.savefig(OUT / "solution.png", bbox_inches="tight")


def grids(dumps):
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 7.0), dpi=150)
    for ax, dump in zip(axes.flat, dumps):
        M, N, X, Y, field = load(dump)
        img = draw(ax, X, Y, field, f"{M}×{N}", labels=False)
        ax.set_title(f"{M}×{N},  w(0,0) = {field[M // 2, N // 2]:.4f}", color=INK, fontsize=10, loc="left")
    for ax in axes[0]:
        ax.set_xlabel("")
    for ax in axes[:, 1]:
        ax.set_ylabel("")
    fig.subplots_adjust(hspace=0.28, wspace=0.12)
    colorbar(fig, img, axes)
    fig.suptitle("Решение на сгущающихся сетках", color=INK, fontsize=12, x=0.05, ha="left")
    fig.savefig(OUT / "solution_grids.png", bbox_inches="tight")


def convergence():
    runs = parse(sorted((HERE.parent / "01_serial" / "runs" / "a4").glob("serial.*.out")))
    grids_ = [(20, 20), (40, 40), (80, 80), (400, 600), (800, 1200)]
    w = [min(runs[("serial", M, N, 1)])[2] for M, N in grids_]
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=150)
    xs = range(len(grids_))
    ax.plot(xs, w, color="#2a78d6", linewidth=2, marker="o", markersize=7,
            markeredgecolor="white", markeredgewidth=1.5)
    for x, v in zip(xs, w):
        ax.annotate(f"{v:.4f}", (x, v), textcoords="offset points", xytext=(0, 9), ha="center",
                    color=INK, fontsize=9)
    ax.set_xticks(list(xs), [f"{M}×{N}" for M, N in grids_])
    ax.set_ylim(0.2375, 0.2445)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    style(ax, "Сетка M×N", "w(0, 0)", "Значение в точке (0, 0) при сгущении сетки")
    fig.tight_layout()
    fig.savefig(OUT / "grid_convergence.png")


SERIES3 = [("80x80", "#2a78d6", "o"), ("400x600", "#eb6834", "s"), ("800x1200", "#1baf7a", "^")]


def history(kind, name):
    data = np.loadtxt(OUT / "history" / f"{kind}_{name}.txt", skiprows=1)
    return data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4]


def log_panel(ax, title, ylabel):
    ax.set_yscale("log")
    ax.grid(True, color=GRID, linewidth=0.8)
    style(ax, "Номер итерации k", ylabel, title)


def iterations():
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)
    for name, color, _ in SERIES3:
        k, dw, res, _, _ = history("real", name)
        label = name.replace("x", "×")
        left.plot(k, dw, color=color, linewidth=1.6, label=label)
        right.plot(k, res, color=color, linewidth=1.6, label=label)
    left.axhline(1e-6, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.2)
    left.annotate("δ = 10⁻⁶", (0, 1e-6), textcoords="offset points", xytext=(4, 4), color=MUTED, fontsize=9)
    log_panel(left, "Изменение решения за итерацию", r"$\|w^{(k+1)} - w^{(k)}\|_E$")
    log_panel(right, "Невязка", r"$\|r^{(k)}\|_E = \|B - Aw^{(k)}\|_E$")
    left.legend(frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT / "iterations.png")


def errors():
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)
    for name, color, _ in SERIES3:
        label = name.replace("x", "×")
        k, _, _, err_c, _ = history("real", name)
        left.plot(k, err_c, color=color, linewidth=1.6, label=label)
        k, _, _, err_c, _ = history("exact", name)
        right.plot(k, err_c, color=color, linewidth=1.6, label=label)
        right.annotate(f"{err_c[-1]:.1e}", (k[-1], err_c[-1]), textcoords="offset points",
                       xytext=(4, -10), color=INK, fontsize=8)
    log_panel(left, "Вариант 18: ошибка относительно w* (δ = 10⁻¹²)", r"$\max\,|w^{(k)} - w^*|$")
    log_panel(right, "Тестовая задача: ошибка относительно u = sin(x+y)", r"$\max\,|w^{(k)} - u|$")
    left.legend(frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT / "error_iterations.png")


def binding():
    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=150)
    ax.plot([1, 16], [1, 16], linestyle=(0, (4, 3)), color=MUTED, linewidth=1.2, label="идеальное")
    for folder, label, color, marker in [("a2", "попытка 2: без привязки", "#eb6834", "s"),
                                         ("a3", "попытка 3: нить на ядро", "#2a78d6", "o")]:
        runs = parse(sorted((HERE / "runs" / folder).glob("omp.*.out")))
        base = min(runs[("omp", 400, 600, 1)])[0]
        ts = [1, 2, 4, 8, 16]
        sp = [base / min(runs[("omp", 400, 600, t)])[0] for t in ts]
        ax.plot(ts, sp, color=color, linewidth=2, marker=marker, markersize=7,
                markeredgecolor="white", markeredgewidth=1.5, label=label)
        ax.annotate(f"{sp[-1]:.1f}", (ts[-1], sp[-1]), textcoords="offset points",
                    xytext=(8, -4), color=INK, fontsize=9)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ticks = [1, 2, 4, 8, 16]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.set_yticks(ticks, [str(t) for t in ticks])
    ax.set_xlim(0.8, 22)
    ax.set_ylim(0.8, 22)
    ax.grid(True, color=GRID, linewidth=0.8)
    style(ax, "Число OpenMP-нитей", "Ускорение относительно 1 нити",
          "Влияние привязки нитей к ядрам, сетка 400×600")
    ax.legend(frameon=False, loc="upper left", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT / "binding.png")


def main():
    OUT.mkdir(exist_ok=True)
    binding()
    convergence()
    iterations()
    errors()
    dumps = sys.argv[1:]
    if dumps:
        solution(dumps[-1])
    if len(dumps) == 4:
        grids(dumps)


if __name__ == "__main__":
    main()
