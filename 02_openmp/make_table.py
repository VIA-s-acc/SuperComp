import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = [("400x600", "#2a78d6", "o"), ("800x1200", "#eb6834", "s")]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def parse(paths):
    runs = defaultdict(list)
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if not line.startswith("RESULT"):
                continue
            f = dict(kv.split("=", 1) for kv in line.split()[1:])
            key = (f["impl"], int(f["M"]), int(f["N"]), int(f["T"]))
            runs[key].append((float(f["time"]), int(f["iters"]), float(f["w00"])))
    return runs


def best(runs, key):
    return min(runs[key]) if key in runs else None


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python3 make_table.py table1.*.out")
    runs = parse(sys.argv[1:])
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)

    md = ["# Таблица 1. OpenMP, IBM Polus", "",
          "Время — минимум из двух запусков. Ускорение = время serial / время OpenMP на той же сетке.", "",
          "| Нити | Сетка M×N | Итерации | Время, с | Ускорение |", "|---|---|---|---|---|"]
    csv = ["threads,M,N,iters,time,speedup"]
    points = defaultdict(list)
    for (M, N), threads in [((400, 600), [2, 4, 8, 16, 32]), ((800, 1200), [2, 4, 8, 16, 32])]:
        base = best(runs, ("serial", M, N, 1))
        for t in threads:
            r = best(runs, ("omp", M, N, t))
            if not r or not base:
                md.append(f"| {t} | {M}×{N} | — | — | — |")
                continue
            sp = base[0] / r[0]
            points[f"{M}x{N}"].append((t, sp))
            md.append(f"| {t} | {M}×{N} | {r[1]} | {r[0]:.4f} | {sp:.2f} |")
            csv.append(f"{t},{M},{N},{r[1]},{r[0]:.6f},{sp:.3f}")

    md += ["", "Последовательная версия (база ускорения):", "",
           "| Сетка M×N | Итерации | Время, с | w(0,0) |", "|---|---|---|---|"]
    for M, N in [(80, 80), (400, 600), (800, 1200)]:
        r = best(runs, ("serial", M, N, 1))
        if r:
            md.append(f"| {M}×{N} | {r[1]} | {r[0]:.4f} | {r[2]:.12f} |")

    md += ["", "## 80×80: OpenMP против serial", "",
           "| Нити | Итерации | Время, с | Ускорение | w(0,0) | Отличие от serial |", "|---|---|---|---|---|---|"]
    s80 = best(runs, ("serial", 80, 80, 1))
    for t in [1, 4, 16]:
        r = best(runs, ("omp", 80, 80, t))
        if r and s80:
            md.append(f"| {t} | {r[1]} | {r[0]:.4f} | {s80[0] / r[0]:.2f} | {r[2]:.12f} | {abs(r[2] - s80[2]):.1e} |")

    (out / "table1.md").write_text("\n".join(md) + "\n")
    (out / "table1.csv").write_text("\n".join(csv) + "\n")

    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=150)
    top = 32
    ax.plot([1, top], [1, top], linestyle=(0, (4, 3)), color=MUTED, linewidth=1.2, label="идеальное")
    for name, color, marker in SERIES:
        pts = points.get(name)
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=color, linewidth=2, marker=marker, markersize=7,
                markeredgecolor="white", markeredgewidth=1.5, label=name.replace("x", "×"))
        ax.annotate(f"{ys[-1]:.1f}", (xs[-1], ys[-1]), textcoords="offset points",
                    xytext=(8, -4), color=INK, fontsize=9)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ticks = [1, 2, 4, 8, 16, 32]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.set_yticks(ticks, [str(t) for t in ticks])
    lowest = min([sp for pts in points.values() for _, sp in pts] + [1])
    ticks = [t for t in [0.25, 0.5] if t >= lowest / 2] + ticks
    ax.set_yticks(ticks, [f"{t:g}" for t in ticks])
    ax.set_xlim(1.6, 40)
    ax.set_ylim(min(1.6, lowest / 1.25), 40)
    ax.set_xlabel("Число OpenMP-нитей", color=MUTED)
    ax.set_ylabel("Ускорение относительно serial", color=MUTED)
    ax.set_title("Ускорение OpenMP-программы на IBM Polus", color=INK, fontsize=12, loc="left")
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.tick_params(colors=MUTED, length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.legend(frameon=False, loc="upper left", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out / "speedup_omp.png")
    print((out / "table1.md").read_text())


if __name__ == "__main__":
    main()
