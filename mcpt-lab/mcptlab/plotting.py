"""Plots for permutation tests. Gray histogram = what noise produced; dark line = the real result."""
import glob
import os

import numpy as np

NOISE, REAL, INK, MUTED, GRID = "#9aa7b8", "#1f2d3d", "#1f2d3d", "#5b6b7c", "#c9d1db"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(labelsize=8, colors=MUTED)


def _panel(ax, null, real, title, xlabel="profit factor", fmt="{:.3f}"):
    null = np.asarray(null, float)
    p = (1 + np.sum(null >= real)) / (len(null) + 1)
    ax.hist(null, bins=40, color=NOISE, edgecolor="white", linewidth=0.5)
    ax.axvline(real, color=REAL, lw=2)
    ymax = ax.get_ylim()[1]
    ax.text(real, ymax * 0.93, " real " + fmt.format(real) + " ", color=INK, fontsize=9, va="top",
            ha="left" if real < np.percentile(null, 80) else "right")
    ax.set_title(f"{title}\np = {p:.3f}  ({len(null)} permutations)", fontsize=10, color=INK, loc="left")
    ax.set_xlabel(xlabel, fontsize=8, color=MUTED)
    _style(ax)
    return p


def plot_null(res, path, title):
    """One test: histogram of the null distribution with the real value marked."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4))
    xl = ("best in-sample " if res["test"].startswith("in_sample") else "") + "profit factor"
    _panel(ax, res["null"], res["real"], title, xl)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_grid(null_files, path, cols=3, suptitle=None, xlabel="profit factor", title_fn=None, fmt="{:.3f}"):
    """Many tests in one figure, from CSVs written by core.save_null (header 'title|real')."""
    from .core import load_null
    plt = _plt()
    panels = [load_null(f) for f in null_files]
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 3.3 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (title, real, null) in zip(axes, panels):
        _panel(ax, null, real, title_fn(title) if title_fn else title, xlabel, fmt)
    for ax in axes[len(panels):]:
        ax.axis("off")
    fig.suptitle(suptitle or "Permutation tests: gray = best result on shuffled data, line = real data",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=130); plt.close(fig)
    return path


def plot_runs(runs_dir="mcpt_runs", path="all_tests.png"):
    """Everything your ledger has auto-saved, in one figure."""
    files = sorted(glob.glob(os.path.join(runs_dir, "*_null.csv")))
    if not files:
        raise FileNotFoundError(f"no *_null.csv files in {runs_dir}")
    return plot_grid(files, path)
