"""Shared matplotlib style: one validated categorical order, recessive chrome, one diverging map."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from glioma.config import FIGURES  # noqa: E402

# categorical slots in fixed order (validated light-mode palette); never cycled
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div_blue_red", ["#1c5cab", "#86b6ef", "#f0efec", "#ec9a8f", "#c0302f"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titleweight": "bold",
    "axes.titlesize": 11, "axes.labelsize": 9.5, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 8.5, "text.color": INK, "lines.linewidth": 2,
    "font.family": "sans-serif", "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
})


def colors_for(classes) -> dict:
    """Colour follows the entity: class -> slot by its position in the canonical class order."""
    return {c: SERIES[i % len(SERIES)] for i, c in enumerate(classes)}


def save(fig, name: str) -> str:
    path = FIGURES / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path.name
