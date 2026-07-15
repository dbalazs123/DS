"""Plotting helpers with a consistent, readable default theme."""

from __future__ import annotations

import matplotlib as mpl
from cycler import cycler

# A restrained, colour-blind-friendly categorical palette (Okabe-Ito).
PALETTE = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
    "#F0E442",
    "#000000",
]


def set_theme(context: str = "notebook") -> None:
    """Apply the toolkit's default matplotlib theme.

    Sets a clean, legible style and the shared colour cycle so every chart in
    the project looks like it belongs to the same report.

    Args:
        context: One of ``"notebook"``, ``"talk"`` or ``"paper"``; controls the
            base font size.
    """
    font_sizes = {"paper": 10.0, "notebook": 12.0, "talk": 15.0}
    mpl.rcParams.update(
        {
            "figure.figsize": (8.0, 5.0),
            "figure.dpi": 100,
            "font.size": font_sizes.get(context, 12.0),
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.3,
            "axes.prop_cycle": cycler(color=PALETTE),
        }
    )


__all__ = ["PALETTE", "set_theme"]
