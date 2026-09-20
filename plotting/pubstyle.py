"""Publication-quality matplotlib helpers (figures4papers house style).

This module implements the API described in
``figures4papers/scientific-figure-making/references/api.md``:

* ``PALETTE`` / ``DEFAULT_COLORS`` -- the semantic colour map (blue for the
  key / proposed setting, reds for contrasts and baselines, neutrals for
  reference lines).
* ``apply_publication_style`` -- rcParams preset: sans-serif type, top/right
  spines removed, frameless legends, no grid.
* ``make_trend`` -- multi-series line plot with distinct markers and line
  styles so that series are never distinguished by colour alone.
* ``finalize_figure`` -- ``tight_layout(pad=2)`` + export to PNG and PDF.

It also provides ``save_results`` / ``load_results`` so that every experiment
script can persist its numbers to JSON and re-render the figure without
re-running the (sometimes expensive) experiment.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import sys

import matplotlib
import matplotlib.colors
import matplotlib.ticker

if not sys.flags.interactive and "ipykernel" not in sys.modules:
    # Headless by default: every script here is run from the command line.
    matplotlib.use("Agg")

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterSciNotation, LogLocator, NullFormatter, ScalarFormatter

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "blue_light": "#9BC8FA",
    "green_1": "#DDF3DE",
    "green_2": "#AADCA9",
    "green_3": "#8BCF8B",
    "red_1": "#F6CFCB",
    "red_2": "#E9A6A1",
    "red_soft": "#D88F8A",
    "red_strong": "#B64342",
    "red_dark": "#850C0A",
    "neutral": "#CFCECE",
    "gray_mid": "#767676",
    "gray_dark": "#4D4D4D",
    "highlight": "#FFD700",
    "teal": "#42949E",
    "violet": "#9A4D8E",
}

# Ordered colours used when a caller does not pass its own. Blue is reserved
# for the key / stable setting; reds and neutrals for the contrasts.
DEFAULT_COLORS = [
    PALETTE["blue_main"],
    PALETTE["red_strong"],
    PALETTE["red_soft"],
    PALETTE["gray_mid"],
    PALETTE["teal"],
    PALETTE["violet"],
]

# Secondary encodings: assigned in fixed order, in parallel with colours, so
# that a reader with colour-vision deficiency (or a grayscale print) can still
# tell the series apart.
DEFAULT_MARKERS = ["o", "s", "^", "D", "v", "P"]
DEFAULT_LINESTYLES = ["-"] * 6  # markers carry the secondary encoding

_PREFERRED_FONTS = ("Helvetica", "Arial", "Helvetica Neue", "DejaVu Sans", "sans-serif")


# --------------------------------------------------------------------------- #
# Style
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FigureStyle:
    """Typographic / spine settings.

    ``font_size`` is the base size; axis labels and tick labels are derived
    from it (see :func:`apply_publication_style`). A two-panel figure of
    width 12 in that is later scaled to a 5.5 in text column keeps an
    effective ~8 pt type at ``font_size=18``.
    """

    font_size: int = 18
    axes_linewidth: float = 2.0
    use_tex: bool = False
    font_family: tuple[str, ...] = _PREFERRED_FONTS


def _first_available_font(candidates: Iterable[str]) -> str | None:
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return None


def apply_publication_style(style: FigureStyle | None = None) -> FigureStyle:
    """Configure matplotlib rcParams once, before any figure is created."""
    style = style or FigureStyle()
    plt.style.use("default")

    family = list(style.font_family)
    chosen = _first_available_font(family)

    rc: dict[str, Any] = {
        "font.family": "sans-serif",
        "font.sans-serif": family,
        "font.size": style.font_size,
        "axes.labelsize": style.font_size + 2,
        "axes.titlesize": style.font_size + 2,
        "xtick.labelsize": style.font_size - 2,
        "ytick.labelsize": style.font_size - 2,
        "legend.fontsize": style.font_size - 3,
        "axes.linewidth": style.axes_linewidth,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.axisbelow": True,
        "xtick.major.size": 8,
        "ytick.major.size": 8,
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
        "xtick.minor.size": 4,
        "ytick.minor.size": 4,
        "xtick.minor.width": 1.0,
        "ytick.minor.width": 1.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "legend.handlelength": 2.4,
        "lines.linewidth": 3,
        "lines.markersize": 9,
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,  # embed TrueType so text stays editable in PDF
        "ps.fonttype": 42,
        "text.usetex": style.use_tex,
    }
    if chosen and not style.use_tex:
        # Keep math (e.g. d_max / d_min) in the same face as the body text.
        rc.update(
            {
                "mathtext.fontset": "custom",
                "mathtext.rm": chosen,
                "mathtext.it": f"{chosen}:italic",
                "mathtext.bf": f"{chosen}:bold",
            }
        )
    plt.rcParams.update(rc)
    return style


def create_subplots(nrows: int = 1, ncols: int = 1, figsize=None, **kwargs):
    """Return ``(fig, axes)`` with ``axes`` always a flat 1-D array."""
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    axes = np.atleast_1d(axes).ravel()
    return fig, axes


# --------------------------------------------------------------------------- #
# Plot helpers
# --------------------------------------------------------------------------- #


def make_trend(
    ax,
    x: Sequence[float],
    y_series: Sequence[Sequence[float]],
    labels: Sequence[str],
    colors: Sequence[str] | None = None,
    markers: Sequence[str] | None = None,
    linestyles: Sequence[str] | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    xscale: str | None = None,
    yscale: str | None = None,
    linewidth: float = 3.0,
    markersize: float = 9.0,
    alpha: float = 1.0,
    show_shadow: bool = False,
    y_lower: Sequence[Sequence[float]] | None = None,
    y_upper: Sequence[Sequence[float]] | None = None,
) -> list[Line2D]:
    """Plot several series against a shared ``x``.

    Each series gets its own colour *and* marker *and* line style (assigned in
    fixed order), which is the house convention for print-safe line plots.
    Optional ``y_lower`` / ``y_upper`` draw an uncertainty band when
    ``show_shadow`` is True.
    """
    x = np.asarray(x, dtype=float)
    if len(y_series) != len(labels):
        raise ValueError("y_series and labels must have the same length")
    colors = list(colors or DEFAULT_COLORS)
    markers = list(markers or DEFAULT_MARKERS)
    linestyles = list(linestyles or DEFAULT_LINESTYLES)

    handles: list[Line2D] = []
    for i, (y, label) in enumerate(zip(y_series, labels)):
        y = np.asarray(y, dtype=float)
        if y.shape != x.shape:
            raise ValueError(f"series {label!r} has length {y.size}, expected {x.size}")
        color = colors[i % len(colors)]
        (line,) = ax.plot(
            x,
            y,
            color=color,
            marker=markers[i % len(markers)],
            linestyle=linestyles[i % len(linestyles)],
            linewidth=linewidth,
            markersize=markersize,
            markeredgecolor="white",
            markeredgewidth=1.2,
            alpha=alpha,
            label=label,
            zorder=3,
        )
        handles.append(line)
        if show_shadow and y_lower is not None and y_upper is not None:
            ax.fill_between(x, y_lower[i], y_upper[i], color=color, alpha=0.15, linewidth=0)

    if xscale:
        ax.set_xscale(xscale)
    if yscale:
        ax.set_yscale(yscale)
    if xlabel:
        ax.set_xlabel(xlabel, labelpad=10)
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=10)
    return handles


def make_boxplot(
    ax,
    data: Sequence[Sequence[float]],
    labels: Sequence[str],
    colors: Sequence[str] | None = None,
    whis=(5, 95),
    width: float = 0.45,
    ylabel: str | None = None,
    yscale: str | None = None,
    showfliers: bool = False,
):
    """Filled box plots, one box per series, in the house palette.

    Boxes are filled with the series colour at reduced opacity and outlined
    in the full colour; the median is drawn as a dark line. Whiskers default
    to the 5th-95th percentiles and outliers are hidden (as in the paper).
    """
    colors = list(colors or DEFAULT_COLORS)
    bp = ax.boxplot(
        [np.asarray(d, dtype=float) for d in data],
        positions=np.arange(len(data)),
        widths=width,
        whis=whis,
        showfliers=showfliers,
        patch_artist=True,
        medianprops={"color": PALETTE["gray_dark"], "linewidth": 2.5},
        whiskerprops={"linewidth": 2},
        capprops={"linewidth": 2},
        boxprops={"linewidth": 2},
    )
    for i, patch in enumerate(bp["boxes"]):
        color = colors[i % len(colors)]
        patch.set_facecolor(matplotlib.colors.to_rgba(color, 0.35))
        patch.set_edgecolor(color)
    for i, (whisk_lo, whisk_hi) in enumerate(zip(bp["whiskers"][::2], bp["whiskers"][1::2])):
        color = colors[i % len(colors)]
        whisk_lo.set_color(color)
        whisk_hi.set_color(color)
    for i, (cap_lo, cap_hi) in enumerate(zip(bp["caps"][::2], bp["caps"][1::2])):
        color = colors[i % len(colors)]
        cap_lo.set_color(color)
        cap_hi.set_color(color)
    ax.set_xticks(np.arange(len(data)))
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.6, len(data) - 0.4)
    if yscale:
        ax.set_yscale(yscale)
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=10)
    return bp


def add_reference_line(
    ax,
    y: float,
    label: str | None = None,
    color: str = "black",
    alpha: float = 0.35,
    linewidth: float = 3.0,
    linestyle: str = "--",
) -> Line2D:
    """Horizontal reference line (e.g. an instability threshold)."""
    return ax.axhline(y=y, color=color, alpha=alpha, linewidth=linewidth, linestyle=linestyle, label=label, zorder=1)


def format_log_axis(ax, axis: str = "x", base: float = 10, ticks: Sequence[float] | None = None) -> None:
    """Tidy a logarithmic axis.

    * ``base=10``: major ticks at powers of ten (labelled ``10^k``), minor
      ticks at 2..9 without labels.
    * any other base (or explicit ``ticks``): label the given tick positions
      with plain integers, which is what you want for powers of two.
    """
    ax_obj = ax.xaxis if axis == "x" else ax.yaxis
    if ticks is not None:
        ax_obj.set_major_locator(matplotlib.ticker.FixedLocator(list(ticks)))
        fmt = ScalarFormatter()
        fmt.set_scientific(False)
        ax_obj.set_major_formatter(fmt)
        ax_obj.set_minor_locator(matplotlib.ticker.NullLocator())
        return
    if base == 10:
        ax_obj.set_major_locator(LogLocator(base=10.0, numticks=12))
        ax_obj.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=12))
        lo, hi = ax.get_xlim() if axis == "x" else ax.get_ylim()
        decades = np.log10(hi / lo) if lo > 0 and hi > 0 else 2.0
        if decades < 1.5:
            # Fewer than ~1.5 decades visible: label the minor ticks (2, 5) too,
            # otherwise the axis may carry a single labelled tick.
            ax_obj.set_minor_formatter(
                LogFormatterSciNotation(base=10.0, labelOnlyBase=False, minor_thresholds=(2, 0.5))
            )
        else:
            ax_obj.set_minor_formatter(NullFormatter())
    else:
        ax_obj.set_major_locator(LogLocator(base=base))
        fmt = ScalarFormatter()
        fmt.set_scientific(False)
        ax_obj.set_major_formatter(fmt)


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #

_SUPPORTED = {"pdf", "svg", "eps", "png", "jpg", "jpeg", "tif", "tiff"}


def finalize_figure(
    fig,
    out_path: str | os.PathLike,
    formats: Sequence[str] | None = None,
    dpi: int = 300,
    close: bool = True,
    pad: float = 2.0,
    **savefig_kwargs,
) -> list[Path]:
    """Apply ``tight_layout(pad=pad)`` and save to one or more formats.

    ``out_path`` may carry an extension (used when ``formats`` is None) or be
    a bare stem. Parent directories are created. Returns the saved paths.
    """
    out_path = Path(out_path)
    if formats is None:
        formats = [out_path.suffix.lstrip(".")] if out_path.suffix else ["png", "pdf"]
    stem = out_path.with_suffix("") if out_path.suffix.lstrip(".") in _SUPPORTED else out_path
    stem.parent.mkdir(parents=True, exist_ok=True)

    fig.tight_layout(pad=pad)
    saved: list[Path] = []
    for fmt in formats:
        fmt = fmt.lower().lstrip(".")
        if fmt not in _SUPPORTED:
            raise ValueError(f"unsupported format {fmt!r}; choose from {sorted(_SUPPORTED)}")
        target = stem.with_suffix(f".{fmt}")
        fig.savefig(target, dpi=dpi, format=fmt, **savefig_kwargs)
        saved.append(target)
    if close:
        plt.close(fig)
    return saved


# --------------------------------------------------------------------------- #
# Result persistence
# --------------------------------------------------------------------------- #


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    if hasattr(obj, "to_dict") and callable(obj.to_dict):  # pandas DataFrame
        return _to_jsonable(obj.to_dict(orient="records"))
    return obj


def save_results(results: Any, path: str | os.PathLike) -> Path:
    """Write experiment results (dicts / lists / numpy / DataFrame) to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(_to_jsonable(results), fh, indent=2)
    return path


def load_results(path: str | os.PathLike) -> Any:
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"No saved results at {path}. Run the experiment first (without --plot-only).")
    with open(path) as fh:
        return json.load(fh)
