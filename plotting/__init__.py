"""Shared publication figure style for the ANN-stability experiments.

The conventions here are adapted from the ``scientific-figure-making`` skill in
the figures4papers repository (https://github.com/ChenLiu-1996/figures4papers).
"""

from .pubstyle import (  # noqa: F401
    DEFAULT_COLORS,
    DEFAULT_LINESTYLES,
    DEFAULT_MARKERS,
    PALETTE,
    FigureStyle,
    add_reference_line,
    add_shared_legend,
    apply_publication_style,
    create_subplots,
    finalize_figure,
    format_log_axis,
    load_results,
    make_boxplot,
    make_trend,
    save_results,
)
