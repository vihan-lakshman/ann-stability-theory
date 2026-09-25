"""Multi-vector stability across data distributions (Chamfer vs. mean pooling).

Companion to ``synthetic_stability.py``. That script fixes one adversarial
construction (nearest/furthest neighbours plus their antipodes) and sweeps the
dimension; this one sweeps the dimension for *several* data distributions, to
show how the gap between Chamfer distance and mean pooling depends on the
geometry of the sets rather than on one hand-built counter-example.

Three generators are available (``--distribution``):

``iid``
    Every vector is an i.i.d. standard Gaussian draw. The baseline in which
    both aggregations lose contrast as the dimension grows.
``antipodal``
    Every set contains ``v`` and ``-v``, so mean pooling cancels; queries are
    noisy copies of the first ``Q`` documents. The distributional analogue of
    the construction in ``synthetic_stability.py``.
``lowrank-modes``
    ``num_modes`` fixed semantic modes on the unit sphere, each with its own
    locally low-rank Gaussian variation; a document draws ``data_modes`` of
    them and a query ``query_modes``. The closest of the three to real
    multi-vector encoders.

Distances here are squared Euclidean (``synthetic_stability.py`` uses cosine),
so the stability ratios are not directly comparable between the two figures.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from plotting import (  # noqa: E402
    PALETTE,
    FigureStyle,
    add_reference_line,
    apply_publication_style,
    create_subplots,
    finalize_figure,
    format_log_axis,
    load_results,
    make_trend,
    save_results,
)

DEFAULT_SEED = 42


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def compute_metrics(dist_matrix: np.ndarray) -> Tuple[float, float]:
    """Relative variance and stability ratio from a ``(Q, N)`` distance matrix.

    Both are computed per query row and then averaged over queries.
    """
    mean_dists = dist_matrix.mean(axis=1)
    var_dists = dist_matrix.var(axis=1)
    min_dists = dist_matrix.min(axis=1)
    max_dists = dist_matrix.max(axis=1)

    rel_var = (var_dists / (mean_dists ** 2)).mean()
    # Stability ratio (max / min): the larger, the more contrast survives.
    stability_ratio = (max_dists / np.where(min_dists == 0, 1e-9, min_dists)).mean()
    return float(rel_var), float(stability_ratio)


# --------------------------------------------------------------------------- #
# Data generators
# --------------------------------------------------------------------------- #


def iid_gaussian(d: int, N: int, Q: int, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Standard i.i.d. noise: the default case, where stability decays."""
    dataset = np.random.normal(0, 1, (N, k, d))
    queries = np.random.normal(0, 1, (Q, k, d))
    return dataset, queries


def antipodal(d: int, N: int, Q: int, k: int = 4, noise_std: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """The antipodal distribution: for every ``v`` in a set, ``-v`` is present too."""
    # k must be even so that every vector has an antipodal partner.
    assert k % 2 == 0, "k must be even for antipodal pairs"

    base_docs = np.random.normal(0, 1, (N, k // 2, d))

    dataset = np.zeros((N, k, d))
    dataset[:, : k // 2, :] = base_docs
    dataset[:, k // 2 :, :] = -base_docs

    # Queries are noisy versions of the first Q documents, so each has a match.
    queries = dataset[:Q].copy() + np.random.normal(0, noise_std, (Q, k, d))

    return dataset, queries


def lowrank_modes(
    d: int,
    N: int,
    Q: int,
    num_modes: int = 20,
    data_modes: int = 2,
    query_modes: int = 2,
    noise_sigma: float = 0.01,
    local_rank: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """``data_modes`` of ``num_modes`` semantic modes per set, with low-rank noise."""
    k, n, nq = num_modes, data_modes, query_modes

    # Fixed modes on the unit sphere.
    modes = np.random.normal(0, 1, (k, d))
    modes /= np.linalg.norm(modes, axis=1, keepdims=True)

    # A (d, local_rank) orthonormal basis per mode, for its local variation.
    bases = np.random.normal(0, 1, (k, d, local_rank))
    for i in range(k):
        q, _ = np.linalg.qr(bases[i])
        bases[i] = q[:, :local_rank]

    def get_noisy_samples(mode_indices: np.ndarray) -> np.ndarray:
        M = len(mode_indices)
        z = np.random.normal(0, noise_sigma, (M, local_rank))
        # Batched bases[i] @ z[i]: lift the low-rank noise into d dimensions.
        noise = np.einsum("mij,mj->mi", bases[mode_indices], z)
        samples = modes[mode_indices] + noise
        return samples / np.linalg.norm(samples, axis=1, keepdims=True)

    dataset = np.zeros((N, n, d))
    for i in range(N):
        dataset[i] = get_noisy_samples(np.random.choice(k, n, replace=False))

    queries = np.zeros((Q, nq, d))
    for i in range(Q):
        queries[i] = get_noisy_samples(np.random.choice(k, nq, replace=False))

    return dataset, queries


# The configurations the paper reports, keyed by ``--distribution``.
DISTRIBUTIONS: Dict[str, Dict] = {
    "iid": {
        "fn": iid_gaussian,
        "kwargs": {"k": 5},
        "title": "i.i.d. Gaussian vectors",
    },
    "antipodal": {
        "fn": antipodal,
        "kwargs": {"k": 4, "noise_std": 0.1},
        "title": "Antipodal sets",
    },
    "lowrank-modes": {
        "fn": lowrank_modes,
        "kwargs": {
            "num_modes": 10,
            "data_modes": 10,
            "query_modes": 1,
            "noise_sigma": 0.5,
            "local_rank": 4,
        },
        "title": "Low-rank semantic modes",
    },
}


# --------------------------------------------------------------------------- #
# Experiment
# --------------------------------------------------------------------------- #


def analyze_stability(
    dims: List[int],
    N: int,
    Q: int,
    data_gen_func: Callable[[int, int, int], Tuple[np.ndarray, np.ndarray]],
    seed: int = DEFAULT_SEED,
) -> Dict[str, List[float]]:
    """Sweep the dimension, reporting both metrics for both aggregations.

    ``np.random`` is reseeded per dimension so that each point is independent of
    how many dimensions precede it in ``dims``.
    """
    results = {
        "chamfer_stability": [],
        "chamfer_relvar": [],
        "avgpool_stability": [],
        "avgpool_relvar": [],
    }

    print(f"{'Dim':>6} | {'Chamfer Ratio':>13} | {'Chamfer RelVar':>14} | "
          f"{'AvgPool Ratio':>13} | {'AvgPool RelVar':>14}")
    print("-" * 90)

    for d in dims:
        np.random.seed(seed)

        dataset, queries = data_gen_func(d, N, Q)
        k_doc = dataset.shape[1]
        k_query = queries.shape[1]

        avg_dists = np.zeros((Q, N))
        chamfer_dists = np.zeros((Q, N))

        D_flat = dataset.reshape(-1, d)  # (N * k_doc, d)
        d_sq = np.sum(D_flat ** 2, axis=1, keepdims=True).T

        for i in range(Q):
            q_i = queries[i]  # (k_query, d)

            # Squared Euclidean, pairwise: (k_query, N * k_doc)
            dists_pw = np.sum(q_i ** 2, axis=1, keepdims=True) + d_sq - 2 * (q_i @ D_flat.T)

            # dists_pw[x, y, z]: query vector x to doc y's vector z.
            dists_pw = dists_pw.reshape(k_query, N, k_doc)

            # Mean pooling: average over query vectors and over doc vectors.
            avg_dists[i] = dists_pw.mean(axis=(0, 2))

            # Chamfer: min over doc vectors, then mean over query vectors.
            chamfer_dists[i] = dists_pw.min(axis=2).mean(axis=0)

        for key, dists in [("chamfer", chamfer_dists), ("avgpool", avg_dists)]:
            rv, sr = compute_metrics(dists)
            results[f"{key}_relvar"].append(rv)
            results[f"{key}_stability"].append(sr)

        print(f"{d:>6} | {results['chamfer_stability'][-1]:>13.4f} | "
              f"{results['chamfer_relvar'][-1]:>14.6f} | "
              f"{results['avgpool_stability'][-1]:>13.4f} | "
              f"{results['avgpool_relvar'][-1]:>14.6f}")

    return results


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #


def plot_results(dimensions: List[int], results: Dict, output_path: str) -> None:
    """Two-panel figure: stability ratio and relative variance vs. dimension.

    Same encoding as ``synthetic_stability.py``: blue = Chamfer distance
    (provably stable), red = mean pooling, markers as the print-safe secondary
    channel.
    """
    apply_publication_style(FigureStyle(font_size=18, axes_linewidth=2))
    fig, (ax1, ax2) = create_subplots(1, 2, figsize=(13, 4.8))

    labels = ["Chamfer distance", "Mean pooling"]
    colors = [PALETTE["blue_main"], PALETTE["red_strong"]]
    markers = ["o", "s"]

    make_trend(
        ax1,
        dimensions,
        [results["chamfer_stability"], results["avgpool_stability"]],
        labels,
        colors=colors,
        markers=markers,
        xlabel="Dimension",
        ylabel=r"Stability ratio ($d_{\max}/d_{\min}$)",
        xscale="log",
        yscale="log",
    )
    add_reference_line(ax1, 1.0, label="Instability threshold")
    format_log_axis(ax1, "x")
    format_log_axis(ax1, "y")
    ax1.legend(loc="best")

    make_trend(
        ax2,
        dimensions,
        [results["chamfer_relvar"], results["avgpool_relvar"]],
        labels,
        colors=colors,
        markers=markers,
        xlabel="Dimension",
        ylabel="Relative variance",
        xscale="log",
        yscale="log",
    )
    format_log_axis(ax2, "x")
    format_log_axis(ax2, "y")
    ax2.legend(loc="best")

    saved = finalize_figure(fig, output_path, formats=["png", "pdf"], dpi=300)
    print("\nPlot saved to: " + ", ".join(str(s) for s in saved))


def print_analysis(results: Dict, title: str) -> None:
    print()
    print("=" * 90)
    print(f"Analysis - {title}")
    print("=" * 90)
    print()
    print("Chamfer distance:")
    print(f"  - Stability ratio: {min(results['chamfer_stability']):.4f} to "
          f"{max(results['chamfer_stability']):.4f}")
    print(f"  - Relative variance: {min(results['chamfer_relvar']):.6f} to "
          f"{max(results['chamfer_relvar']):.6f}")
    print()
    print("Mean pooling:")
    print(f"  - Stability ratio: {min(results['avgpool_stability']):.4f} to "
          f"{max(results['avgpool_stability']):.4f}")
    print(f"  - Relative variance: {min(results['avgpool_relvar']):.6f} to "
          f"{max(results['avgpool_relvar']):.6f}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def default_paths(distribution: str) -> Tuple[Path, Path]:
    stem = f"multivector_distribution_{distribution.replace('-', '_')}"
    return ROOT / "results" / f"{stem}.json", ROOT / "figures" / stem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-vector stability across data distributions: Chamfer vs. mean pooling."
    )
    parser.add_argument("--distribution", choices=sorted(DISTRIBUTIONS), default="lowrank-modes",
                        help="Data distribution to sweep (default: lowrank-modes)")
    parser.add_argument("--min-dim-exp", type=int, default=2, help="Smallest dimension is 2**this")
    parser.add_argument("--max-dim-exp", type=int, default=15, help="Largest dimension is 2**this")
    parser.add_argument("--num-docs", type=int, default=500, help="Number of document sets (N)")
    parser.add_argument("--num-queries", type=int, default=100, help="Number of query sets (Q)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="NumPy seed, re-applied per dimension")
    parser.add_argument("--results", default=None, help="JSON file to write/read experiment results")
    parser.add_argument("--figure", default=None, help="Output figure stem (PNG + PDF are written)")
    parser.add_argument("--plot-only", action="store_true", help="Skip the experiment and re-plot saved results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = DISTRIBUTIONS[args.distribution]
    results_path, figure_path = default_paths(args.distribution)
    results_path = args.results or str(results_path)
    figure_path = args.figure or str(figure_path)

    if args.plot_only:
        saved = load_results(results_path)
        results, dimensions = saved["results"], saved["dimensions"]
    else:
        dimensions = [2 ** i for i in range(args.min_dim_exp, args.max_dim_exp + 1)]
        params = dict(
            distribution=args.distribution,
            generator_kwargs=spec["kwargs"],
            N=args.num_docs,
            Q=args.num_queries,
            seed=args.seed,
        )

        print("=" * 90)
        print(f"Multi-Vector Stability Across Distributions - {spec['title']}")
        print("=" * 90)
        print()

        results = analyze_stability(
            dims=dimensions,
            N=args.num_docs,
            Q=args.num_queries,
            data_gen_func=lambda d, n, q: spec["fn"](d, n, q, **spec["kwargs"]),
            seed=args.seed,
        )
        save_results({"results": results, "dimensions": dimensions, "params": params}, results_path)
        print(f"\nResults saved to: {results_path}")
        print_analysis(results, spec["title"])

    plot_results(dimensions, results, figure_path)


if __name__ == "__main__":
    main()
