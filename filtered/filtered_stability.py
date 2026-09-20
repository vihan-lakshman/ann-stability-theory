import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

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

RESULTS_PATH = ROOT / "results" / "filtered_stability.json"
FIGURE_PATH = ROOT / "figures" / "filtered_stability"


def generate_unstable_vectors(rng: np.random.Generator, n: int, dim: int) -> np.ndarray:
    """
    Generate vectors from iid Gaussian distribution (unstable by Beyer et al.).
    """
    vectors = rng.normal(0, 1.0 / np.sqrt(dim), size=(n, dim))
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.where(norms > 0, norms, 1)
    return vectors


def cosine_distance(q: np.ndarray, docs: np.ndarray) -> np.ndarray:
    """Compute cosine distance between query and documents."""
    dots = docs @ q
    dots = np.clip(dots, -1.0, 1.0)
    return 1.0 - dots


def cosine_distance_batch(queries: np.ndarray, docs: np.ndarray) -> np.ndarray:
    """Compute cosine distances between all queries and all documents.
    
    Returns: (n_queries, n_docs) array of distances
    """
    dots = queries @ docs.T  # (n_queries, n_docs)
    dots = np.clip(dots, -1.0, 1.0)
    return 1.0 - dots


def assign_filters_with_negative_covariance_fixed_db(
    docs: np.ndarray,
    queries: np.ndarray,
    p_mismatch: float,
    rng: np.random.Generator,
    correlation_strength: float = 0.999
) -> np.ndarray:
    """
    Assign filter matches to documents such that there's negative covariance
    between distance and filter match for typical queries.
    
    Strategy: Compute average distance from each document to all queries.
    Documents that are close to queries on average become mismatches.
    This creates negative covariance: close documents don't match the filter.
    
    Returns: (n_docs,) array of filter match indicators (1=match, 0=mismatch)
    """
    n_docs = len(docs)
    
    # Compute average distance from each document to all queries
    # Shape: (n_queries, n_docs) -> mean over queries -> (n_docs,)
    all_distances = cosine_distance_batch(queries, docs)
    avg_distances = np.mean(all_distances, axis=0)
    
    # Sort documents by average distance (closest first)
    sorted_indices = np.argsort(avg_distances)
    
    # Number of mismatches (closest documents on average)
    n_mismatch = int(n_docs * p_mismatch)
    
    # Initialize all as matches
    filter_matches = np.ones(n_docs)
    
    # The closest n_mismatch documents (on average) are mismatches
    n_deterministic = int(n_mismatch * correlation_strength)
    n_random = n_mismatch - n_deterministic
    
    # Deterministic: closest documents are mismatches
    filter_matches[sorted_indices[:n_deterministic]] = 0
    
    # Random: sample from the next chunk for some noise
    if n_random > 0:
        random_pool = sorted_indices[n_deterministic:n_deterministic + 2*n_random]
        random_mismatches = rng.choice(
            random_pool, 
            size=min(n_random, len(random_pool)), 
            replace=False
        )
        filter_matches[random_mismatches] = 0
    
    return filter_matches


def assign_filters_independent(
    n_docs: int,
    p_mismatch: float,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Assign filter matches independently of distance (zero covariance).
    """
    mismatches = rng.random(n_docs) < p_mismatch
    return (~mismatches).astype(float)


def compute_filtered_distances(
    base_distances: np.ndarray,
    filter_matches: np.ndarray,
    penalty: float
) -> np.ndarray:
    """
    Apply penalty to distances for documents that don't match the filter.
    
    δ'(q, d) = δ(q, d) + α * (1 - match)
    """
    return base_distances + penalty * (1 - filter_matches)


def compute_stability_metrics(distances: np.ndarray) -> Tuple[float, float]:
    """Compute stability ratio and relative variance."""
    if distances.size == 0:
        return 1.0, 0.0
    
    d_min = np.min(distances)
    d_max = np.max(distances)
    mean_d = np.mean(distances)
    var_d = np.var(distances)
    
    if d_min < 1e-10:
        d_min = 1e-10
    if mean_d < 1e-10:
        mean_d = 1e-10
    
    stability_ratio = d_max / d_min
    relative_variance = var_d / (mean_d ** 2)
    
    return stability_ratio, relative_variance


def compute_covariance(distances: np.ndarray, filter_matches: np.ndarray) -> float:
    """Compute covariance between distance and filter mismatch indicator."""
    mismatch_indicator = 1 - filter_matches
    return np.cov(distances, mismatch_indicator)[0, 1]


def run_single_experiment(
    rng: np.random.Generator,
    dim: int,
    n_docs: int,
    n_queries: int,
    p_mismatch: float,
    penalty_multipliers: Dict[str, float],
    use_negative_covariance: bool = True
) -> Dict[str, Dict[str, List[float]]]:
    """Run filtered search experiment for a single dimension with FIXED database."""
    
    results = {name: {"ratio": [], "relvar": []} for name in penalty_multipliers}
    covariances = []
    
    docs = generate_unstable_vectors(rng, n_docs, dim)
    
    queries = generate_unstable_vectors(rng, n_queries, dim)
    
    if use_negative_covariance:
        filter_matches = assign_filters_with_negative_covariance_fixed_db(
            docs, queries, p_mismatch, rng
        )
    else:
        filter_matches = assign_filters_independent(n_docs, p_mismatch, rng)
    
    # Compute distance scale from first few queries for penalty scaling
    sample_distances = cosine_distance_batch(queries[:min(10, n_queries)], docs)
    distance_scale = np.std(sample_distances)
    
    # Process each query against the fixed database
    for q_idx in range(n_queries):
        query = queries[q_idx]
        
        # Compute base cosine distances to fixed database
        base_distances = cosine_distance(query, docs)
        
        # Track covariance for this query
        cov = compute_covariance(base_distances, filter_matches)
        covariances.append(cov)
        
        # Compute filtered distances for each penalty setting
        for name, multiplier in penalty_multipliers.items():
            if name == "Large Penalty":
                # Large penalty should be larger than the theoretical threshold of 8
                penalty = 8.1
            else:
                # Small/no penalty scales with distance variance
                penalty = multiplier * distance_scale
            
            filtered_distances = compute_filtered_distances(
                base_distances, filter_matches, penalty
            )
            ratio, relvar = compute_stability_metrics(filtered_distances)
            results[name]["ratio"].append(ratio)
            results[name]["relvar"].append(relvar)
    
    results["_covariance"] = np.mean(covariances)
    return results


def run_experiment(
    dimensions: List[int],
    n_docs: int = 1000,
    n_queries: int = 50,
    p_mismatch: float = 0.5,
    n_trials: int = 10,
    seed: int = 42
) -> Tuple[Dict, List[int], float]:
    """Run the full filtered search stability experiment with fixed databases."""
    
    rng = np.random.default_rng(seed)
    
    # Estimate Δ_max
    delta_max = 2.0
    
    # Compute theoretical threshold from Theorem 6.3
    threshold = 2 * delta_max / (1 - p_mismatch)
    
    # Define penalty multipliers (relative to distance standard deviation)
    penalty_multipliers = {
        "No Penalty": 0.0,
        "Small Penalty": 0.5,  # 0.5x distance std dev
        "Large Penalty": 1.0,  # placeholder, handled specially
    }
    
    print("=" * 85)
    print("Filtered Vector Search Stability Experiment (Fixed Database)")
    print("=" * 85)
    print(f"\nParameters:")
    print(f"  Documents in database: {n_docs}")
    print(f"  Queries per trial: {n_queries}")
    print(f"  Filter mismatch probability: {p_mismatch}")
    print(f"  Δ_max (max base distance): {delta_max}")
    print(f"  Theorem 6.3 threshold: α > {threshold:.4f}")
    print(f"\nPenalty settings:")
    print(f"  No Penalty: 0")
    print(f"  Small Penalty: 0.5 × std(distances) - scales with dimension")
    print(f"  Large Penalty: 12.0 (fixed, > threshold)")
    print(f"\nUsing NEGATIVE covariance construction:")
    print(f"  Documents close to queries ON AVERAGE are more likely to NOT match")
    print(f"  Filter attributes are assigned ONCE per database, not per query")
    print()
    
    # Initialize metrics storage
    metrics = {name: {"ratio": [], "relvar": []} for name in penalty_multipliers}
    avg_covariances = []
    
    print(f"{'Dim':>6} | {'Cov(δ,1_c)':>10} | ", end="")
    for name in penalty_multipliers:
        print(f"{name:>15} | ", end="")
    print()
    print("-" * 85)
    
    for dim in tqdm(dimensions, desc="Running experiment"):
        dim_results = {name: {"ratio": [], "relvar": []} for name in penalty_multipliers}
        dim_covariances = []
        
        for _ in range(n_trials):
            trial_results = run_single_experiment(
                rng=rng,
                dim=dim,
                n_docs=n_docs,
                n_queries=n_queries,
                p_mismatch=p_mismatch,
                penalty_multipliers=penalty_multipliers,
                use_negative_covariance=True
            )
            
            dim_covariances.append(trial_results["_covariance"])
            
            for name in penalty_multipliers:
                dim_results[name]["ratio"].extend(trial_results[name]["ratio"])
                dim_results[name]["relvar"].extend(trial_results[name]["relvar"])
        
        avg_cov = np.mean(dim_covariances)
        avg_covariances.append(avg_cov)
        
        print(f"{dim:>6} | {avg_cov:>10.6f} | ", end="")
        for name in penalty_multipliers:
            avg_ratio = np.mean(dim_results[name]["ratio"])
            avg_relvar = np.mean(dim_results[name]["relvar"])
            metrics[name]["ratio"].append(avg_ratio)
            metrics[name]["relvar"].append(avg_relvar)
            print(f"{avg_ratio:>7.2f} / {avg_relvar:.4f} | ", end="")
        print()
    
    print(f"\nAverage Cov(δ, 1_c) across dimensions: {np.mean(avg_covariances):.6f}")
    print("(Negative covariance confirms closer docs are more likely to mismatch)")
    
    return metrics, dimensions, threshold


def print_analysis(metrics: Dict, threshold: float):
    """Print analysis of the experimental results."""
    print()
    print("=" * 85)
    print("Analysis")
    print("=" * 85)
    print()
    print(f"Theorem 6.3 threshold: α > {threshold:.4f}")
    print()
    
    for name in metrics:
        ratios = metrics[name]["ratio"]
        relvars = metrics[name]["relvar"]
        
        ratio_decay = ratios[-1] / ratios[0] if ratios[0] > 0 else 0
        relvar_decay = relvars[-1] / relvars[0] if relvars[0] > 0 else 0
        
        # Stable if: ratio stays well above 1 AND relvar doesn't collapse
        is_stable = (ratios[-1] > 3) and (relvar_decay > 0.3)
        
        status = "STABLE ✓" if is_stable else "UNSTABLE ✗"
        
        print(f"{name}:")
        print(f"  Stability ratio: {ratios[0]:.2f} → {ratios[-1]:.2f} (decay: {ratio_decay:.3f})")
        print(f"  Relative variance: {relvars[0]:.4f} → {relvars[-1]:.6f} (decay: {relvar_decay:.3f})")
        print(f"  Status: {status}")
        print()
    
    print("Key insight from Theorem 6.3:")
    print("  With negative Cov(δ, 1_c), small penalties can reduce variance because")
    print("  they shift close-but-mismatching documents away, compressing the distribution.")
    print("  Only large penalties (above threshold) create enough separation to induce stability.")


def plot_results(
    metrics: Dict,
    dimensions: List[int],
    threshold: float,
    output_path: str,
):
    """Two-panel figure: stability ratio and relative variance vs. dimension.

    House style (figures4papers): sans-serif type, no grid, top/right spines
    off, frameless legend. Blue marks the setting the theorem predicts to be
    stable (large penalty); reds mark the unstable contrasts.
    """
    apply_publication_style(FigureStyle(font_size=18, axes_linewidth=2))
    fig, (ax1, ax2) = create_subplots(1, 2, figsize=(13, 4.8))

    order = ["Large Penalty", "Small Penalty", "No Penalty"]
    labels = ["Large penalty", "Small penalty", "No penalty"]
    colors = [PALETTE["blue_main"], PALETTE["red_soft"], PALETTE["red_strong"]]
    markers = ["o", "s", "^"]

    # Panel 1: stability ratio (log-log), with the instability threshold at 1.
    make_trend(
        ax1,
        dimensions,
        [metrics[name]["ratio"] for name in order],
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
    ax1.legend(loc="upper right")

    # Panel 2: relative variance (log-log). Clamp exact zeros for the log axis.
    make_trend(
        ax2,
        dimensions,
        [[max(v, 1e-6) for v in metrics[name]["relvar"]] for name in order],
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
    ax2.legend(loc="lower left")

    saved = finalize_figure(fig, output_path, formats=["png", "pdf"], dpi=300)
    print("\nPlot saved to: " + ", ".join(str(s) for s in saved))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filtered search stability experiment (Theorem 6.3).")
    parser.add_argument("--results", default=str(RESULTS_PATH), help="JSON file to write/read experiment results")
    parser.add_argument("--figure", default=str(FIGURE_PATH), help="Output figure stem (PNG + PDF are written)")
    parser.add_argument("--plot-only", action="store_true", help="Skip the experiment and re-plot saved results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dimensions = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]

    if args.plot_only:
        saved = load_results(args.results)
        metrics, dims, threshold = saved["metrics"], saved["dimensions"], saved["threshold"]
    else:
        params = dict(n_docs=10000, n_queries=100, p_mismatch=0.5, n_trials=1, seed=42)
        metrics, dims, threshold = run_experiment(dimensions=dimensions, **params)
        save_results(
            {"metrics": metrics, "dimensions": dims, "threshold": threshold, "params": params},
            args.results,
        )
        print(f"Results saved to: {args.results}")
        print_analysis(metrics, threshold)

    plot_results(metrics=metrics, dimensions=dims, threshold=threshold, output_path=args.figure)
