import argparse
import sys
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from plotting import (  # noqa: E402
    PALETTE,
    FigureStyle,
    add_reference_line,
    add_shared_legend,
    apply_publication_style,
    create_subplots,
    finalize_figure,
    format_log_axis,
    load_results,
    make_trend,
    save_results,
)

RESULTS_PATH = ROOT / "results" / "sparse_stability.json"
FIGURE_DIR = ROOT / "figures"

# The four regimes of the synthetic generator. Blue marks the regime the
# theorem predicts to be stable; reds and gray mark the three contrasts.
SCENARIOS = [
    {
        "name": "Both CoI and Overlap",
        "label": "Both CoI and overlap",
        "enforce_concentration": True,
        "enforce_overlap": True,
        "color": PALETTE["blue_main"],
        "marker": "o",
    },
    {
        "name": "CoI Only",
        "label": "CoI only",
        "enforce_concentration": True,
        "enforce_overlap": False,
        "color": PALETTE["red_strong"],
        "marker": "s",
    },
    {
        "name": "Overlap Only",
        "label": "Overlap only",
        "enforce_concentration": False,
        "enforce_overlap": True,
        "color": PALETTE["red_soft"],
        "marker": "^",
    },
    {
        "name": "Neither",
        "label": "Neither",
        "enforce_concentration": False,
        "enforce_overlap": False,
        "color": PALETTE["gray_mid"],
        "marker": "D",
    },
]


class RealisticSparseVectorGenerator:
    """Generate sparse vectors that better match learned sparse embeddings."""

    def __init__(
        self,
        dim: int,
        avg_sparsity: int,
        alpha_p: float,
        overlap_prob: float,
        gamma: float,
        p: int = 2,
        mode: str = "stable",
        enforce_concentration: bool = True,
        enforce_overlap: bool = True,
        head_block: str = "all",
    ):
        """
        :param dim: Vector dimensionality
        :param avg_sparsity: Average number of non-zeros
        :param alpha_p: Concentration parameter
        :param overlap_prob: Probability of semantic overlap (pi)
        :param gamma: Minimum overlap mass
        :param p: Lp norm
        :param mode: 'stable' or 'unstable'
        :param enforce_concentration: If False, use uniform sampling
        :param enforce_overlap: If False, avoid semantic overlap
        :param head_block: 'first_half' or 'second_half' or 'all'
        """
        self.dim = dim
        self.avg_sparsity = avg_sparsity
        self.alpha_p = alpha_p
        self.overlap_prob = overlap_prob
        self.gamma = gamma
        self.p = p
        self.mode = mode
        self.enforce_concentration = enforce_concentration
        self.enforce_overlap = enforce_overlap

        if mode == "stable":
            # Create importance distribution
            if enforce_concentration:
                self.importance_power = 1.5
                ranks = np.arange(1, dim + 1)
                importance_scores = 1.0 / (ranks**self.importance_power)

                if head_block == "first_half":
                    mask = np.zeros(dim)
                    mask[: dim // 2] = 1.0
                    importance_scores *= mask + 1e-12

                elif head_block == "second_half":
                    mask = np.zeros(dim)
                    mask[dim // 2 :] = 1.0
                    importance_scores *= mask + 1e-12

                self.dim_importance = importance_scores / importance_scores.sum()
            else:
                # Uniform importance
                self.dim_importance = np.ones(dim) / dim

            self.neighborhood_size = max(20, dim // 100)

    def sample_active_dimensions(
        self,
        n_dims: int,
        preference_center: int = None,
        preference_strength: float = 0.0,
    ) -> np.ndarray:
        """Sample which dimensions are active."""
        if self.mode == "unstable" or not self.enforce_concentration:
            return np.random.choice(self.dim, n_dims, replace=False)

        probs = self.dim_importance.copy()

        if preference_center is not None and preference_strength > 0:
            neighborhood_bonus = np.zeros(self.dim)
            start = max(0, preference_center - self.neighborhood_size // 2)
            end = min(self.dim, preference_center + self.neighborhood_size // 2)
            neighborhood_bonus[start:end] = 1.0

            if neighborhood_bonus.sum() > 0:
                neighborhood_bonus = neighborhood_bonus / neighborhood_bonus.sum()
                probs = (
                    1 - preference_strength
                ) * probs + preference_strength * neighborhood_bonus

        selected = np.random.choice(
            self.dim, size=min(n_dims, self.dim), replace=False, p=probs
        )
        return selected

    def generate_concentrated_vector(
        self, semantic_center: int = None, force_overlap_dims: set = None
    ) -> np.ndarray:
        """Generate a sparse vector with optional concentration."""
        vec = np.zeros(self.dim)

        actual_sparsity = max(
            10, int(np.random.normal(self.avg_sparsity, self.avg_sparsity * 0.2))
        )
        actual_sparsity = min(actual_sparsity, self.dim)

        if self.mode == "stable":
            if semantic_center is None:
                top_portion = max(10, self.dim // 10)
                semantic_center = np.random.choice(top_portion)

            if force_overlap_dims is not None and len(force_overlap_dims) > 0:
                overlap_list = list(force_overlap_dims)
                n_forced = min(len(overlap_list), actual_sparsity)
                forced_dims = overlap_list[:n_forced]
                remaining_needed = actual_sparsity - n_forced

                if remaining_needed > 0:
                    available = list(set(range(self.dim)) - force_overlap_dims)

                    if self.enforce_concentration:
                        probs = self.dim_importance[available].copy()
                        for i, dim in enumerate(available):
                            dist = abs(dim - semantic_center)
                            if dist < self.neighborhood_size:
                                probs[i] *= 2.0
                        probs = probs / probs.sum()
                    else:
                        probs = np.ones(len(available)) / len(available)

                    additional = np.random.choice(
                        available,
                        size=min(remaining_needed, len(available)),
                        replace=False,
                        p=probs,
                    )
                    active_dims = np.concatenate([forced_dims, additional])
                else:
                    active_dims = np.array(forced_dims)
            else:
                if self.enforce_concentration:
                    active_dims = self.sample_active_dimensions(
                        actual_sparsity,
                        preference_center=semantic_center,
                        preference_strength=0.7,
                    )
                else:
                    active_dims = np.random.choice(
                        self.dim, actual_sparsity, replace=False
                    )

            n_active = len(active_dims)

            if self.enforce_concentration:
                dim_importances = self.dim_importance[active_dims]
                sorted_indices = np.argsort(-dim_importances)
                sorted_dims = active_dims[sorted_indices]

                ranks = np.arange(1, n_active + 1)
                weights = 1.0 / (ranks**1.5)
                weights = weights / weights.sum()

                for i, dim_idx in enumerate(sorted_dims):
                    vec[dim_idx] = (weights[i] * self.alpha_p) ** (1 / self.p)

                remaining_dims = list(set(range(self.dim)) - set(active_dims))
                if len(remaining_dims) > 0:
                    noise_dims = np.random.choice(
                        remaining_dims, size=min(10, len(remaining_dims)), replace=False
                    )
                    noise_budget = (1 - self.alpha_p) ** (1 / self.p)
                    noise_vals = np.random.exponential(1e-7, len(noise_dims))
                    noise_vals = noise_vals / (noise_vals.sum() + 1e-10) * noise_budget
                    vec[noise_dims] = noise_vals
            else:
                weights = np.random.exponential(1.0, n_active)
                vec[active_dims] = weights
        else:
            active_dims = np.random.choice(self.dim, actual_sparsity, replace=False)
            weights = np.random.exponential(1.0, actual_sparsity)
            vec[active_dims] = weights

        lp_norm = np.sum(np.abs(vec) ** self.p) ** (1 / self.p)
        if lp_norm > 1e-10:
            vec = vec / lp_norm
        else:
            vec[0] = 1.0

        return vec

    def generate_query_document_pair(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate query-document pair with controlled overlap."""
        if self.mode == "stable":
            top_portion = max(10, self.dim // 10)
            semantic_center = np.random.choice(top_portion)

            query = self.generate_concentrated_vector(semantic_center=semantic_center)

            query_powers = np.abs(query) ** self.p
            query_nonzero = np.nonzero(query)[0]
            query_top_k = min(self.avg_sparsity, len(query_nonzero))

            if query_top_k > 0:
                query_top_indices = np.argsort(query_powers)[-query_top_k:]
            else:
                query_top_indices = []

            has_overlap = np.random.random() < self.overlap_prob

            if self.enforce_overlap and has_overlap and len(query_top_indices) > 0:
                n_shared = max(
                    int(query_top_k * 0.7), int(query_top_k * self.gamma / self.alpha_p)
                )
                n_shared = min(n_shared, query_top_k)

                query_top_sorted = query_top_indices[
                    np.argsort(-query_powers[query_top_indices])
                ]
                shared_dims = set(query_top_sorted[:n_shared])

                doc = self.generate_concentrated_vector(
                    semantic_center=semantic_center, force_overlap_dims=shared_dims
                )
            else:
                min_distance = max(1, self.dim // 20)
                valid_centers = [
                    c
                    for c in range(top_portion)
                    if abs(c - semantic_center) >= min_distance
                ]

                if len(valid_centers) == 0:
                    other_center = np.random.choice(top_portion)
                else:
                    other_center = np.random.choice(valid_centers)

                doc = self.generate_concentrated_vector(semantic_center=other_center)

            lp_norm = np.sum(np.abs(doc) ** self.p) ** (1 / self.p)
            if lp_norm > 1e-10:
                doc = doc / lp_norm
            else:
                doc[0] = 1.0
        else:
            query = self.generate_concentrated_vector()
            doc = self.generate_concentrated_vector()

        return query, doc

    def verify_concentration(self, vec: np.ndarray) -> float:
        """Verify concentration of top-k dimensions."""
        vec_powers = np.abs(vec) ** self.p
        sorted_powers = np.sort(vec_powers)[::-1]

        nonzero = np.nonzero(vec)[0]
        top_k = min(self.avg_sparsity, len(nonzero))

        if top_k > 0:
            top_k_mass = np.sum(sorted_powers[:top_k])
        else:
            top_k_mass = 0.0

        return top_k_mass

    def compute_overlap(self, query: np.ndarray, doc: np.ndarray) -> float:
        """Compute overlap mass."""
        query_powers = np.abs(query) ** self.p
        doc_powers = np.abs(doc) ** self.p

        query_nonzero = np.nonzero(query)[0]
        doc_nonzero = np.nonzero(doc)[0]

        query_top_k = min(self.avg_sparsity, len(query_nonzero))
        doc_top_k = min(self.avg_sparsity, len(doc_nonzero))

        if query_top_k > 0 and doc_top_k > 0:
            query_top = set(np.argsort(query_powers)[-query_top_k:])
            doc_top = set(np.argsort(doc_powers)[-doc_top_k:])

            overlap_indices = query_top.intersection(doc_top)
            overlap_mass = sum(
                min(query_powers[i], doc_powers[i]) for i in overlap_indices
            )
        else:
            overlap_mass = 0.0

        return overlap_mass


def compute_distances_batch(query, documents, p):
    """Vectorized distance computation."""
    distances = np.sum(np.abs(query - documents) ** p, axis=1) ** (1 / p)
    return distances


class SparseSearchStabilityExperiment:
    """Validate theorem with different overlap regimes."""

    def __init__(self, p: int = 2, use_sampling: bool = True):
        self.p = p
        self.use_sampling = use_sampling

    def compute_stability_metrics(
        self, queries: np.ndarray, documents: np.ndarray, sample_size: int = 1000
    ) -> Tuple[float, float]:
        """Compute stability metrics."""
        n_queries = queries.shape[0]
        n_docs = documents.shape[0]

        min_distances = []
        max_distances = []
        sampled_distances = []

        for q in queries:
            dists = compute_distances_batch(q, documents, self.p)
            min_distances.append(np.min(dists))
            max_distances.append(np.max(dists))

            if self.use_sampling and n_docs > sample_size:
                idx = np.random.choice(n_docs, sample_size, replace=False)
                sampled_distances.extend(dists[idx])
            else:
                sampled_distances.extend(dists)

        min_distances = np.array(min_distances)
        max_distances = np.array(max_distances)

        stability_ratios = max_distances / (min_distances + 1e-10)
        avg_stability_ratio = np.mean(stability_ratios)

        sampled_distances = np.array(sampled_distances)
        mean_dist = np.mean(sampled_distances)
        var_dist = np.var(sampled_distances)
        rel_var = var_dist / (mean_dist**2 + 1e-10)

        return avg_stability_ratio, rel_var

    def run_overlap_sensitivity_experiment(
        self,
        dimensions: List[int],
        overlap_probs: List[float],
        n_queries: int = 100,
        n_docs: int = 10000,
        avg_sparsity: int = 30,
        alpha_p: float = 0.80,
        gamma: float = 0.50,
    ) -> Dict:
        """Run experiment across different π values."""

        scenarios = SCENARIOS

        all_results = {}

        for pi in overlap_probs:
            print(f"\n{'=' * 70}")
            print(f"TESTING OVERLAP PROBABILITY π = {pi}")
            print(f"{'=' * 70}")

            results = {
                "dimensions": dimensions,
                "pi": pi,
                "scenarios": {
                    s["name"]: {
                        "ratios": [],
                        "relvars": [],
                        "concentrations": [],
                        "overlaps": [],
                    }
                    for s in scenarios
                },
            }

            for dim in dimensions:
                print(f"\nDimension: {dim}")

                for scenario in scenarios:
                    if scenario["name"] == "CoI Only":
                        gen_q = RealisticSparseVectorGenerator(
                            dim=dim,
                            avg_sparsity=avg_sparsity,
                            alpha_p=alpha_p,
                            overlap_prob=pi,
                            gamma=gamma,
                            p=self.p,
                            mode="stable",
                            enforce_concentration=True,
                            enforce_overlap=False,
                            head_block="first_half",
                        )
                        gen_d = RealisticSparseVectorGenerator(
                            dim=dim,
                            avg_sparsity=avg_sparsity,
                            alpha_p=alpha_p,
                            overlap_prob=pi,
                            gamma=gamma,
                            p=self.p,
                            mode="stable",
                            enforce_concentration=True,
                            enforce_overlap=False,
                            head_block="second_half",
                        )
                    else:
                        gen_q = gen_d = RealisticSparseVectorGenerator(
                            dim=dim,
                            avg_sparsity=avg_sparsity,
                            alpha_p=alpha_p,
                            overlap_prob=pi,
                            gamma=gamma,
                            p=self.p,
                            mode="stable",
                            enforce_concentration=scenario["enforce_concentration"],
                            enforce_overlap=scenario["enforce_overlap"],
                            head_block="all",
                        )

                    queries, docs = [], []
                    conc, overlap = [], []

                    for _ in range(n_queries):
                        q = gen_q.generate_concentrated_vector()
                        d = gen_d.generate_concentrated_vector()
                        queries.append(q)
                        docs.append(d)
                        conc.append(gen_q.verify_concentration(q))
                        overlap.append(gen_q.compute_overlap(q, d))

                    for _ in range(n_docs - n_queries):
                        docs.append(gen_d.generate_concentrated_vector())

                    queries = np.array(queries)
                    docs = np.array(docs)

                    ratio, relvar = self.compute_stability_metrics(queries, docs)

                    results["scenarios"][scenario["name"]]["ratios"].append(ratio)
                    results["scenarios"][scenario["name"]]["relvars"].append(relvar)
                    results["scenarios"][scenario["name"]]["concentrations"].append(
                        np.mean(conc)
                    )
                    results["scenarios"][scenario["name"]]["overlaps"].append(
                        np.mean(overlap)
                    )

                    print(
                        f"  [{scenario['name']}] Ratio={ratio:.2f}, RelVar={relvar:.5f}, "
                        f"CoI={np.mean(conc):.3f}, OoI={np.mean(overlap):.3f}"
                    )

            all_results[pi] = results

        return all_results


def plot_single_pi_result(results: Dict, output_stem: Path):
    """Two-panel figure for one overlap probability: stability ratio and relative variance."""
    apply_publication_style(FigureStyle(font_size=18, axes_linewidth=2))
    fig, (ax1, ax2) = create_subplots(1, 2, figsize=(13, 4.8))
    dims = results["dimensions"]

    labels = [sc["label"] for sc in SCENARIOS]
    colors = [sc["color"] for sc in SCENARIOS]
    markers = [sc["marker"] for sc in SCENARIOS]

    make_trend(
        ax1,
        dims,
        [results["scenarios"][sc["name"]]["ratios"] for sc in SCENARIOS],
        labels,
        colors=colors,
        markers=markers,
        xlabel="Dimension",
        ylabel=r"Stability ratio ($d_{\max}/d_{\min}$)",
        xscale="log",
    )
    add_reference_line(ax1, 1.0, label="Instability threshold")
    format_log_axis(ax1, "x")

    make_trend(
        ax2,
        dims,
        [results["scenarios"][sc["name"]]["relvars"] for sc in SCENARIOS],
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

    add_shared_legend(fig, ax1)
    saved = finalize_figure(fig, output_stem, formats=["png", "pdf"], dpi=300, bbox_inches="tight")
    print("  Saved: " + ", ".join(str(p) for p in saved))


def _plot_metric_comparison(
    all_results: Dict,
    metric: str,
    ylabel: str,
    output_stem: Path,
    yscale: str | None = None,
    reference: float | None = None,
):
    """Single panel: one metric vs. dimension, one curve per regime.

    Colours follow the house palette (blue = the regime the theorem predicts
    to be stable, reds = one property only, gray = neither). When the results
    contain several overlap probabilities, each pi gets its own line style.
    """
    apply_publication_style(FigureStyle(font_size=18, axes_linewidth=2))
    fig, (ax,) = create_subplots(1, 1, figsize=(8, 6))
    pi_values = sorted(all_results.keys(), key=float)
    pi_linestyles = ["-", "--", ":", "-."]
    dims = all_results[pi_values[0]]["dimensions"]

    series, labels, colors, markers, linestyles = [], [], [], [], []
    for j, pi in enumerate(pi_values):
        for sc in SCENARIOS:
            series.append(all_results[pi]["scenarios"][sc["name"]][metric])
            label = sc["label"]
            if len(pi_values) > 1:
                label += rf" ($\pi = {float(pi):g}$)"
            labels.append(label)
            colors.append(sc["color"])
            markers.append(sc["marker"])
            linestyles.append(pi_linestyles[j % len(pi_linestyles)])

    lines = make_trend(
        ax,
        dims,
        series,
        labels,
        colors=colors,
        markers=markers,
        linestyles=linestyles,
        xlabel="Dimension",
        ylabel=ylabel,
        xscale="log",
        yscale=yscale,
    )
    # "Overlap only" and "Neither" nearly coincide; stack the curves so the
    # gray "Neither" line sits underneath and the later regimes stay visible.
    for i, line in enumerate(lines):
        line.set_zorder(3 + len(lines) - i)
    if reference is not None:
        add_reference_line(ax, reference, label="Instability threshold")
    format_log_axis(ax, "x")
    if yscale == "log":
        format_log_axis(ax, "y")
    if len(pi_values) == 1:
        ax.set_title(rf"{ylabel} ($\pi = {float(pi_values[0]):g}$)", pad=10)
    else:
        ax.set_title(ylabel, pad=10)
    ax.legend(loc="best", ncol=1 if len(pi_values) == 1 else 2)

    saved = finalize_figure(fig, output_stem, formats=["png", "pdf"], dpi=300)
    print("  Saved: " + ", ".join(str(p) for p in saved))


def plot_pi_comparison(all_results: Dict, output_stem: Path):
    """Relative variance vs. dimension, one curve per regime (log-log)."""
    _plot_metric_comparison(all_results, "relvars", "Relative variance", output_stem, yscale="log")


def plot_ratio_comparison(all_results: Dict, output_stem: Path):
    """Stability ratio d_max/d_min vs. dimension, one curve per regime."""
    _plot_metric_comparison(all_results, "ratios", "Stability ratio", output_stem, reference=1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthetic sparse-embedding stability experiment (Theorem 7.4).")
    parser.add_argument("--results", default=str(RESULTS_PATH), help="JSON file to write/read experiment results")
    parser.add_argument("--figure-dir", default=str(FIGURE_DIR), help="Directory for the output figures")
    parser.add_argument("--plot-only", action="store_true", help="Skip the experiment and re-plot saved results")
    parser.add_argument(
        "--dimensions",
        type=int,
        nargs="+",
        default=[128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 131072],
        help="Dimensions to sweep. Peak memory is ~3 x n_docs x dim x 8 bytes per scenario "
        "(~31 GB at 131072), so trim this list on small machines.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    np.random.seed(42)

    dimensions = args.dimensions
    overlap_probs = [0.5]

    if args.plot_only:
        all_results = load_results(args.results)["all_results"]
    else:
        print("=" * 70)
        print("OVERLAP SENSITIVITY ANALYSIS")
        print("=" * 70)
        print(f"Dimensions: {dimensions}")
        print(f"Overlap probabilities (π): {overlap_probs}")
        print(f"Scenarios: Both CoI and Overlap, CoI Only, Overlap Only, Neither")
        print("=" * 70)

        params = dict(n_queries=100, n_docs=10000, avg_sparsity=30, alpha_p=0.83, gamma=0.20)
        experiment = SparseSearchStabilityExperiment(p=2, use_sampling=True)
        all_results = experiment.run_overlap_sensitivity_experiment(
            dimensions=dimensions, overlap_probs=overlap_probs, **params
        )
        all_results = {str(pi): res for pi, res in all_results.items()}
        save_results({"all_results": all_results, "params": params, "p": 2}, args.results)
        print(f"\nResults saved to: {args.results}")

    figure_dir = Path(args.figure_dir)
    for pi, results in all_results.items():
        plot_single_pi_result(results, figure_dir / f"sparse_stability_pi_{float(pi):.1f}")
    plot_pi_comparison(all_results, figure_dir / "sparse_pi_comparison")
    plot_ratio_comparison(all_results, figure_dir / "sparse_ratio_comparison")

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
