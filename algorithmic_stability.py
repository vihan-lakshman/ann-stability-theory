import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import hnswlib
import faiss
import time
from typing import Tuple, Dict, List
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from plotting import (  # noqa: E402
    PALETTE,
    FigureStyle,
    apply_publication_style,
    create_subplots,
    finalize_figure,
    format_log_axis,
    load_results,
    make_trend,
    save_results,
)

RESULTS_PATH = ROOT / "results" / "hnsw_ivf_recall.json"
FIGURE_PATH = ROOT / "figures" / "hnsw_ivf_recall"


class SearchAlgorithm:
    """Base class for search algorithms."""
    def __init__(self, docs: np.ndarray):
        self.docs = docs.astype(np.float32)
        self.dim = docs.shape[1]
        self.n_docs = docs.shape[0]
    def build_index(self): raise NotImplementedError
    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]: raise NotImplementedError


class HNSWAlgorithm(SearchAlgorithm):
    """HNSW implementation using hnswlib."""
    def __init__(self, docs: np.ndarray, ef_construction: int = 200, M: int = 16):
        super().__init__(docs)
        self.ef_construction, self.M, self.index = ef_construction, M, None
    def build_index(self):
        self.index = hnswlib.Index(space='l2', dim=self.dim)
        self.index.init_index(max_elements=self.n_docs, ef_construction=self.ef_construction, M=self.M)
        self.index.add_items(self.docs)
    def search(self, queries: np.ndarray, k: int, ef: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        if self.index is None: raise ValueError("Index not built.")
        self.index.set_ef(ef)
        return self.index.knn_query(queries.astype(np.float32), k=k)


class IVFAlgorithm(SearchAlgorithm):
    """IVF implementation using FAISS."""
    def __init__(self, docs: np.ndarray, nlist: int = None):
        super().__init__(docs)
        self.nlist = nlist if nlist else max(1, int(np.sqrt(self.n_docs)))
        self.index = None
    def build_index(self):
        quantizer = faiss.IndexFlatL2(self.dim)
        self.index = faiss.IndexIVFFlat(quantizer, self.dim, self.nlist)
        self.index.train(self.docs)
        self.index.add(self.docs)
    def search(self, queries: np.ndarray, k: int, nprobe: int = None) -> Tuple[np.ndarray, np.ndarray]:
        if self.index is None: raise ValueError("Index not built.")
        self.index.nprobe = nprobe if nprobe else min(self.nlist, max(1, self.nlist // 4))
        distances, indices = self.index.search(queries.astype(np.float32), k)
        return indices, distances


def generate_stable_dataset(n_docs: int, n_queries: int, dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generates a stable dataset with clear cluster structure."""
    np.random.seed(42)
    n_clusters = max(5, int(np.sqrt(n_docs) / 5))
    cluster_centers = np.random.randn(n_clusters, dim) * (dim**0.25)
    doc_assignments = np.random.randint(0, n_clusters, n_docs)
    query_assignments = np.random.randint(0, n_clusters, n_queries)
    doc_noise = np.random.randn(n_docs, dim) * 0.5
    query_noise = np.random.randn(n_queries, dim) * 0.2
    docs = cluster_centers[doc_assignments] + doc_noise
    queries = cluster_centers[query_assignments] + query_noise
    return docs, queries


def generate_unstable_dataset(n_docs: int, n_queries: int, dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generates an unstable dataset from an i.i.d. Gaussian distribution."""
    np.random.seed(42)
    docs = np.random.randn(n_docs, dim)
    queries = np.random.randn(n_queries, dim)
    return docs, queries


def get_algorithm(name: str, docs: np.ndarray) -> SearchAlgorithm:
    """Factory function for creating algorithm instances."""
    return {'hnsw': HNSWAlgorithm, 'ivf': IVFAlgorithm}[name.lower()](docs)

def evaluate_recall(pred_indices: np.ndarray, true_indices: np.ndarray) -> float:
    """Calculates recall@k."""
    k = true_indices.shape[1]
    recall = sum(len(set(p) & set(t)) / k for p, t in zip(pred_indices, true_indices))
    return recall / len(pred_indices)

def run_full_experiment(dimensions: List[int], algorithms: List[str], n_docs: int, n_queries: int, k: int = 10):
    results = []
    gt_cache = {}
    
    print("Pre-calculating ground truth for recall evaluation...")
    for dim in tqdm(dimensions, desc="Ground Truth"):
        stable_docs, stable_queries = generate_stable_dataset(n_docs, n_queries, dim)
        unstable_docs, unstable_queries = generate_unstable_dataset(n_docs, n_queries, dim)
        
        gt_stable_nn = faiss.IndexFlatL2(dim)
        gt_stable_nn.add(stable_docs.astype(np.float32))
        _, gt_stable_indices = gt_stable_nn.search(stable_queries.astype(np.float32), k)
        
        gt_unstable_nn = faiss.IndexFlatL2(dim)
        gt_unstable_nn.add(unstable_docs.astype(np.float32))
        _, gt_unstable_indices = gt_unstable_nn.search(unstable_queries.astype(np.float32), k)
        
        gt_cache[dim] = {'stable': gt_stable_indices, 'unstable': gt_unstable_indices}

    print("\nRunning main experiment...")
    for dim in tqdm(dimensions, desc="Dimensions"):
        stable_docs, stable_queries = generate_stable_dataset(n_docs, n_queries, dim)
        unstable_docs, unstable_queries = generate_unstable_dataset(n_docs, n_queries, dim)
        
        for algo_name in algorithms:
            # Stable Dataset Evaluation
            stable_algo = get_algorithm(algo_name, stable_docs)
            stable_algo.build_index()
            stable_pred_indices, _ = stable_algo.search(stable_queries, k)
            stable_recall = evaluate_recall(stable_pred_indices, gt_cache[dim]['stable'])
            
            # Unstable Dataset Evaluation
            unstable_algo = get_algorithm(algo_name, unstable_docs)
            unstable_algo.build_index()
            unstable_pred_indices, _ = unstable_algo.search(unstable_queries, k)
            unstable_recall = evaluate_recall(unstable_pred_indices, gt_cache[dim]['unstable'])
            
            results.append({
                'Algorithm': algo_name.upper(),
                'Dimension': dim,
                'Stable Recall': stable_recall,
                'Unstable Recall': unstable_recall,
            })
    return pd.DataFrame(results)

def summarize_results(results_df: pd.DataFrame):
    """Prints a clean, formatted summary table of the results."""
    summary_df = results_df[['Algorithm', 'Dimension', 'Stable Recall', 'Unstable Recall']]
    
    pd.set_option('display.float_format', '{:.4f}'.format)
    pd.set_option('display.width', 1000)
    
    header = "PRACTICAL IMPLICATIONS OF STABILITY: SUMMARY TABLE"
    print("\n" + "="*len(header))
    print(header.center(len(header)))
    print("="*len(header))
    print(summary_df.to_string(index=False))
    print("="*len(header))

def plot_results(results_df: pd.DataFrame, output_path: str):
    """One panel per index (HNSW, IVF): recall@10 on stable vs. unstable data.

    Blue = stable dataset, red = unstable dataset; the x-axis is log2 since the
    dimensions are powers of two.
    """
    print("Generating plots...")
    apply_publication_style(FigureStyle(font_size=18, axes_linewidth=2))
    fig, axes = create_subplots(1, 2, figsize=(13, 4.8))

    labels = ["Stable dataset", "Unstable dataset"]
    colors = [PALETTE["blue_main"], PALETTE["red_strong"]]
    markers = ["o", "s"]

    for ax, algo in zip(axes, ["HNSW", "IVF"]):
        data = results_df[results_df["Algorithm"] == algo].sort_values("Dimension")
        dims = data["Dimension"].tolist()
        make_trend(
            ax,
            dims,
            [data["Stable Recall"].tolist(), data["Unstable Recall"].tolist()],
            labels,
            colors=colors,
            markers=markers,
            xlabel="Dimension",
            ylabel="Recall@10",
            xscale="log",
        )
        format_log_axis(ax, "x", ticks=dims)
        ax.set_ylim(0.0, 1.05)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_title(algo, pad=10)
        ax.legend(loc="lower left")

    saved = finalize_figure(fig, output_path, formats=["png", "pdf"], dpi=300)
    print("Plot saved to: " + ", ".join(str(p) for p in saved))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HNSW vs. IVF recall on stable and unstable datasets.")
    parser.add_argument("--n-docs", type=int, default=1_000_000, help="Database size (paper: 1,000,000)")
    parser.add_argument("--n-queries", type=int, default=1000, help="Number of queries (paper: 1000)")
    parser.add_argument("--k", type=int, default=10, help="Recall@k")
    parser.add_argument(
        "--dimensions", type=int, nargs="+", default=[4, 8, 16, 32, 64, 128, 256, 512, 1024], help="Dimensions to sweep"
    )
    parser.add_argument("--results", default=str(RESULTS_PATH), help="JSON file to write/read experiment results")
    parser.add_argument("--figure", default=str(FIGURE_PATH), help="Output figure stem (PNG + PDF are written)")
    parser.add_argument("--plot-only", action="store_true", help="Skip the experiment and re-plot saved results")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.plot_only:
        results_df = pd.DataFrame(load_results(args.results)["records"])
    else:
        results_df = run_full_experiment(
            dimensions=args.dimensions,
            algorithms=["hnsw", "ivf"],
            n_docs=args.n_docs,
            n_queries=args.n_queries,
            k=args.k,
        )
        save_results(
            {
                "records": results_df,
                "params": {"n_docs": args.n_docs, "n_queries": args.n_queries, "k": args.k},
            },
            args.results,
        )
        print(f"Results saved to: {args.results}")
        summarize_results(results_df)
    plot_results(results_df, args.figure)


if __name__ == "__main__":
    main()
