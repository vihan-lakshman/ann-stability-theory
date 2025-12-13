import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
import hnswlib
import faiss
import time
from typing import Tuple, Dict, List
from tqdm import tqdm


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

def plot_results(results_df: pd.DataFrame):
    """Generates styled plots from the results DataFrame."""
    print("Generating plots...")

    # Extract Data from DataFrame
    dimensions = sorted(results_df['Dimension'].unique())
    
    hnsw_data = results_df[results_df['Algorithm'] == 'HNSW'].sort_values('Dimension')
    ivf_data = results_df[results_df['Algorithm'] == 'IVF'].sort_values('Dimension')

    hnsw_stable = hnsw_data['Stable Recall'].tolist()
    hnsw_unstable = hnsw_data['Unstable Recall'].tolist()
    
    ivf_stable = ivf_data['Stable Recall'].tolist()
    ivf_unstable = ivf_data['Unstable Recall'].tolist()

    # Style Configuration
    plt.style.use("default")
    plt.rcParams.update({
        "font.size": 11,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "axes.linewidth": 1.2,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.grid": False,
        "grid.alpha": 0.4,
        "grid.linewidth": 0.8,
        "grid.color": "gray",
        "axes.axisbelow": True,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "figure.dpi": 600,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    palette = ["#2E86AB", "#F18F01"]

    # --- Plot 1: HNSW ---
    ax1.plot(dimensions, hnsw_stable, marker="o", label="Stable Recall", color=palette[0], markerfacecolor="white")
    ax1.plot(dimensions, hnsw_unstable, marker="s", label="Unstable Recall", color=palette[1], markerfacecolor="white")
    
    ax1.set_xlabel("Dimension", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Recall", fontsize=12, fontweight="bold")
    ax1.set_title("HNSW Recall", fontsize=13, fontweight="bold", pad=15)
    ax1.legend()
    ax1.set_xscale("log", base=2)
    ax1.set_ylim(0.2, 1.05) 

    # --- Plot 2: IVF ---
    ax2.plot(dimensions, ivf_stable, marker="o", label="Stable Recall", color=palette[0], markerfacecolor="white")
    ax2.plot(dimensions, ivf_unstable, marker="s", label="Unstable Recall", color=palette[1], markerfacecolor="white")
    
    ax2.set_xlabel("Dimension", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Recall", fontsize=12, fontweight="bold")
    ax2.set_title("IVF Recall", fontsize=13, fontweight="bold", pad=15)
    ax2.legend()
    ax2.set_xscale("log", base=2)
    ax2.set_ylim(0.2, 1.05)

    # --- Grid Formatting ---
    for ax in (ax1, ax2):
        ax.set_xticks(dimensions)
        ax.get_xaxis().set_major_formatter(ScalarFormatter())
        ax.minorticks_on()
        ax.grid(True, which="major", axis="both", alpha=0.4, linewidth=0.8)
        ax.grid(True, which="minor", axis="both", alpha=0.2, linewidth=0.5, linestyle=":")

    plt.tight_layout()
    filename = "hnsw_ivf_recall_styled.png"
    plt.savefig(filename, format="png")
    print(f"Plot saved to {filename}")
    # plt.show() # Uncomment if running in an environment with display support


def main():
    # Reduced parameters for quick testing; increase for full reproduction
    # n_docs=1000000, n_queries=1000 recommended for full scale
    results_df = run_full_experiment(
        dimensions=[4, 8, 16, 32, 64, 128, 256, 512, 1024],
        algorithms=['hnsw', 'ivf'],
        n_docs=1000000,   
        n_queries=1000,
        k=10
    )
    summarize_results(results_df)
    plot_results(results_df)


if __name__ == "__main__":
    main()
