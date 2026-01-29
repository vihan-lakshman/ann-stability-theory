import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator, LogLocator
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from tqdm import tqdm
from typing import List, Tuple


class RealDataStabilityAnalyzer:
    """
    Analyzes the stability of real-world embeddings (e.g., ColBERT on MSMarco)
    by calculating stability metrics for multiple queries.
    """

    def __init__(self, model_name: str = "colbert-ir/colbertv2.0"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

    def _load_data(
        self, num_queries: int, num_docs: int
    ) -> Tuple[List[str], List[str]]:
        """Loads and prepares a subset of the MS Marco dataset."""
        print("Loading MS Marco v2.1 dataset...")
        # Load extra documents to ensure a diverse pool to select from
        dataset = load_dataset(
            "ms_marco", "v2.1", split=f"validation[:{num_queries + 100}]"
        )
        queries_text = dataset["query"][:num_queries]
        passages_raw = [
            text for item in dataset["passages"] for text in item["passage_text"]
        ]
        documents_text = list(dict.fromkeys(passages_raw))[:num_docs]
        print(
            f"Loaded {len(queries_text)} queries and {len(documents_text)} documents."
        )
        return queries_text, documents_text

    def _encode_texts(
        self, texts: List[str], max_length: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encodes a list of texts into normalized embeddings and an attention mask."""
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            embs = self.model(**inputs).last_hidden_state
        return torch.nn.functional.normalize(
            embs, p=2, dim=-1
        ), inputs.attention_mask.bool()

    def run_analysis(
        self, num_queries: int = 100, num_docs: int = 1000
    ) -> Tuple[np.ndarray, ...]:
        """Orchestrates the data loading, encoding, and stability calculations."""
        queries_text, documents_text = self._load_data(num_queries, num_docs)

        print("Encoding texts...")
        query_embs, query_masks = self._encode_texts(queries_text, max_length=128)
        doc_embs, doc_masks = self._encode_texts(documents_text, max_length=180)

        # Lists to store results for each query
        chamfer_ratios, avg_pool_ratios = [], []
        chamfer_relvars, avg_pool_relvars = [], []

        print("Calculating stability metrics for each query...")
        for i in tqdm(range(num_queries), desc="Analyzing Queries"):
            q_emb = query_embs[i][query_masks[i]]

            sim_matrix = torch.einsum("qe,dte->qdt", q_emb, doc_embs)

            # Chamfer
            dist_matrix = 1.0 - sim_matrix
            dist_matrix.masked_fill_(~doc_masks.unsqueeze(0), float("inf"))
            min_dist_per_doc, _ = dist_matrix.min(dim=-1)
            chamfer_dists = min_dist_per_doc.sum(dim=0)

            # Average Pooling
            sim_matrix.masked_fill_(
                ~doc_masks.unsqueeze(0), 0.0
            )  # Use 0 for similarity averaging
            total_sim = sim_matrix.sum(dim=(0, 2))
            num_pairs = q_emb.shape[0] * doc_masks.sum(dim=-1)
            avg_sim = torch.zeros_like(total_sim)
            valid_mask = num_pairs > 0
            avg_sim[valid_mask] = total_sim[valid_mask] / num_pairs[valid_mask]
            avg_pool_dists = 1.0 - avg_sim

            # --- Metric Calculation ---
            # Stability Ratio
            if torch.min(chamfer_dists) > 1e-9:
                chamfer_ratios.append(
                    torch.max(chamfer_dists) / torch.min(chamfer_dists)
                )
            if torch.min(avg_pool_dists) > 1e-9:
                avg_pool_ratios.append(
                    torch.max(avg_pool_dists) / torch.min(avg_pool_dists)
                )

            # Relative Variance
            if torch.mean(chamfer_dists) > 1e-9:
                chamfer_relvars.append(
                    torch.var(chamfer_dists) / (torch.mean(chamfer_dists) ** 2)
                )
            if torch.mean(avg_pool_dists) > 1e-9:
                avg_pool_relvars.append(
                    torch.var(avg_pool_dists) / (torch.mean(avg_pool_dists) ** 2)
                )

        return (
            torch.stack(chamfer_ratios).cpu().numpy(),
            torch.stack(avg_pool_ratios).cpu().numpy(),
            torch.stack(chamfer_relvars).cpu().numpy(),
            torch.stack(avg_pool_relvars).cpu().numpy(),
        )


def plot_results(results: Tuple[np.ndarray, ...]):
    """Generates and saves the two-panel boxplot figure."""
    chamfer_ratios, avg_pool_ratios, chamfer_relvars, avg_pool_relvars = results

    print("Analysis complete. Generating plots...")

    plt.style.use("default")
    plt.rcParams.update(
        {
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
        }
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    palette = ["#2E86AB", "#F18F01"]

    # Plot 1: Stability Ratio Distribution (narrower boxes)
    sns.boxplot(
        data=[chamfer_ratios, avg_pool_ratios],
        palette=palette,
        ax=ax1,
        whis=(5, 95),
        showfliers=False,
        width=0.35,  # thinner boxes
    )
    # ax1.axhline(y=1.0, color="red", linestyle="--", alpha=0.7, linewidth=1.5)
    ax1.set_xticks([0, 1], ["Chamfer Distance", "Average Pooling"], weight="bold")
    ax1.set_ylabel(
        "Stability Ratio ($d_{\\mathrm{max}}/d_{\\mathrm{min}}$)",
        fontsize=12,
        fontweight="bold",
    )
    ax1.set_title(
        "Distribution of Query Stability", fontsize=13, fontweight="bold", pad=15
    )

    # Plot 2: Relative Variance Distribution (log scale, narrower boxes)
    sns.boxplot(
        data=[chamfer_relvars, avg_pool_relvars],
        palette=palette,
        ax=ax2,
        whis=(5, 95),
        showfliers=False,
        width=0.35,  # thinner boxes
    )
    ax2.set_xticks([0, 1], ["Chamfer Distance", "Average Pooling"], weight="bold")
    ax2.set_ylabel("Relative Variance (Log Scale)", fontsize=12, fontweight="bold")
    ax2.set_title(
        "Distribution of Relative Variance", fontsize=13, fontweight="bold", pad=15
    )
    ax2.set_yscale("log")

    yticks = ax2.get_yticks()
    if len(yticks) > 0:
        y0 = yticks[0]  # smallest visible tick (e.g., 1e-3)
        ax2.axhline(
            y=y0,
            color="white",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
        )

    ax1.yaxis.set_minor_locator(AutoMinorLocator(4))

    ax2.yaxis.set_major_locator(LogLocator(base=10.0))
    ax2.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))

    for ax in (ax1, ax2):
        ax.minorticks_on()
        ax.grid(True, which="major", axis="both", alpha=0.4, linewidth=0.8)
        ax.grid(
            True,
            which="minor",
            axis="both",
            alpha=0.2,
            linewidth=0.5,
            linestyle=":",
        )

    plt.tight_layout()
    filename = "colbert_stability_analysis.png"
    plt.savefig(filename, format="png")
    print(f"Plot saved to {filename}")


def main():
    analyzer = RealDataStabilityAnalyzer()
    results = analyzer.run_analysis(num_queries=100, num_docs=1000)
    plot_results(results)


if __name__ == "__main__":
    main()
