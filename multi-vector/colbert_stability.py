import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from plotting import (  # noqa: E402
    PALETTE,
    FigureStyle,
    apply_publication_style,
    create_subplots,
    finalize_figure,
    format_log_axis,
    load_results,
    make_boxplot,
    save_results,
)

RESULTS_PATH = ROOT / "results" / "colbert_stability.json"
FIGURE_PATH = ROOT / "figures" / "colbert_stability"


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


def plot_results(results: dict, output_path: str):
    """Two-panel box plots: per-query stability ratio and relative variance.

    Blue = Chamfer distance (provably stable), red = average pooling. Boxes
    span the interquartile range, whiskers the 5th-95th percentiles.
    """
    print("Analysis complete. Generating plots...")
    apply_publication_style(FigureStyle(font_size=18, axes_linewidth=2))
    fig, (ax1, ax2) = create_subplots(1, 2, figsize=(12, 4.8))

    labels = ["Chamfer distance", "Average pooling"]
    colors = [PALETTE["blue_main"], PALETTE["red_strong"]]

    make_boxplot(
        ax1,
        [results["chamfer_ratios"], results["avg_pool_ratios"]],
        labels,
        colors=colors,
        ylabel=r"Stability ratio ($d_{\max}/d_{\min}$)",
    )
    ax1.set_ylim(bottom=0.9)

    make_boxplot(
        ax2,
        [results["chamfer_relvars"], results["avg_pool_relvars"]],
        labels,
        colors=colors,
        ylabel="Relative variance",
        yscale="log",
    )
    format_log_axis(ax2, "y")

    saved = finalize_figure(fig, output_path, formats=["png", "pdf"], dpi=300)
    print("Plot saved to: " + ", ".join(str(p) for p in saved))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ColBERT stability on MS MARCO: Chamfer vs. average pooling.")
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--num-docs", type=int, default=1000)
    parser.add_argument("--model", default="colbert-ir/colbertv2.0")
    parser.add_argument("--results", default=str(RESULTS_PATH), help="JSON file to write/read experiment results")
    parser.add_argument("--figure", default=str(FIGURE_PATH), help="Output figure stem (PNG + PDF are written)")
    parser.add_argument("--plot-only", action="store_true", help="Skip the experiment and re-plot saved results")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.plot_only:
        results = load_results(args.results)
    else:
        analyzer = RealDataStabilityAnalyzer(model_name=args.model)
        chamfer_ratios, avg_pool_ratios, chamfer_relvars, avg_pool_relvars = analyzer.run_analysis(
            num_queries=args.num_queries, num_docs=args.num_docs
        )
        results = {
            "chamfer_ratios": chamfer_ratios,
            "avg_pool_ratios": avg_pool_ratios,
            "chamfer_relvars": chamfer_relvars,
            "avg_pool_relvars": avg_pool_relvars,
            "params": {"num_queries": args.num_queries, "num_docs": args.num_docs, "model": args.model},
        }
        save_results(results, args.results)
        print(f"Results saved to: {args.results}")
    plot_results(results, args.figure)


if __name__ == "__main__":
    main()
