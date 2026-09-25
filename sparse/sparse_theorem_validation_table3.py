"""Table 3: two-regime bound restricted to Q_alpha (kappa=48, R=8, alpha=0.999, p=2).

Reads results/two_regime/*_pairs.npz from ``sparse_theorem_validation.py --save-pairs``.
For each dataset, reports the bound over all pairs and over the pairs whose query
satisfies C_q(kappa) >= alpha (the Q_alpha restriction used in the paper).

Run from the repo root: uv run python sparse/sparse_theorem_validation_table3.py
"""

import sys

import numpy as np

sys.path.insert(0, "sparse")
from sparse_theorem_validation_sweep import best_bound, gamma_min, theory_Y  # noqa: E402

P = 2
ALPHA = 0.999
PREFIX = "k48_R8_"
DATASETS = ["hotpotqa", "msmarco-v2.1", "natural-questions", "nfcorpus", "trec-covid"]


def main():
    Y, g_min = theory_Y(ALPHA, P), gamma_min(ALPHA, P)
    for ds in DATASETS:
        z = np.load(f"results/two_regime/{ds}_pairs.npz")
        d = z["delta"]
        S, K, Cq, Cd = (z[PREFIX + k] for k in ("S", "K", "Cq", "Cd"))
        for name, mask in [("all", np.ones_like(d, bool)), ("Q_a", Cq >= ALPHA)]:
            dd = d[mask]
            relvar = dd.var() / dd.mean() ** 2
            in_B = (K[mask] == 0) & (Cd[mask] >= ALPHA) & (Cq[mask] >= ALPHA)
            b = best_bound(S[mask], in_B, P, Y, g_min)
            print(
                f"{ds:18s} {name:4s} n={mask.sum():8d} rho={np.mean(Cd[mask] >= ALPHA):.3f} "
                f"beta={b['beta']:.3f} gamma={b['gamma']:.3f} pi={b['pi']:.2e} nA={b['n_A']:6d} "
                f"gap={b['gap']:.3f} bound={b['bound']:.2e} relvar={relvar:.2e} "
                f"frac={100 * b['bound'] / relvar:.5f}% "
                # higher-precision copies for the paper table
                f"bnd={b['bound']:.4e} pi4={b['pi']:.4e} rvx={relvar:.4e} "
                f"g4={b['gamma']:.4f} gap4={b['gap']:.4f}"
            )


if __name__ == "__main__":
    main()
