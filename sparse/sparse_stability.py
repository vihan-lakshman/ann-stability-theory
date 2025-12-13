import argparse
import os
from typing import List, Tuple

import numpy as np
from scipy.sparse import load_npz, csr_matrix
from numpy.random import default_rng


def nnz_stats_per_row(X: csr_matrix) -> dict:
    """
    Get some stats about the number of non-zero elements per row of the matrix X.
    """

    X = X.tocsr()
    nnz_per_row = np.diff(X.indptr)
    return {
        "mean": float(nnz_per_row.mean()),
        "median": float(np.median(nnz_per_row)),
        "p95": float(np.percentile(nnz_per_row, 95)),
        "p99": float(np.percentile(nnz_per_row, 99)),
    }


def compute_heads_and_concentration(
    X: csr_matrix,
    kappa: int,
    p: int,
) -> Tuple[np.ndarray, List[List[Tuple[int, float]]], np.ndarray]:
    """
    For each row x of the matrix X:
      - compute C_x(kappa), the concentration expression.
      - construct head T_x as list of (col, mass_p), where mass_p is |x_i|^p / ||x||_p^p
      - count how often each col appears in a head.

    :param X: sparse matrix of shape (n_rows, m)
    :param kappa: head size
    :param p: p-norm
    :return: concentrations: (n_rows,) array of concentration values
             heads: list of length n_rows, each entry is list[(col, mass_p)]
             head_counts: (m,) array, head_counts[j] = #times col j appears in a head
             This effectively captures how often each token appears in a head since each column
             represents a token.
    """
    X = X.tocsr()
    n_rows, m = X.shape

    data = X.data
    indices = X.indices
    indptr = X.indptr

    concentrations = np.zeros(n_rows, dtype=np.float64)
    heads: List[List[Tuple[int, float]]] = []
    head_counts = np.zeros(m, dtype=np.int64)

    # ||x||_p^p per row
    data_p = np.abs(data) ** p
    row_norm_p = np.add.reduceat(data_p, indptr[:-1])
    eps = 1e-12

    for r in range(n_rows):
        start, end = indptr[r], indptr[r + 1]
        if start == end:
            heads.append([])
            concentrations[r] = 0.0
            continue

        vals_p = data_p[start:end]
        cols = indices[start:end]

        k = min(kappa, vals_p.size)
        # top-k by |x_i|^p
        top_idx = np.argpartition(-vals_p, k - 1)[:k]
        top_vals_p = vals_p[top_idx]
        top_cols = cols[top_idx]

        denom = row_norm_p[r] + eps
        C = float(top_vals_p.sum() / denom)
        concentrations[r] = C

        # normalized p-mass inside the head
        mass_p = top_vals_p / denom
        head = list(zip(top_cols.tolist(), mass_p.tolist()))
        heads.append(head)

        head_counts[top_cols] += 1

    return concentrations, heads, head_counts


def compute_overlap_statistic(
    head_q: List[Tuple[int, float]],
    head_d: List[Tuple[int, float]],
) -> float:
    """
    Computes the overlap statistic as defined in the paper (Theorem 7.4).
    The statistic is defined by the sum of the minimum of the p-masses of the tokens in the intersection of the heads.

    S = sum_{i in T_q \\cap T_d} \\min{ q_i^p, d_i^p },
    where heads contain (col, mass_p) with mass_p already |x_i|^p / ||x||_p^p.

    Note that we do not need to raise the values to the power p since the arguments are already 
    raised to the power p.

    :param head_q: list of tuples (col, mass_p) for the query head
    :param head_d: list of tuples (col, mass_p) for the document head
    :return: overlap statistic S
    """
    if not head_q or not head_d:
        return 0.0
    dq = dict(head_q)
    dd = dict(head_d)
    inter = dq.keys() & dd.keys()
    if not inter:
        return 0.0
    return float(sum(min(dq[i], dd[i]) for i in inter))


def sample_overlap(
    heads_q: List[List[Tuple[int, float]]],
    heads_d: List[List[Tuple[int, float]]],
    n_pairs: int,
    seed: int = 0,
) -> np.ndarray:
    """
    Sample random (q,d) pairs and compute their overlap S.
    """
    rng = default_rng(seed)
    n_q = len(heads_q)
    n_d = len(heads_d)
    S = np.empty(n_pairs, dtype=np.float64)

    for t in range(n_pairs):
        iq = rng.integers(0, n_q)
        id = rng.integers(0, n_d)
        S[t] = compute_overlap_statistic(heads_q[iq], heads_d[id])

    return S


def theoretical_X(gamma: float, p: int) -> float:
    """
    X = (2 - 2*gamma)^{1/p} as defined in the paper (Theorem 7.4).

    :param gamma: Empirical lower bound for the overlap statistic.
    :param p: p-norm
    :return: X
    """
    return float((2.0 - 2.0 * gamma) ** (1.0 / p))


def theoretical_Y(alpha: float, rho: float, pi: float, p: int) -> float:
    """
    Y = [rho / (1 - pi)] * 2^{1/p} * (alpha^{1/p} - (1 - alpha)^{1/p})
    as defined in the paper (Theorem 7.4).

    :param alpha: Concentration threshold alpha
    :param rho: Empirical lower bound for the document concentration
    :param pi: Empirical lower bound for the fraction of positive overlaps.
    :param p: p-norm
    :return: Y
    """
    if pi >= 1.0:
        return float("nan")
    factor = rho / (1.0 - pi)
    gap = (alpha ** (1.0 / p)) - ((1.0 - alpha) ** (1.0 / p))
    return float(factor * (2.0 ** (1.0 / p)) * gap)


def lp_distance_row_pair(
    Q: csr_matrix,
    D: csr_matrix,
    q_idx: int,
    d_idx: int,
    p: int,
) -> float:
    """
    L_p distance between Q[q_idx] and D[d_idx], treating them as sparse vectors.
    Uses sparse subtraction to avoid densifying.
    :param Q: sparse matrix of shape (n_rows, m)
    :param D: sparse matrix of shape (n_rows, m)
    :param q_idx: index of the query row
    :param d_idx: index of the document row
    :param p: p-norm
    :return: L_p distance between Q[q_idx] and D[d_idx]
    """
    diff = Q[q_idx] - D[d_idx]  
    if diff.nnz == 0:
        return 0.0
    return float(np.sum(np.abs(diff.data) ** p) ** (1.0 / p))


def compute_stability_ratio(
    Q: csr_matrix,
    D: csr_matrix,
    sampled_queries: int,
    sampled_docs: int,
    p: int,
    seed: int = 0,
) -> Tuple[float, float]:
    """
    Compute the mean and median stability ratios across the sampled queries. We fix a set of documents 
    so that each query is compared to the same set of documents.

    :param Q: sparse matrix of shape (n_rows, m)
    :param D: sparse matrix of shape (n_rows, m)
    :param sampled_queries: number of queries to sample
    :param sampled_docs: number of documents to sample
    :param p: p-norm
    :param seed: random seed
    :return: mean and median stability ratios

    """
    rng = default_rng(seed)

    n_q = Q.shape[0]
    n_d = D.shape[0]

    if n_q == 0 or n_d == 0:
        return float("nan"), float("nan")

    if sampled_queries is None or sampled_queries <= 0 or sampled_queries >= n_q:
        q_indices = np.arange(n_q)
    else:
        q_indices = rng.choice(n_q, size=sampled_queries, replace=False)

    k = min(sampled_docs, n_d)
    doc_indices = rng.choice(n_d, size=k, replace=False)

    ratios = []

    for qi in q_indices:
        dists = []
        for di in doc_indices:
            dist = lp_distance_row_pair(Q, D, qi, di, p)
            if dist > 0.0:
                dists.append(dist)

        if not dists:
            continue

        dists = np.asarray(dists, dtype=np.float64)
        d_min = dists.min()
        d_max = dists.max()
        ratios.append(d_max / d_min)

    if not ratios:
        return float("nan"), float("nan")

    ratios = np.asarray(ratios, dtype=np.float64)
    return float(ratios.mean()), float(np.median(ratios))


def validate_theorem(
    corpus_path: str,
    query_path: str,
    kappa: int,
    R_values: List[int],
    alpha: float,
    rho_target: float,
    p: int,
    n_pairs_overlap: int,
    pi_frac_of_max: float,
    seed: int,
    csv_out: str,
    dataset_name: str,
    stability_queries: int = 2000,
    stability_sampled_docs: int = 2000,
):
    """
    Empirically validate Theorem 7.4 on sparse embeddings by computing all required constants.

    More concretely, we validate that 
    - Concentration of importance holds for the sparse embeddings with parameters (kappa, alpha, R, rho)
    - Overlap of importance holds with parameters (gamma, pi). Note that we estimate pi from the data by 
      observing the fraction of query-document pairs with positive overlap. 
      This fraction is the maximum possible value pi could take, so we name it pi_max. To control overlap
      we can very this probabilitiy value by setting it to a value in the range (0, pi_max]. We control this
      using the parameter pi_frac_of_max.
    - The Y > X condition holds, where 
        X = (2 - 2*gamma)^(1/p)
        Y = (rho / (1-pi)) * 2^(1/p) * (alpha - (1-alpha)^(1/p))

    NOTE: Since document concentration is only required probabilistically, we set a target probability rho_target
    and search for the smallest R such that Pr_d[C_d(R*kappa) >= alpha] >= rho_target.
 

    :param corpus_path: Path to sparse corpus matrix (NPZ format, CSR matrix)
    :param query_path: Path to sparse query matrix (NPZ format, CSR matrix)
    :param kappa: Head size for queries (number of top dimensions)
    :param R_values: Candidate multipliers for document head size (kappa_d = R * kappa)
    :param alpha: Concentration threshold, typically in (0, 1)
    :param rho_target: Target probability for document concentration condition
    :param p: p-norm for distance computations (typically 1 or 2)
    :param n_pairs_overlap: Number of query-document pairs to sample for overlap statistics
    :param pi_frac_of_max: Fraction of maximum overlap probability to target (in [0, 1])
    :param seed: Random seed for reproducibility
    :param csv_out: Path to output CSV file for results (None to skip CSV output)
    :param dataset_name: Name of dataset for logging and CSV output
    :param stability_queries: Number of queries to sample for stability ratio computation
    :param stability_sampled_docs: Number of documents to sample for stability ratio computation

    :returns: Summary dictionary containing all computed constants:

          - dataset: dataset name
          - kappa: query head size
          - R_star: selected R multiplier
          - kappa_d: document head size (R_star * kappa)
          - alpha: concentration threshold
          - rho: document concentration probability
          - gamma: overlap threshold
          - pi_hat: empirical overlap probability
          - pi_max: maximum observed overlap probability
          - stab_mean: mean stability ratio
          - stab_median: median stability ratio
          - X: theoretical lower bound term
          - Y: theoretical upper bound term
          - Y-X: gap (positive indicates theorem conditions satisfied)
    """

    print(f"Loading corpus from {corpus_path}")
    D = load_npz(corpus_path)
    print(f"Loading queries from {query_path}")
    Q = load_npz(query_path)

    nnz_q = nnz_stats_per_row(Q)
    nnz_d = nnz_stats_per_row(D)
    print(f"Query nnz stats: {nnz_q}")
    print(f"Document nnz stats: {nnz_d}")

    print("\n[Queries] Computing CoI and heads...")
    _, heads_q, _ = compute_heads_and_concentration(Q, kappa, p)

    best_R = None
    best_kappa_d = None
    best_rho = None

    print(
        "\n[Docs] Searching for minimal R such that Pr_d[C_d(R kappa) >= alpha] >= rho_target"
    )
    print(f"  R candidates: {R_values}, rho_target = {rho_target:.3f}")

    for R in R_values:
        kappa_d = R * kappa
        print(f"    R = {R}, kappa_d = {kappa_d}: computing CoI...")
        cd, _, _ = compute_heads_and_concentration(D, kappa_d, p)
        rho = float((cd >= alpha).mean())
        print(f"      Pr_d[C_d(kappa_d) >= alpha] ≈ {rho:.4f}")

        if rho >= rho_target and best_R is None:
            best_R = R
            best_kappa_d = kappa_d
            best_rho = rho
            break

    if best_R is None:
        print(
            "  WARNING: no R in the candidate list reached rho_target; "
            "using the largest R for downstream quantities."
        )
        best_R = R_values[-1]
        best_kappa_d = best_R * kappa
        cd, _, _ = compute_heads_and_concentration(D, best_kappa_d, p)
        best_rho = float((cd >= alpha).mean())

    print(
        f"\n  Selected R* = {best_R}, kappa_d* = {best_kappa_d}, "
        f"Pr_d[C_d(kappa_d*) >= alpha] ≈ {best_rho:.4f}"
    )

    _, heads_d, _ = compute_heads_and_concentration(
        D, best_kappa_d, p
    )

    S_vals = sample_overlap(heads_q, heads_d, n_pairs_overlap, seed=seed + 1)

    # Diagnostics: how many overlaps are > 0, and smallest positive overlap
    positive = S_vals[S_vals > 0.0]
    if positive.size == 0:
        pi_max = 0.0
        gamma_min = 0.0
        gamma = 0.0
        pi_hat = 0.0
        print("  WARNING: no positive overlaps observed; setting gamma=0, pi_hat=0.")
    else:
        pi_max = float(positive.size) / float(S_vals.size)
        gamma_min = float(positive.min())

        print(f"  Fraction of positive overlaps (pi_max) ≈ {pi_max:.4f}")
        print(f"  Smallest positive overlap (gamma_min) ≈ {gamma_min:.6e}")

        pi_frac_eff = max(0.0, min(1.0, float(pi_frac_of_max)))
        # This is the target pi we want to certify. 
        pi_eff = pi_frac_eff * pi_max

        if pi_eff <= 0.0:
            # Degenerate choice: fall back to "any positive overlap"
            gamma = gamma_min
            pi_hat = pi_max
            print(
                "  pi_eff <= 0; using pi_eff = pi_max and gamma = gamma_min "
                "(any positive overlap)."
            )
        else:
            # Quantile level so that Pr[S >= gamma] ≈ pi_eff
            q_level = 1.0 - pi_eff
            gamma = float(np.quantile(S_vals, q_level))
            pi_hat = float((S_vals >= gamma).mean())

        print(f"  Using pi_frac_of_max = {pi_frac_eff:.3f}")
        print(f"  Effective pi_eff target          ≈ {pi_eff:.4f}")
        print(f"  gamma = quantile_{1.0-pi_eff:.3f}(S) ≈ {gamma:.6e}")
        print(f"  Empirical pi_hat = Pr[S >= gamma] ≈ {pi_hat:.4f}")


    X_val = theoretical_X(gamma=gamma, p=p)
    Y_val = theoretical_Y(alpha=alpha, rho=best_rho, pi=pi_hat, p=p)
    gap_val = Y_val - X_val

    print(f"Gap condition Y > X is {gap_val:.6f}")

    stab_mean, stab_median = compute_stability_ratio(
        Q,
        D,
        sampled_queries=stability_queries,
        sampled_docs=stability_sampled_docs,
        p=p,
        seed=seed,
    )
    print(f"Mean stability ratio d_max/d_min is {stab_mean:.4f}")
    print(f"Median stability ratio d_max/d_min is {stab_median:.4f}")

    summary = {
        "dataset": dataset_name,
        "kappa": kappa,
        "R_star": best_R,
        "kappa_d": best_kappa_d,
        "alpha": alpha,
        "rho": best_rho,
        "gamma": gamma,
        "pi_hat": pi_hat,
        "pi_max": pi_max,
        "stab_mean": stab_mean,
        "stab_median": stab_median,
        "X": X_val,
        "Y": Y_val,
        "Y-X": gap_val,
    }

    for k, v in summary.items():
        print(f"  {k:24} = {v}")

    if csv_out is not None:
        csv_dir = os.path.dirname(csv_out)
        if csv_dir and not os.path.exists(csv_dir):
            os.makedirs(csv_dir, exist_ok=True)

        write_header = not os.path.exists(csv_out)
        with open(csv_out, "a") as f:
            if write_header:
                f.write(
                    "dataset,kappa,R_star,kappa_d,alpha,"
                    "rho,gamma,pi_hat,pi_max,"
                    "stab_mean,stab_median,"
                    "X,Y,Y-X\n"
                )
            f.write(
                f"{dataset_name},{kappa},{best_R},{best_kappa_d},{alpha},"
                f"{best_rho:.4f},{gamma:.6e},{pi_hat:.4f},{pi_max:.4f},"
                f"{stab_mean:.4f},{stab_median:.4f},"
                f"{X_val:.6f},{Y_val:.6f},{gap_val:.6f}\n"
            )
        print(f"\nSummary appended to {csv_out}")

    return summary

def main():
    ap = argparse.ArgumentParser(
        description="Empirically validate Theorem 7.4 constants on SPLADE embeddings"
    )
    ap.add_argument("--corpus-npz", required=True, help="Path to corpus_splade.npz")
    ap.add_argument("--queries-npz", required=True, help="Path to queries_splade.npz")
    ap.add_argument(
        "--kappa", type=int, default=18, help="Head size kappa for queries"
    )
    ap.add_argument(
        "--R-values",
        type=str,
        default="1,2,3,4,5,6,7,8",
        help="Comma-separated list of R candidates (R * kappa for docs)",
    )
    ap.add_argument(
        "--alpha", type=float, default=0.85, help="Concentration threshold alpha"
    )
    ap.add_argument(
        "--rho-target",
        type=float,
        default=0.9,
        help="Target pass rate for document concentration",
    )
    ap.add_argument("--p", type=int, default=2, help="l_p norm (1 or 2 typically)")
    ap.add_argument(
        "--pairs",
        type=int,
        default=200_000,
        help="# of (q,d) pairs for overlap sampling",
    )

    # NEW: how aggressively to use pi_max when defining (gamma, pi)
    ap.add_argument(
        "--pi-frac-of-max",
        type=float,
        default=0.95,
        help=(
            "Fraction of pi_max used to define pi_eff. "
            "pi_eff = pi_frac_of_max * pi_max, and "
            "gamma is chosen so that Pr[S >= gamma] ≈ pi_eff."
        ),
    )

    ap.add_argument("--seed", type=int, default=0, help="Random seed")

    ap.add_argument(
        "--dataset-name",
        type=str,
        default="DATASET",
        help="Name of dataset for CSV output",
    )
    ap.add_argument(
        "--csv-out",
        type=str,
        default=None,
        help="Optional path to CSV file. If not provided, defaults to output/{dataset_name}_results.csv",
    )
    ap.add_argument(
        "--stability-queries",
        type=int,
        default=2000,
        help="Max #queries used to estimate stability ratio",
    )
    ap.add_argument(
        "--stability-docs",
        type=int,
        default=2000,
        help="#documents sampled for stability ratio",
    )

    args = ap.parse_args()
    R_values = [int(r) for r in args.R_values.split(",") if r.strip()]

    csv_out = args.csv_out
    if csv_out is None:
        csv_out = f"output/{args.dataset_name}_results.csv"

    validate_theorem(
        corpus_path=args.corpus_npz,
        query_path=args.queries_npz,
        kappa=args.kappa,
        R_values=R_values,
        alpha=args.alpha,
        rho_target=args.rho_target,
        p=args.p,
        n_pairs_overlap=args.pairs,
        pi_frac_of_max=args.pi_frac_of_max,
        seed=args.seed,
        csv_out=csv_out,
        dataset_name=args.dataset_name,
        stability_queries=args.stability_queries,
        stability_docs=args.stability_docs,
    )


if __name__ == "__main__":
    main()
