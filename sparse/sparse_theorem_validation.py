"""
Empirical validation of the two-regime sparse stability theorem.

For a random query-document pair (q, d) of nonnegative unit l_p vectors, let T_q be
the top-kappa coordinates of q and T_d the top-(R kappa) coordinates of d, and let

    S  = sum_{i in T_q ∩ T_d} min{q_i^p, d_i^p},        K = |T_q ∩ T_d|,
    A  = {S >= gamma},                                   (overlapping heads)
    B  = {K = 0} ∩ {C_d(R kappa) >= alpha} ∩ {C_q(kappa) >= alpha}.   (separated heads)

On A, delta <= X = (2 - 2 gamma)^{1/p}; on B, delta >= L = 2^{1/p}(alpha^{1/p} - (1-alpha)^{1/p}).
If Pr(A) >= pi, Pr(B) >= beta and L > X, then

    RelVar(delta) >= pi * beta * (L - X)^2 / 2^{2/p}.

NOTE: the theorem states query concentration for *every* query. The proof only uses
it on B, so we fold {C_q >= alpha} into B; the bound is then valid even when a few
queries fail concentration, and we report the query pass rate separately.

We also evaluate the low-cross-mass variant of B (remark in the proof note), where
exact worst-case cross mass 1 - alpha is replaced by a measured bound eta:

    B_eta = B ∩ {sum_{T_q} d_i^p <= eta} ∩ {sum_{T_d} q_i^p <= eta},
    L_eta = 2^{1/p}(alpha^{1/p} - eta^{1/p}).

Pairs are sampled independently from the query and document marginals, which is the
sampling law used by the paper's definition of stability.
"""

import argparse
import itertools
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from scipy.sparse import csr_matrix, load_npz

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from plotting import save_results  # noqa: E402

DATASETS = ["hotpotqa", "trec-covid", "natural-questions", "msmarco-v2.1", "nfcorpus"]
RESULTS_DIR = ROOT / "results" / "two_regime"


def normalize_rows(X: csr_matrix, p: int) -> csr_matrix:
    """Drop empty rows and scale every row to unit l_p norm."""
    X = X.tocsr().astype(np.float32)
    X.data = np.abs(X.data)
    X = X[np.diff(X.indptr) > 0]
    norms = np.asarray(X.power(p).sum(axis=1)).ravel() ** (1.0 / p)
    X = X.multiply(1.0 / norms[:, None]).tocsr()
    return X


def to_dense_gpu(X: csr_matrix, device: torch.device) -> torch.Tensor:
    return torch.sparse_csr_tensor(
        torch.from_numpy(X.indptr.astype(np.int64)),
        torch.from_numpy(X.indices.astype(np.int64)),
        torch.from_numpy(X.data),
        size=X.shape,
    ).to(device).to_dense()


def head_mask(mass: torch.Tensor, k: int):
    """Top-k head of every row as a boolean mask (zero-mass coordinates are never in a head)."""
    k = min(k, mass.shape[1])
    vals, idx = mass.topk(k, dim=1)
    mask = torch.zeros_like(mass, dtype=torch.bool)
    mask.scatter_(1, idx, vals > 0)
    return mask, vals.sum(dim=1)


@torch.no_grad()
def concentration_all_rows(X: csr_matrix, ks: List[int], p: int, device, batch: int = 8192) -> Dict[int, np.ndarray]:
    """C_x(k) for every row of X and every k in ks."""
    out = {k: [] for k in ks}
    kmax = min(max(ks), X.shape[1])
    for s in range(0, X.shape[0], batch):
        mass = to_dense_gpu(X[s : s + batch], device) ** p
        csum = mass.topk(kmax, dim=1).values.cumsum(dim=1)
        for k in ks:
            out[k].append(csum[:, min(k, kmax) - 1].cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


@torch.no_grad()
def sample_pair_statistics(
    Q: csr_matrix,
    D: csr_matrix,
    kappas: List[int],
    Rs: List[int],
    p: int,
    n_pairs: int,
    device,
    seed: int,
    batch: int = 4096,
) -> Dict:
    """Per-pair statistics for independently sampled (q, d) pairs, for every (kappa, R)."""
    rng = np.random.default_rng(seed)
    iq = rng.integers(0, Q.shape[0], size=n_pairs)
    idd = rng.integers(0, D.shape[0], size=n_pairs)

    configs = list(itertools.product(kappas, Rs))
    fields = ["K", "S", "Cq", "Cd", "eta_q", "eta_d"]
    stats = {cfg: {f: np.empty(n_pairs, dtype=np.float32) for f in fields} for cfg in configs}
    delta = np.empty(n_pairs, dtype=np.float32)
    cosine = np.empty(n_pairs, dtype=np.float32)

    for s in range(0, n_pairs, batch):
        e = min(s + batch, n_pairs)
        Xq = to_dense_gpu(Q[iq[s:e]], device)
        Xd = to_dense_gpu(D[idd[s:e]], device)
        Mq, Md = Xq**p, Xd**p
        delta[s:e] = ((Xq - Xd).abs() ** p).sum(dim=1).pow(1.0 / p).cpu().numpy()
        cosine[s:e] = (
            (Xq * Xd).sum(dim=1) / (Xq.norm(dim=1) * Xd.norm(dim=1))
        ).cpu().numpy()
        Mmin = torch.minimum(Mq, Md)

        q_heads = {k: head_mask(Mq, k) for k in kappas}
        d_heads = {kd: head_mask(Md, kd) for kd in sorted({k * R for k, R in configs})}
        for kappa, R in configs:
            mq, Cq = q_heads[kappa]
            md, Cd = d_heads[kappa * R]
            inter = mq & md
            st = stats[(kappa, R)]
            st["K"][s:e] = inter.sum(dim=1).cpu().numpy()
            st["S"][s:e] = (Mmin * inter).sum(dim=1).cpu().numpy()
            st["Cq"][s:e] = Cq.cpu().numpy()
            st["Cd"][s:e] = Cd.cpu().numpy()
            st["eta_q"][s:e] = (Md * mq).sum(dim=1).cpu().numpy()
            st["eta_d"][s:e] = (Mq * md).sum(dim=1).cpu().numpy()
        if (s // batch) % 50 == 0:
            print(f"    pairs {e}/{n_pairs}", flush=True)

    return {"stats": stats, "delta": delta, "cosine": cosine}


def theory_L(alpha: float, p: int, eta: float = None) -> float:
    eta = (1.0 - alpha) if eta is None else eta
    return float(2 ** (1.0 / p) * (alpha ** (1.0 / p) - eta ** (1.0 / p)))


def theory_X(gamma, p: int):
    return (2.0 - 2.0 * np.asarray(gamma)) ** (1.0 / p)


def gamma_threshold(alpha: float, p: int, eta: float = None) -> float:
    """Smallest gamma for which L > X, i.e. 2 - 2 gamma < L^p."""
    return 1.0 - theory_L(alpha, p, eta) ** p / 2.0


def best_bound(S: np.ndarray, in_B: np.ndarray, delta: np.ndarray, L: float, gamma_min: float, p: int, rng, n_boot: int):
    """
    Maximise pi(gamma) * beta * (L - X(gamma))^2 / 2^{2/p} over gamma > gamma_min, and
    attach a multinomial-bootstrap CI (A and B are disjoint, so counts are multinomial).
    """
    n = S.size
    beta = float(in_B.mean())
    out = {
        "L": L, "gamma_min": gamma_min, "beta": beta, "n_B": int(in_B.sum()),
        "gamma": None, "pi": 0.0, "n_A": 0, "X": None, "gap": None, "bound": 0.0,
        "pi_ci": [0.0, 3.0 / n], "beta_ci": None, "bound_ci": [0.0, 0.0],
        "max_delta_on_A": None, "min_delta_on_B": float(delta[in_B].min()) if in_B.any() else None,
    }
    lo, hi = np.quantile(rng.binomial(n, beta, size=n_boot) / n, [0.025, 0.975]) if beta > 0 else (0.0, 3.0 / n)
    out["beta_ci"] = [float(lo), float(hi)]

    cand = np.unique(S[S > gamma_min])
    if cand.size == 0 or beta == 0.0:
        return out
    if cand.size > 400:
        cand = np.quantile(cand, np.linspace(0, 1, 400))
    S_sorted = np.sort(S)
    pis = 1.0 - np.searchsorted(S_sorted, cand, side="left") / n
    bounds = pis * beta * (L - theory_X(cand, p)) ** 2 / 2 ** (2.0 / p)
    j = int(np.argmax(bounds))
    gamma, pi = float(cand[j]), float(pis[j])
    in_A = S >= gamma

    counts = rng.multinomial(n, [pi, beta, max(0.0, 1.0 - pi - beta)], size=n_boot) / n
    gap = L - float(theory_X(gamma, p))
    boot = counts[:, 0] * counts[:, 1] * gap**2 / 2 ** (2.0 / p)
    out.update({
        "gamma": gamma, "pi": pi, "n_A": int(in_A.sum()), "X": float(theory_X(gamma, p)),
        "gap": gap, "bound": float(bounds[j]),
        "pi_ci": [float(x) for x in np.quantile(counts[:, 0], [0.025, 0.975])],
        "bound_ci": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
        "max_delta_on_A": float(delta[in_A].max()),
    })
    return out


def analyse(sample: Dict, conc_q: Dict, conc_d: Dict, alphas: List[float], etas: List[float], p: int, seed: int, n_boot: int) -> List[Dict]:
    rng = np.random.default_rng(seed)
    delta = sample["delta"]
    rows = []
    for (kappa, R), st in sample["stats"].items():
        for alpha in alphas:
            conc = (st["Cq"] >= alpha) & (st["Cd"] >= alpha)
            in_B = (st["K"] == 0) & conc
            row = {
                "kappa": kappa, "R": R, "alpha": alpha,
                "query_coi_pass": float((conc_q[kappa] >= alpha).mean()),
                "query_coi_min": float(conc_q[kappa].min()),
                "rho": float((conc_d[kappa * R] >= alpha).mean()),
                "pi_max": float((st["S"] > 0).mean()),
                "pr_disjoint": float((st["K"] == 0).mean()),
                "S_quantiles": {q: float(np.quantile(st["S"], q)) for q in (0.5, 0.9, 0.99, 0.999, 0.9999)},
                "S_max": float(st["S"].max()),
                "cross_mass_on_B_q99": float(np.quantile(np.maximum(st["eta_q"], st["eta_d"])[in_B], 0.99)) if in_B.any() else None,
                "exact": best_bound(st["S"], in_B, delta, theory_L(alpha, p), gamma_threshold(alpha, p), p, rng, n_boot),
                "low_cross_mass": {},
            }
            for eta in etas:
                if eta >= 1.0 - alpha:
                    continue
                in_B_eta = in_B & (st["eta_q"] <= eta) & (st["eta_d"] <= eta)
                row["low_cross_mass"][eta] = best_bound(
                    st["S"], in_B_eta, delta, theory_L(alpha, p, eta), gamma_threshold(alpha, p, eta), p, rng, n_boot
                )
            rows.append(row)
    return rows


def validate_dataset(name: str, emb_dir: Path, args, device) -> Dict:
    print(f"\n=== {name} ===", flush=True)
    Q = normalize_rows(load_npz(emb_dir / name / "queries_splade.npz"), args.p)
    D = normalize_rows(load_npz(emb_dir / name / "corpus_splade.npz"), args.p)
    print(f"  queries {Q.shape}, docs {D.shape}", flush=True)

    conc_q = concentration_all_rows(Q, args.kappas, args.p, device)
    conc_d = concentration_all_rows(D, sorted({k * R for k in args.kappas for R in args.Rs}), args.p, device)
    sample = sample_pair_statistics(Q, D, args.kappas, args.Rs, args.p, args.pairs, device, args.seed)

    delta, cosine = sample["delta"], sample["cosine"]
    result = {
        "dataset": name, "p": args.p, "n_pairs": args.pairs, "m": int(Q.shape[1]),
        "n_queries": int(Q.shape[0]), "n_docs": int(D.shape[0]),
        "relvar": float(delta.var() / delta.mean() ** 2),
        "delta_mean": float(delta.mean()), "delta_min": float(delta.min()), "delta_max": float(delta.max()),
        "cosine_mean": float(cosine.mean()), "cosine_var": float(cosine.var()),
        "cosine_quantiles": {q: float(np.quantile(cosine, q)) for q in (0.5, 0.9, 0.99, 0.999, 0.9999)},
        "configs": analyse(sample, conc_q, conc_d, args.alphas, args.etas, args.p, args.seed, args.bootstrap),
    }

    out_dir = Path(args.results_dir)
    save_results(result, out_dir / f"{name}.json")
    if args.save_pairs:
        arrays = {"delta": delta, "cosine": cosine}
        for (kappa, R), st in sample["stats"].items():
            arrays.update({f"k{kappa}_R{R}_{f}": v for f, v in st.items()})
        np.savez_compressed(out_dir / f"{name}_pairs.npz", **arrays)
    print_summary(result)
    return result


def print_summary(result: Dict):
    print(f"\n  [{result['dataset']}] measured RelVar = {result['relvar']:.5f}, "
          f"mean cos = {result['cosine_mean']:.4f}, Var(cos) = {result['cosine_var']:.5f}")
    hdr = f"  {'kappa':>5} {'R':>2} {'alpha':>5} {'qCoI':>6} {'rho':>6} {'beta':>7} {'g_min':>6} {'gamma':>6} {'pi':>9} {'L-X':>7} {'bound':>10}"
    print(hdr)
    for row in result["configs"]:
        ex = row["exact"]
        fmt = lambda v, s: format(v, s) if v is not None else "-"
        print(f"  {row['kappa']:>5} {row['R']:>2} {row['alpha']:>5} {row['query_coi_pass']:>6.3f} {row['rho']:>6.3f} "
              f"{ex['beta']:>7.4f} {ex['gamma_min']:>6.3f} {fmt(ex['gamma'], '6.3f'):>6} {ex['pi']:>9.2e} "
              f"{fmt(ex['gap'], '7.4f'):>7} {ex['bound']:>10.3e}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Validate the two-regime sparse stability theorem on SPLADE embeddings.")
    ap.add_argument("--embeddings-dir", default=str(ROOT / "splade_embeddings"))
    ap.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--kappas", type=int, nargs="+", default=[8, 18, 24, 48])
    ap.add_argument("--Rs", type=int, nargs="+", default=[1, 3, 6, 8])
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.75, 0.85, 0.9, 0.95])
    ap.add_argument("--etas", type=float, nargs="+", default=[0.01, 0.02, 0.05],
                    help="Cross-head mass bounds for the low-cross-mass variant of B")
    ap.add_argument("--pairs", type=int, default=2_000_000, help="# of independently sampled (q, d) pairs")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save-pairs", action="store_true", help="Also save the per-pair statistics (NPZ)")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device(args.device)
    for name in args.datasets:
        validate_dataset(name, Path(args.embeddings_dir), args, device)
