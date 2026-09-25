"""CPU-only parameter sweep for the two-regime sparse stability theorem.

Reads the per-pair statistics saved by ``sparse_theorem_validation.py --save-pairs``
and searches (kappa, R, alpha, gamma) for the largest certified lower bound

    RelVar >= pi * beta * (Y - X)^2 / 2^{2/p},   Y > X,

with a much finer alpha grid than the original run. Also reports, for a few
target values of pi, the gamma the data can support and the alpha that would
be needed to satisfy the gap condition at that gamma ("what would it take").
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def theory_Y(alpha, p, eta=None):
    eta = (1.0 - alpha) if eta is None else eta
    return 2 ** (1.0 / p) * (alpha ** (1.0 / p) - eta ** (1.0 / p))


def theory_X(gamma, p):
    return (2.0 - 2.0 * np.asarray(gamma, dtype=float)) ** (1.0 / p)


def gamma_min(alpha, p, eta=None):
    return 1.0 - theory_Y(alpha, p, eta) ** p / 2.0


def alpha_needed(gamma, p):
    """Smallest alpha with Y(alpha) > X(gamma) (exact worst-case cross mass), by bisection."""
    lo, hi = 0.5, 1.0
    if theory_Y(hi, p) <= theory_X(gamma, p):
        return None
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if theory_Y(mid, p) > theory_X(gamma, p):
            hi = mid
        else:
            lo = mid
    return hi


def best_bound(S, in_B, p, Y, g_min):
    beta = float(in_B.mean())
    S_sorted = np.sort(S)
    cand = np.unique(S_sorted[S_sorted > g_min])
    if beta == 0.0 or cand.size == 0:
        return {"beta": beta, "gamma": None, "pi": 0.0, "n_A": 0, "bound": 0.0, "gap": None}
    # pi(gamma) = fraction of S >= gamma
    n = S.size
    pis = 1.0 - np.searchsorted(S_sorted, cand, side="left") / n
    gaps = Y - theory_X(cand, p)
    bounds = pis * beta * gaps**2 / 2 ** (2.0 / p)
    j = int(np.argmax(bounds))
    return {
        "beta": beta, "gamma": float(cand[j]), "pi": float(pis[j]),
        "n_A": int(round(pis[j] * n)), "bound": float(bounds[j]), "gap": float(gaps[j]),
    }


def sweep_dataset(path: Path, p: int, alphas, etas, target_pis):
    z = np.load(path)
    delta = z["delta"]
    relvar = float(delta.var() / delta.mean() ** 2)
    cfgs = sorted({tuple(map(int, m.groups())) for k in z.files for m in [re.match(r"k(\d+)_R(\d+)_S$", k)] if m})

    rows = []
    for kappa, R in cfgs:
        pre = f"k{kappa}_R{R}_"
        S, K, Cq, Cd = z[pre + "S"], z[pre + "K"], z[pre + "Cq"], z[pre + "Cd"]
        eta_max = np.maximum(z[pre + "eta_q"], z[pre + "eta_d"])
        disjoint = K == 0
        for alpha in alphas:
            in_B = disjoint & (Cd >= alpha) & (Cq >= alpha)
            row = {"kappa": kappa, "R": R, "alpha": alpha, "rho": float((Cd >= alpha).mean()),
                   "qcoi": float((Cq >= alpha).mean()), "disjoint": float(disjoint.mean()),
                   "gamma_min": gamma_min(alpha, p)}
            row["exact"] = best_bound(S, in_B, p, theory_Y(alpha, p), gamma_min(alpha, p))
            row["eta"] = {}
            for eta in etas:
                if eta >= 1 - alpha:  # no tighter than the exact worst case
                    continue
                row["eta"][eta] = best_bound(S, in_B & (eta_max <= eta), p, theory_Y(alpha, p, eta), gamma_min(alpha, p, eta))
            rows.append(row)

    # "What would it take": per (kappa, R), for each target pi, the gamma the data
    # supports and the alpha needed for the gap condition at that gamma.
    wwit = []
    for kappa, R in cfgs:
        pre = f"k{kappa}_R{R}_"
        S, K, Cq, Cd = z[pre + "S"], z[pre + "K"], z[pre + "Cq"], z[pre + "Cd"]
        disjoint = K == 0
        for tp in target_pis:
            gamma = float(np.quantile(S, 1.0 - tp))
            a_need = alpha_needed(gamma, p)
            if a_need is None:
                wwit.append({"kappa": kappa, "R": R, "target_pi": tp, "gamma": gamma, "alpha_needed": None})
                continue
            in_B = disjoint & (Cd >= a_need) & (Cq >= a_need)
            beta = float(in_B.mean())
            wwit.append({"kappa": kappa, "R": R, "target_pi": tp, "gamma": gamma, "alpha_needed": a_need,
                         "rho_at_alpha": float((Cd >= a_need).mean()), "qcoi_at_alpha": float((Cq >= a_need).mean()),
                         "beta_at_alpha": beta})
    return {"dataset": path.stem.replace("_pairs", ""), "p": p, "n_pairs": int(delta.size), "relvar": relvar,
            "rows": rows, "what_would_it_take": wwit}


def fmt(x, f):
    return "   -" if x is None else format(x, f)


def print_report(res, top=8):
    print(f"\n=== {res['dataset']}  (p={res['p']}, measured RelVar = {res['relvar']:.2e}, {res['n_pairs']} pairs) ===")
    rows = sorted(res["rows"], key=lambda r: -r["exact"]["bound"])
    print(f"  Best exact configs (worst-case cross mass 1-alpha):")
    print(f"  {'kappa':>5} {'R':>2} {'alpha':>6} {'rho':>6} {'qCoI':>6} {'beta':>7} {'g_min':>6} {'gamma':>6} {'pi':>9} {'n_A':>6} {'Y-X':>6} {'bound':>9} {'bnd/RV':>8}")
    for r in rows[:top]:
        e = r["exact"]
        print(f"  {r['kappa']:>5} {r['R']:>2} {r['alpha']:>6.3f} {r['rho']:>6.3f} {r['qcoi']:>6.3f} {e['beta']:>7.4f} "
              f"{r['gamma_min']:>6.3f} {fmt(e['gamma'], '6.3f'):>6} {e['pi']:>9.2e} {e['n_A']:>6} {fmt(e['gap'], '6.3f'):>6} "
              f"{e['bound']:>9.2e} {e['bound']/res['relvar']:>8.1e}")
    # best per alpha
    print(f"  Best exact bound per alpha (any kappa, R):")
    for alpha in sorted({r["alpha"] for r in res["rows"]}):
        r = max((r for r in res["rows"] if r["alpha"] == alpha), key=lambda r: r["exact"]["bound"])
        e = r["exact"]
        print(f"    alpha={alpha:.3f}: kappa={r['kappa']:>2} R={r['R']} beta={e['beta']:.3f} g_min={r['gamma_min']:.3f} "
              f"gamma={fmt(e['gamma'], '.3f')} pi={e['pi']:.2e} (n_A={e['n_A']}) bound={e['bound']:.2e}")
    # best eta variant
    best_eta = max(((r, eta, b) for r in res["rows"] for eta, b in r["eta"].items()), key=lambda t: t[2]["bound"], default=None)
    if best_eta:
        r, eta, b = best_eta
        print(f"  Best low-cross-mass bound: eta={eta} kappa={r['kappa']} R={r['R']} alpha={r['alpha']:.3f} beta={b['beta']:.3f} "
              f"gamma={fmt(b['gamma'], '.3f')} pi={b['pi']:.2e} (n_A={b['n_A']}) bound={b['bound']:.2e} "
              f"({b['bound']/res['relvar']:.1e} of RelVar)")
    print(f"  What would it take (gamma = (1-pi) quantile of S; alpha needed for Y > X at that gamma):")
    print(f"  {'kappa':>5} {'R':>2} {'target_pi':>9} {'gamma':>7} {'alpha_need':>10} {'rho':>6} {'qCoI':>6} {'beta':>7}")
    for w in res["what_would_it_take"]:
        if w["alpha_needed"] is None:
            print(f"  {w['kappa']:>5} {w['R']:>2} {w['target_pi']:>9.0e} {w['gamma']:>7.4f} {'impossible':>10}")
        else:
            print(f"  {w['kappa']:>5} {w['R']:>2} {w['target_pi']:>9.0e} {w['gamma']:>7.4f} {w['alpha_needed']:>10.4f} "
                  f"{w['rho_at_alpha']:>6.3f} {w['qcoi_at_alpha']:>6.3f} {w['beta_at_alpha']:>7.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(ROOT / "results" / "two_regime"))
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.75, 0.8, 0.85, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999])
    ap.add_argument("--etas", type=float, nargs="+", default=[0.001, 0.005, 0.01, 0.02, 0.05])
    ap.add_argument("--target-pis", type=float, nargs="+", default=[1e-1, 1e-2, 1e-3, 1e-4])
    ap.add_argument("--out", default=None, help="JSON output (default: <results-dir>/sweep_p<p>.json)")
    args = ap.parse_args()

    rdir = Path(args.results_dir)
    files = sorted(rdir.glob("*_pairs.npz"))
    if args.datasets:
        files = [f for f in files if f.stem.replace("_pairs", "") in args.datasets]
    all_res = []
    for f in files:
        res = sweep_dataset(f, args.p, args.alphas, args.etas, args.target_pis)
        print_report(res)
        all_res.append(res)
    out = Path(args.out) if args.out else rdir / f"sweep_p{args.p}.json"
    out.write_text(json.dumps(all_res, indent=1, default=float))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
