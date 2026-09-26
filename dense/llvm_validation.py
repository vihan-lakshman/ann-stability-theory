"""Appendix C: does the Durrant & Kaban linear latent variable model (Theorem C.1) fit real embeddings?

For each corpus we embed a fixed random subset of documents with three dense encoders and SPLADE,
fit a linear latent variable model x = A y + delta, delta ~ N(0, s2 I) (PPCA, L chosen by held-out
log-likelihood) and measure finite-m proxies for the theorem's condition that some latent factor's
cumulative contribution sum_i a_il^2 grows linearly with the dimension m:

  relevant    fraction of coordinates whose systematic variance sum_l a_il^2 exceeds the noise s2
  PR/m        median over factors of the loading participation ratio (sum_i a_il^2)^2 / (m sum_i a_il^4)
              (~1/3 for dense Gaussian loadings, 1/m for a factor on a single coordinate)
  top-1%      median share of a factor's loading mass on its top 1% of coordinates
  RV_inf      2 sum_{i!=j} C_ij^2 / tr(C)^2, the m -> infinity limit of the relative variance of the
              squared distance implied by the covariance C, assuming new coordinates resemble the old ones

Everything goes through the n x n Gram matrix, so SPLADE's ~30k-wide vocabulary never needs an m x m
covariance. SPLADE coordinates are restricted to the active vocabulary (non-zero in some document).

Outputs: results/llvm_validation.json, tables/llvm_dense.tex, tables/llvm_splade.tex
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from plotting import load_results, save_results  # noqa: E402

RESULTS_PATH = ROOT / "results" / "llvm_validation.json"
EMB_DIR = ROOT / "results" / "llvm_embeddings"
TABLE_DIR = ROOT / "tables"

CORPORA = [("scifact", "SciFact"), ("nfcorpus", "NFCorpus"), ("arguana", "ArguAna")]
DENSE_MODELS = [
    # key, HF model, label, document prefix
    ("bge-small", "BAAI/bge-small-en-v1.5", "bge-small-en-v1.5", ""),
    ("minilm", "sentence-transformers/all-MiniLM-L6-v2", "all-MiniLM-L6-v2", ""),
    ("e5-base", "intfloat/e5-base-v2", "e5-base-v2", "passage: "),
]
SPLADE_MODEL = "naver/splade-cocondenser-ensembledistil"
L_GRID = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]


# --------------------------------------------------------------------------- #
# Embedding
# --------------------------------------------------------------------------- #


def load_docs(corpus: str) -> list[str]:
    from datasets import load_dataset

    c = load_dataset(f"mteb/{corpus}", "corpus")["corpus"]
    return [(t + " " + x).strip() if t else x for t, x in zip(c["title"], c["text"])]


def embed_all(n_docs: int, seed: int, max_seq_length: int):
    """Encode each corpus subset once; later runs reuse results/llvm_embeddings/."""
    from sentence_transformers import SentenceTransformer, SparseEncoder

    EMB_DIR.mkdir(parents=True, exist_ok=True)
    encoders = {}
    for corpus, _ in CORPORA:
        docs = load_docs(corpus)
        idx = np.sort(np.random.default_rng(seed).permutation(len(docs))[:n_docs])
        sub = [docs[i] for i in idx]
        for key, name, _, prefix in DENSE_MODELS:
            path = EMB_DIR / f"{corpus}__{key}.npy"
            if path.exists():
                continue
            if key not in encoders:
                encoders[key] = SentenceTransformer(name, device="cpu")
                encoders[key].max_seq_length = max_seq_length
            X = encoders[key].encode([prefix + d for d in sub], batch_size=64, normalize_embeddings=True,
                                     convert_to_numpy=True, show_progress_bar=True)
            np.save(path, X.astype(np.float32))
            print(f"{corpus}/{key}: {X.shape}", flush=True)
        path = EMB_DIR / f"{corpus}__splade.npz"
        if not path.exists():
            if "splade" not in encoders:
                encoders["splade"] = SparseEncoder(SPLADE_MODEL, device="cpu")
                encoders["splade"].max_seq_length = max_seq_length
            T = encoders["splade"].encode_document(sub, batch_size=32, convert_to_sparse_tensor=True,
                                                   show_progress_bar=True).coalesce().cpu()
            i, v = T.indices().numpy(), T.values().numpy()
            X = sp.csr_matrix((v, (i[0], i[1])), shape=tuple(T.shape), dtype=np.float32)
            sp.save_npz(path, X)
            print(f"{corpus}/splade: {X.shape}, median nnz/doc {np.median(np.diff(X.indptr)):.0f}", flush=True)


def load_embeddings(corpus: str, key: str):
    """Dense float64 matrix (active columns only for SPLADE), full width, and nnz per row."""
    if key == "splade":
        X = sp.load_npz(EMB_DIR / f"{corpus}__splade.npz").tocsr()
        nnz = np.diff(X.indptr)
        active = np.flatnonzero(X.getnnz(0) > 0)
        return X.tocsc()[:, active].toarray().astype(np.float64), X.shape[1], nnz
    X = np.load(EMB_DIR / f"{corpus}__{key}.npy").astype(np.float64)
    return X, X.shape[1], np.full(len(X), X.shape[1])


# --------------------------------------------------------------------------- #
# Linear latent variable model (PPCA) via the Gram matrix
# --------------------------------------------------------------------------- #


def spectrum(Xc: np.ndarray):
    """Non-zero covariance eigenvalues and principal directions of centred X (n << m friendly)."""
    n = Xc.shape[0]
    G = Xc @ Xc.T / (n - 1)
    w, U = np.linalg.eigh(G)
    w, U = w[::-1], U[:, ::-1]
    keep = w > 1e-12 * w[0]
    w, U = w[keep], U[:, keep]
    return w, Xc.T @ U / np.sqrt(w * (n - 1)), G


def ppca(X: np.ndarray, L: int):
    mu = X.mean(0)
    Xc = X - mu
    m = Xc.shape[1]
    w, V, _ = spectrum(Xc)
    total = (Xc ** 2).sum() / (len(Xc) - 1)
    s2 = max((total - w[:L].sum()) / (m - L), 1e-12)
    return mu, V[:, :L], np.maximum(w[:L] - s2, 0), s2


def ppca_loglik(X, mu, V, lam, s2) -> float:
    """Mean log N(x; mu, V diag(lam) V^T + s2 I) for orthonormal V, via Woodbury."""
    m = X.shape[1]
    Xc = X - mu
    P = Xc @ V
    quad = ((Xc ** 2).sum(1) - (P ** 2 * (lam / (lam + s2))).sum(1)) / s2
    logdet = np.log(lam + s2).sum() + (m - len(lam)) * np.log(s2)
    return float(np.mean(-0.5 * (quad + logdet + m * np.log(2 * np.pi))))


def fit_llvm(X: np.ndarray, rng: np.random.Generator):
    perm = rng.permutation(len(X))
    tr, te = X[perm[: len(X) * 3 // 4]], X[perm[len(X) * 3 // 4:]]
    grid = [L for L in L_GRID if L < min(X.shape[1], len(tr))]
    scores = {L: ppca_loglik(te, *ppca(tr, L)) for L in grid}
    L = max(scores, key=scores.get)
    _, V, lam, s2 = ppca(X, L)
    return L, V * np.sqrt(lam), lam, s2, scores


def analyse(X: np.ndarray, m_full: int, nnz: np.ndarray, rng: np.random.Generator) -> dict:
    n, m = X.shape
    Xc = X - X.mean(0)
    _, _, G = spectrum(Xc)
    v = Xc.var(0, ddof=1)
    L, A, lam, s2, scores = fit_llvm(X, rng)
    h = (A ** 2).sum(1)                                   # per-coordinate systematic variance
    col = A ** 2
    pr = col.sum(0) ** 2 / np.maximum((col ** 2).sum(0), 1e-300) / m
    top = max(1, m // 100)
    # measured relative variance of squared distances over (a sample of) document pairs
    sq = (X * X).sum(1)
    D = np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0)[np.triu_indices(n, 1)]
    return dict(
        n=n, m=m, m_full=m_full, nnz_median=float(np.median(nnz)),
        L=L, L_at_grid_top=L == max(scores), s2=float(s2),
        signal_share=float(lam.sum() / (lam.sum() + m * s2)),
        frac_relevant=float((h > s2).mean()),
        frac_relevant_full=float((h > s2).sum() / m_full),
        loading_pr_median=float(np.median(pr)),
        top1pct_mass_median=float(np.median(np.sort(col, 0)[::-1][:top].sum(0) / col.sum(0))),
        rv_inf=float(2 * ((G ** 2).sum() - (v ** 2).sum()) / v.sum() ** 2),
        rv_measured=float(D.var() / D.mean() ** 2),
    )


# --------------------------------------------------------------------------- #
# LaTeX
# --------------------------------------------------------------------------- #

HEADER = r"""\begin{tabular}{llrrrrrr}
\toprule
Corpus & Model & $m$ & $L$ & Relevant coords (\%) & Loading PR$/m$ & Top-1\% mass (\%) & $\mathrm{RV}_\infty$ \\
\midrule"""


def row(r: dict, corpus_label: str, model_label: str) -> str:
    return (f"{corpus_label} & {model_label} & {r['m']:,} & {r['L']} & {100 * r['frac_relevant']:.1f} & "
            f"{r['loading_pr_median']:.3f} & {100 * r['top1pct_mass_median']:.1f} & {r['rv_inf']:.4f} \\\\")


def table(rows: list[str], caption: str, label: str) -> str:
    return "\n".join(["% requires \\usepackage{booktabs}", r"\begin{table}[t]", r"\centering", r"\small",
                      r"\resizebox{\linewidth}{!}{%", HEADER, *rows, r"\bottomrule", r"\end{tabular}}",
                      rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}", ""])


def write_tables(res: dict, n_docs: int):
    TABLE_DIR.mkdir(exist_ok=True)
    dense_rows = []
    for i, (c, cl) in enumerate(CORPORA):
        if i:
            dense_rows.append(r"\midrule")
        for j, (k, _, ml, _) in enumerate(DENSE_MODELS):
            dense_rows.append(row(res[f"{c}/{k}"], cl if j == 0 else "", ml))
    ns = sorted({res[f"{c}/splade"]["n"] for c, _ in CORPORA})
    n_note = f"$n = {n_docs:,}$ documents per corpus".replace(",", "{,}")
    short = [f"{res[f'{c}/splade']['n']:,}".replace(",", "{,}") + f" for {cl}"
             for c, cl in CORPORA if res[f"{c}/splade"]["n"] < n_docs]
    if short:
        n_note += ", " + ", ".join(short)
    common = (
        r"Finite-dimensional proxies for the condition of Theorem~C.1 that some latent factor's cumulative "
        r"contribution $\sum_i a_{il}^2$ grows linearly in $m$, from a PPCA fit $x = Ay + \delta$, "
        r"$\delta \sim \mathcal{N}(0, \sigma^2 I)$, with $L$ chosen by held-out log-likelihood (" + n_note + "). "
        r"\emph{Relevant coords}: fraction of coordinates whose systematic variance $\sum_l a_{il}^2$ exceeds the "
        r"noise variance $\sigma^2$. \emph{Loading PR$/m$}: median over factors of the participation ratio "
        r"$(\sum_i a_{il}^2)^2 / (m \sum_i a_{il}^4)$ (dense Gaussian loadings give $\approx 1/3$; a factor on a "
        r"single coordinate gives $1/m$). \emph{Top-1\% mass}: median share of a factor's loading mass on its top "
        r"1\% of coordinates. $\mathrm{RV}_\infty = 2\sum_{i \neq j} \Sigma_{ij}^2 / (\operatorname{tr}\Sigma)^2$: "
        r"the $m \to \infty$ limit of the relative variance of $\|q - d\|^2$ implied by the covariance $\Sigma$, "
        r"assuming added coordinates resemble the observed ones."
    )
    dense = table(dense_rows, r"Dense embeddings satisfy the linear latent variable condition of Theorem~C.1. "
                  + common, "tab:llvm-dense")

    sp_rows = [row(res[f"{c}/splade"], cl, "SPLADE") for c, cl in CORPORA]
    full = [100 * res[f"{c}/splade"]["frac_relevant_full"] for c, _ in CORPORA]
    nnz = [res[f"{c}/splade"]["nnz_median"] for c, _ in CORPORA]
    m_full = res[f"{CORPORA[0][0]}/splade"]["m_full"]
    splade = table(
        sp_rows,
        r"SPLADE (\texttt{" + SPLADE_MODEL + r"}) embeddings violate the linear latent variable condition of "
        rf"Theorem~C.1. $m$ is the active vocabulary (coordinates non-zero in at least one document) out of "
        f"${m_full:,}$".replace(",", "{,}") + r"; measured over the full vocabulary, the relevant fraction is "
        rf"{min(full):.1f}--{max(full):.1f}\%. Documents have a median of {min(nnz):.0f}--{max(nnz):.0f} non-zero "
        r"coordinates. Other quantities and the setup are as in Table~\ref{tab:llvm-dense}.",
        "tab:llvm-splade",
    )
    for name, text in [("llvm_dense.tex", dense), ("llvm_splade.tex", splade)]:
        (TABLE_DIR / name).write_text(text)
        print(f"Table written to: {TABLE_DIR / name}")


# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test the Durrant-Kaban linear latent variable condition on "
                                                 "dense and SPLADE embeddings (Appendix C).")
    parser.add_argument("--n-docs", type=int, default=4000, help="Documents sampled per corpus")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--results", default=str(RESULTS_PATH))
    parser.add_argument("--tables-only", action="store_true", help="Re-write the tables from saved results")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.tables_only:
        saved = load_results(args.results)
        write_tables(saved["results"], saved["params"]["n_docs"])
        return
    embed_all(args.n_docs, args.seed, args.max_seq_length)
    rng = np.random.default_rng(args.seed)
    res = {}
    for corpus, _ in CORPORA:
        for key in [k for k, *_ in DENSE_MODELS] + ["splade"]:
            r = analyse(*load_embeddings(corpus, key), rng)
            res[f"{corpus}/{key}"] = r
            print(f"{corpus}/{key}: " + ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                                  for k, v in r.items()), flush=True)
    save_results({"results": res, "params": vars(args) | {"splade_model": SPLADE_MODEL}}, args.results)
    print(f"Results saved to: {args.results}")
    write_tables(res, args.n_docs)


if __name__ == "__main__":
    main()
