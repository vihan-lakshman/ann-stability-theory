"""
Empirically estimate p_max (filter mismatch probability) for Theorem 5.3 of:

  "Breaking the Curse of Dimensionality: On the Stability of Modern
   Vector Retrieval" (ICML 2026 submission)

Theorem 5.3: filtered search is stable when penalty  α > 2Δ / (1 - p_max).
With unit-normalised vectors Δ ≤ 2, so the threshold is  α > 4 / (1 - p_max).

════════════════════════════════════════════════════════════════
INSTALL
════════════════════════════════════════════════════════════════
    pip install datasets numpy tqdm

Both datasets stream from HuggingFace — no manual downloads needed.

════════════════════════════════════════════════════════════════
USAGE
════════════════════════════════════════════════════════════════
    python3 estimate_pmax.py --dataset arxiv
    python3 estimate_pmax.py --dataset redcaps
    python3 estimate_pmax.py --dataset both --selectivity all
    python3 estimate_pmax.py --dataset both --selectivity all --n_queries 1000 --max_docs 500000
"""

import argparse
import json
import random
import sys
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Filter predicates
# ─────────────────────────────────────────────────────────────────────────────

def sel_frac(selectivity, rng):
    # type: (str, random.Random) -> float
    return {"low":    lambda: rng.uniform(0.001, 0.01),
            "medium": lambda: rng.uniform(0.01,  0.10),
            "high":   lambda: rng.uniform(0.10,  1.00)}[selectivity]()


def range_mask(values, frac):
    # type: (np.ndarray, float) -> np.ndarray
    lo_pct = random.uniform(0.0, max(0.0, 1.0 - frac)) * 100
    hi_pct = min(lo_pct + frac * 100, 100.0)
    lo = float(np.percentile(values, lo_pct))
    hi = float(np.percentile(values, hi_pct))
    return (values >= lo) & (values <= hi)


def set_mask(labels, frac):
    # type: (np.ndarray, float) -> np.ndarray
    unique, counts = np.unique(labels, return_counts=True)
    order = np.argsort(counts)[::-1]
    chosen, cum = [], 0.0
    n = len(labels)
    for idx in order:
        if cum >= frac:
            break
        chosen.append(unique[idx])
        cum += counts[idx] / n
    return np.isin(labels, chosen)


PREDICATE_HELP = {
    "numeric1":    "range on 1st numeric attr",
    "numeric2":    "range on 2nd numeric attr",
    "categorical": "membership on categorical attr",
    "conjunction": "range1 AND range2  (2-attr)",
    "disjunction": "range1 OR  range2  (2-attr)",
    "conj_cat":    "range1 AND category (3-attr)",
}


# ─────────────────────────────────────────────────────────────────────────────
# Core estimator
# ─────────────────────────────────────────────────────────────────────────────

def estimate_pmax(num1, num2, cat, n_queries, selectivity, predicate, seed=42):
    rng = random.Random(seed)
    np.random.seed(seed)
    N = len(num1)
    qidx = rng.sample(range(N), min(n_queries, N))
    mismatches = []

    for _ in tqdm(qidx, desc="  %-14s [%s]" % (predicate, selectivity), leave=False):
        frac = sel_frac(selectivity, rng)
        fe   = min(frac ** 0.5, 1.0)

        if predicate == "numeric1":
            mask = range_mask(num1, frac)
        elif predicate == "numeric2":
            mask = range_mask(num2, frac)
        elif predicate == "categorical":
            mask = set_mask(cat, frac)
        elif predicate == "conjunction":
            mask = range_mask(num1, fe) & range_mask(num2, fe)
        elif predicate == "disjunction":
            fe2  = min(1 - (1 - frac) ** 0.5, 1.0)
            mask = range_mask(num1, fe2) | range_mask(num2, fe2)
        elif predicate == "conj_cat":
            mask = range_mask(num1, fe) & set_mask(cat, fe)
        else:
            raise ValueError(predicate)

        mismatches.append(1.0 - float(mask.mean()))

    arr   = np.array(mismatches)
    p_max = float(arr.max())
    # Theorem 5.3:  α > 2Δ/(1-p_max).  Δ=2 for unit-norm vectors → α > 4/(1-p_max)
    alpha = (4.0 / (1.0 - p_max)) if p_max < 1.0 else float("inf")
    return dict(
        predicate    = predicate,
        selectivity  = selectivity,
        n            = len(qidx),
        p_max        = round(p_max,        6),
        p_mean       = round(float(arr.mean()), 6),
        p_std        = round(float(arr.std()),  6),
        alpha        = round(alpha, 4) if alpha != float("inf") else float("inf"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# ArXiv  —  HuggingFace streaming  (arxiv-community/arxiv_dataset)
# ─────────────────────────────────────────────────────────────────────────────

def load_arxiv(max_docs, hf_cache=None):
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("Run:  pip install datasets")

    print("\n  Streaming ArXiv from HuggingFace (arxiv-community/arxiv_dataset) ...")
    kwargs = dict(split="train", streaming=True)
    if hf_cache:
        kwargs["cache_dir"] = hf_cache
    ds = load_dataset("gfissore/arxiv-abstracts-2021", **kwargs)

    years, cat_ids, cat2id = [], [], {}
    for i, rec in enumerate(tqdm(ds, total=max_docs, desc="  Records")):
        if i >= max_docs:
            break
        raw = rec.get("categories") or ""
        # categories may be a list (gfissore) or a space-separated string (arxiv-community)
        if isinstance(raw, list):
            primary = raw[0] if raw else "unknown"
        else:
            parts = raw.strip().split()
            primary = parts[0] if parts else "unknown"
        if primary not in cat2id:
            cat2id[primary] = len(cat2id)
        cat_ids.append(cat2id[primary])
        upd = (rec.get("update_date") or "")[:4]
        try:
            yr = int(upd)
            if not (1990 <= yr <= 2030):
                raise ValueError
        except ValueError:
            aid = rec.get("id") or ""
            try:
                yy = int(aid[:2]); yr = (2000 + yy) if yy < 50 else (1900 + yy)
            except (ValueError, IndexError):
                yr = 0
        years.append(yr)

    years_arr = np.array(years, dtype=np.float32)
    y_min, y_max = years_arr.min(), years_arr.max()
    norm_years = (years_arr - y_min) / (y_max - y_min + 1e-9) * 1000.0
    cat_arr = np.array(cat_ids, dtype=np.int32)
    print("  %d papers | %d categories | years %d-%d"
          % (len(years_arr), len(cat2id), int(y_min), int(y_max)))
    return years_arr, norm_years, cat_arr


def run_arxiv(args):
    num1, num2, cat = load_arxiv(args.max_docs, getattr(args, "hf_cache", None))
    results = []
    print()
    for pred in args.predicates:
        results.append(estimate_pmax(num1, num2, cat,
                                     args.n_queries, args.selectivity, pred, args.seed))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# RedCaps  —  HuggingFace streaming  (kdexd/red_caps)
#             Falls back to local ZIP if --redcaps_dir contains one.
# ─────────────────────────────────────────────────────────────────────────────

def load_redcaps_hf(max_docs, hf_cache=None):
    """Stream RedCaps annotations from HuggingFace (metadata only, no images)."""
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("Run:  pip install datasets")

    print("\n  Streaming RedCaps from HuggingFace (kdexd/red_caps) ...")
    kwargs = dict(split="train", streaming=True)
    if hf_cache:
        kwargs["cache_dir"] = hf_cache
    ds = load_dataset("kdexd/red_caps", **kwargs)

    scores, utcs, sr_ids, sr2id = [], [], [], {}
    for i, rec in enumerate(tqdm(ds, total=max_docs, desc="  Records")):
        if i >= max_docs:
            break
        sr = rec.get("subreddit") or "unknown"
        if sr not in sr2id:
            sr2id[sr] = len(sr2id)
        scores.append(float(rec.get("score") or 0))
        utcs.append(float(rec.get("created_utc") or 0))
        sr_ids.append(sr2id[sr])

    s = np.array(scores, dtype=np.float32)
    u = np.array(utcs,   dtype=np.float32)
    r = np.array(sr_ids, dtype=np.int32)
    print("  %d annotations | %d subreddits | score [%.0f, %.0f]"
          % (len(s), len(sr2id), s.min(), s.max()))
    return s, u, r


def load_redcaps_zip(redcaps_dir, max_docs):
    """Load RedCaps from a local annotations ZIP or already-extracted directory."""
    ann_dir  = Path(redcaps_dir) / "annotations"
    zip_path = Path(redcaps_dir) / "redcaps_v1.0_annotations.zip"

    if not ann_dir.exists() or not any(ann_dir.glob("*.json")):
        if zip_path.exists() and zipfile.is_zipfile(zip_path):
            print("  Extracting %s ..." % zip_path.name)
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(redcaps_dir)
        else:
            return None, None, None   # caller will fall back to HF

    json_files = sorted(ann_dir.glob("*.json"))
    if not json_files:
        return None, None, None

    print("\n  Loading RedCaps from local ZIP (%d files, limit %d) ..."
          % (len(json_files), max_docs))
    scores, utcs, sr_ids, sr2id = [], [], [], {}
    total = 0

    for jf in tqdm(json_files, desc="  Files"):
        if total >= max_docs:
            break
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        sr = (data.get("info") or {}).get("subreddit") or jf.stem.rsplit("_", 1)[0]
        if sr not in sr2id:
            sr2id[sr] = len(sr2id)
        sid = sr2id[sr]
        for ann in data.get("annotations", []):
            if total >= max_docs:
                break
            scores.append(float(ann.get("score", 0)))
            utcs.append(float(ann.get("created_utc", 0)))
            sr_ids.append(sid)
            total += 1

    if not total:
        return None, None, None

    s = np.array(scores, dtype=np.float32)
    u = np.array(utcs,   dtype=np.float32)
    r = np.array(sr_ids, dtype=np.int32)
    print("  %d annotations | %d subreddits | score [%.0f, %.0f]"
          % (total, len(sr2id), s.min(), s.max()))
    return s, u, r


def run_redcaps(args):
    # Try local ZIP first (faster if already downloaded), then HF streaming
    s, u, r = None, None, None
    if args.redcaps_dir:
        Path(args.redcaps_dir).mkdir(parents=True, exist_ok=True)
        s, u, r = load_redcaps_zip(args.redcaps_dir, args.max_docs)

    if s is None:
        s, u, r = load_redcaps_hf(args.max_docs, getattr(args, "hf_cache", None))

    if s is None:
        print("  [ERROR] Could not load RedCaps from any source.")
        return None

    results = []
    print()
    for pred in args.predicates:
        results.append(estimate_pmax(s, u, r,
                                     args.n_queries, args.selectivity, pred, args.seed))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(label, results):
    if not results:
        return
    W = 76
    print("\n" + "=" * W)
    print("  %s  --  Theorem 5.3  (Delta=2, unit-norm vectors)" % label)
    print("=" * W)
    print("  %-16s %-30s %-8s %-8s %s"
          % ("Predicate", "Description", "p_max", "p_mean", "alpha >"))
    print("  " + "-" * (W - 2))
    for r in results:
        desc  = PREDICATE_HELP.get(r["predicate"], "")
        alpha = ("%.3f" % r["alpha"]) if r["alpha"] != float("inf") else "inf"
        print("  %-16s %-30s %-8.4f %-8.4f %s"
              % (r["predicate"], desc, r["p_max"], r["p_mean"], alpha))

    finite = [r["alpha"] for r in results if r["alpha"] != float("inf")]
    if finite:
        print("\n  >> Worst-case: set  alpha > %.4f  for stability across all predicates." % max(finite))
        print("     (Theorem 5.3:  alpha > 2*Delta/(1-p_max) = 4/(1-p_max)  with Delta=2)")
    print("=" * W + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset",     choices=["arxiv", "redcaps", "both"], default="both")
    p.add_argument("--redcaps_dir", default=None,
                   help="Optional local dir with redcaps_v1.0_annotations.zip "
                        "(or already-extracted annotations/). Falls back to HF streaming.")
    p.add_argument("--hf_cache",    default=None,
                   help="Optional HuggingFace local cache dir")
    p.add_argument("--n_queries",   type=int, default=1000)
    p.add_argument("--max_docs",    type=int, default=500_000)
    p.add_argument("--selectivity", default="medium",
                   choices=["low", "medium", "high", "all"])
    p.add_argument("--predicates",  nargs="+",
                   default=["numeric1", "numeric2", "categorical", "conjunction"],
                   choices=list(PREDICATE_HELP))
    p.add_argument("--seed",        type=int, default=42)
    args = p.parse_args()

    sels = ["low", "medium", "high"] if args.selectivity == "all" else [args.selectivity]

    for sel in sels:
        args.selectivity = sel
        if args.dataset in ("arxiv", "both"):
            r = run_arxiv(args)
            if r:
                print_summary("ArXiv [%s]" % sel, r)
        if args.dataset in ("redcaps", "both"):
            r = run_redcaps(args)
            if r:
                print_summary("RedCaps [%s]" % sel, r)


if __name__ == "__main__":
    main()
