import argparse
import os
from typing import List, Tuple

import numpy as np
import torch
from datasets import load_dataset
from scipy.sparse import csr_matrix, save_npz
from sentence_transformers import SparseEncoder
from tqdm import tqdm

MODEL_NAME = "naver/splade-v3"
SEED = 42


def build_csr_from_sparse_tensors(
    sparse_tensors: List[torch.Tensor],
    vocab_size: int,
) -> csr_matrix:
    """
    Convert a list of 1D sparse COO tensors (length V) into a CSR matrix (N x V).
    Each tensor corresponds to one document/query.
    """
    data = []
    indices = []
    indptr = [0]

    for t in sparse_tensors:
        t = t.coalesce()
        idx = t.indices()[0].cpu().numpy()
        vals = t.values().cpu().numpy()

        data.append(vals)
        indices.append(idx)
        indptr.append(indptr[-1] + len(vals))

    if len(data) == 0:
        return csr_matrix((0, vocab_size), dtype=np.float32)

    data = np.concatenate(data).astype(np.float32)
    indices = np.concatenate(indices).astype(np.int32)
    indptr = np.array(indptr, dtype=np.int64)

    return csr_matrix((data, indices, indptr), shape=(len(sparse_tensors), vocab_size))


def encode_texts_sparse(
    model: SparseEncoder,
    texts: List[str],
    batch_size: int = 32,
    is_query: bool = False,
) -> Tuple[csr_matrix, int]:
    """
    Encode a list of texts with SPLADE via SparseEncoder and return a CSR matrix.
    """
    all_sparse_tensors: List[torch.Tensor] = []
    vocab_size = None

    encode_fn = model.encode_query if is_query else model.encode_document

    for start in tqdm(
        range(0, len(texts), batch_size),
        desc="Encoding queries" if is_query else "Encoding corpus",
    ):
        batch = texts[start : start + batch_size]

        batch_embs = encode_fn(
            batch,
            batch_size=len(batch),
            convert_to_tensor=False,
            convert_to_sparse_tensor=True,
            show_progress_bar=False,
        )

        if vocab_size is None and len(batch_embs) > 0:
            vocab_size = batch_embs[0].shape[-1]

        all_sparse_tensors.extend(batch_embs)

    if vocab_size is None:
        vocab_size = (
            model.get_max_vocab_size()
            if hasattr(model, "get_max_vocab_size")
            else 30522
        )

    csr = build_csr_from_sparse_tensors(all_sparse_tensors, vocab_size=vocab_size)
    return csr, vocab_size


def get_output_paths(base_dir: str, dataset_slug: str):
    out_dir = os.path.join(base_dir, dataset_slug)
    os.makedirs(out_dir, exist_ok=True)

    corpus_path = os.path.join(out_dir, "corpus_splade.npz")
    query_path = os.path.join(out_dir, "queries_splade.npz")
    ids_corpus_path = os.path.join(out_dir, "corpus_ids.npy")
    ids_query_path = os.path.join(out_dir, "query_ids.npy")
    meta_path = os.path.join(out_dir, "meta.txt")

    return out_dir, corpus_path, query_path, ids_corpus_path, ids_query_path, meta_path


def embeddings_exist(
    corpus_path: str, query_path: str, ids_corpus_path: str, ids_query_path: str
) -> bool:
    return all(
        os.path.exists(p)
        for p in [corpus_path, query_path, ids_corpus_path, ids_query_path]
    )


def embed_hotpotqa(
    splade: SparseEncoder,
    max_docs: int,
    max_queries: int,
    batch_size: int,
    base_output_dir: str,
):
    print("\n=== HotpotQA (BeIR/hotpotqa) ===")
    (
        out_dir,
        corpus_path,
        query_path,
        ids_corpus_path,
        ids_query_path,
        meta_path,
    ) = get_output_paths(base_output_dir, "hotpotqa")

    if embeddings_exist(corpus_path, query_path, ids_corpus_path, ids_query_path):
        print(f"Embeddings already exist at {out_dir}, skipping HotpotQA.")
        return

    cache_dir = os.environ.get("HF_DATASETS_CACHE", None)

    print("Loading HotpotQA corpus and queries from Hugging Face...")
    corpus_ds = load_dataset(
        "BeIR/hotpotqa", "corpus", trust_remote_code=True, cache_dir=cache_dir
    )["corpus"]
    queries_ds = load_dataset(
        "BeIR/hotpotqa", "queries", trust_remote_code=True, cache_dir=cache_dir
    )["queries"]

    # Random subset for docs/queries (if limits set)
    if max_docs > 0 and max_docs < len(corpus_ds):
        corpus_ds = corpus_ds.shuffle(seed=SEED).select(range(max_docs))
    if max_queries > 0 and max_queries < len(queries_ds):
        queries_ds = queries_ds.shuffle(seed=SEED).select(range(max_queries))

    corpus_ids = corpus_ds["_id"]
    corpus_texts = [
        ((t or "") + " " + (x or "")).strip()
        for t, x in zip(corpus_ds["title"], corpus_ds["text"])
    ]

    query_ids = queries_ds["_id"]
    query_texts = queries_ds["text"]

    print(f"Loaded {len(corpus_texts)} HotpotQA docs and {len(query_texts)} queries.")

    print("Encoding HotpotQA corpus (documents) with SPLADE...")
    corpus_csr, vocab_size = encode_texts_sparse(
        splade, corpus_texts, batch_size=batch_size, is_query=False
    )

    print("Encoding HotpotQA queries with SPLADE...")
    query_csr, _ = encode_texts_sparse(
        splade, query_texts, batch_size=batch_size, is_query=True
    )

    print("Saving HotpotQA embeddings and IDs...")
    save_npz(corpus_path, corpus_csr)
    save_npz(query_path, query_csr)
    np.save(ids_corpus_path, np.array(corpus_ids))
    np.save(ids_query_path, np.array(query_ids))

    with open(meta_path, "w") as f:
        f.write(f"vocab_size={vocab_size}\n")
        f.write(f"num_docs={corpus_csr.shape[0]}\n")
        f.write(f"num_queries={query_csr.shape[0]}\n")
        f.write(f"model={MODEL_NAME}\n")
        f.write("dataset=BeIR/hotpotqa\n")

    print(f"HotpotQA done. Saved to {out_dir}.")


def embed_natural_questions(
    splade: SparseEncoder,
    max_docs: int,
    max_queries: int,
    batch_size: int,
    base_output_dir: str,
):
    (
        out_dir,
        corpus_path,
        query_path,
        ids_corpus_path,
        ids_query_path,
        meta_path,
    ) = get_output_paths(base_output_dir, "natural-questions")

    if embeddings_exist(corpus_path, query_path, ids_corpus_path, ids_query_path):
        print(f"Embeddings already exist at {out_dir}, skipping Natural Questions.")
        return

    cache_dir = os.environ.get("HF_DATASETS_CACHE", None)
    rng = np.random.default_rng(SEED)

    print("Loading Natural Questions from Hugging Face...")
    ds = load_dataset(
        "sentence-transformers/natural-questions",
        split="train",
        cache_dir=cache_dir,
    )

    all_queries = list(ds["query"])
    all_docs = list(ds["answer"])

    if max_queries > 0 and max_queries < len(all_queries):
        q_idx = rng.choice(len(all_queries), size=max_queries, replace=False)
        all_queries = [all_queries[i] for i in q_idx]

    if max_docs > 0 and max_docs < len(all_docs):
        d_idx = rng.choice(len(all_docs), size=max_docs, replace=False)
        all_docs = [all_docs[i] for i in d_idx]

    query_ids = np.arange(len(all_queries))
    corpus_ids = np.arange(len(all_docs))

    corpus_csr, vocab_size = encode_texts_sparse(
        splade, all_docs, batch_size=batch_size, is_query=False
    )

    query_csr, _ = encode_texts_sparse(
        splade, all_queries, batch_size=batch_size, is_query=True
    )

    save_npz(corpus_path, corpus_csr)
    save_npz(query_path, query_csr)
    np.save(ids_corpus_path, np.array(corpus_ids))
    np.save(ids_query_path, np.array(query_ids))

    with open(meta_path, "w") as f:
        f.write(f"vocab_size={vocab_size}\n")
        f.write(f"num_docs={corpus_csr.shape[0]}\n")
        f.write(f"num_queries={query_csr.shape[0]}\n")
        f.write(f"model={MODEL_NAME}\n")
        f.write("dataset=sentence-transformers/natural-questions\n")

    print(f"Natural Questions done. Saved to {out_dir}.")


def embed_trec_covid(
    splade: SparseEncoder,
    max_docs: int,
    max_queries: int,
    batch_size: int,
    base_output_dir: str,
):
    print("\n=== TREC-COVID (BeIR/trec-covid) ===")
    (
        out_dir,
        corpus_path,
        query_path,
        ids_corpus_path,
        ids_query_path,
        meta_path,
    ) = get_output_paths(base_output_dir, "trec-covid")

    if embeddings_exist(corpus_path, query_path, ids_corpus_path, ids_query_path):
        print(f"Embeddings already exist at {out_dir}, skipping TREC-COVID.")
        return

    cache_dir = os.environ.get("HF_DATASETS_CACHE", None)

    corpus_ds = load_dataset(
        "BeIR/trec-covid", "corpus", trust_remote_code=True, cache_dir=cache_dir
    )["corpus"]
    queries_ds = load_dataset(
        "BeIR/trec-covid", "queries", trust_remote_code=True, cache_dir=cache_dir
    )["queries"]

    if max_docs > 0 and max_docs < len(corpus_ds):
        corpus_ds = corpus_ds.shuffle(seed=SEED).select(range(max_docs))
    if max_queries > 0 and max_queries < len(queries_ds):
        queries_ds = queries_ds.shuffle(seed=SEED).select(range(max_queries))

    corpus_ids = corpus_ds["_id"]
    corpus_texts = [
        ((t or "") + " " + (x or "")).strip()
        for t, x in zip(corpus_ds["title"], corpus_ds["text"])
    ]

    query_ids = queries_ds["_id"]
    query_texts = queries_ds["text"]

    print(f"Loaded {len(corpus_texts)} TREC-COVID docs and {len(query_texts)} queries.")

    print("Encoding TREC-COVID corpus (documents) with SPLADE...")
    corpus_csr, vocab_size = encode_texts_sparse(
        splade, corpus_texts, batch_size=batch_size, is_query=False
    )

    print("Encoding TREC-COVID queries with SPLADE...")
    query_csr, _ = encode_texts_sparse(
        splade, query_texts, batch_size=batch_size, is_query=True
    )

    save_npz(corpus_path, corpus_csr)
    save_npz(query_path, query_csr)
    np.save(ids_corpus_path, np.array(corpus_ids))
    np.save(ids_query_path, np.array(query_ids))

    with open(meta_path, "w") as f:
        f.write(f"vocab_size={vocab_size}\n")
        f.write(f"num_docs={corpus_csr.shape[0]}\n")
        f.write(f"num_queries={query_csr.shape[0]}\n")
        f.write(f"model={MODEL_NAME}\n")
        f.write("dataset=BeIR/trec-covid\n")

    print(f"TREC-COVID done. Saved to {out_dir}.")


def embed_nfcorpus(
    splade: SparseEncoder,
    max_docs: int,
    max_queries: int,
    batch_size: int,
    base_output_dir: str,
):
    (
        out_dir,
        corpus_path,
        query_path,
        ids_corpus_path,
        ids_query_path,
        meta_path,
    ) = get_output_paths(base_output_dir, "nfcorpus")

    if embeddings_exist(corpus_path, query_path, ids_corpus_path, ids_query_path):
        print(f"Embeddings already exist at {out_dir}, skipping NFCorpus.")
        return

    cache_dir = os.environ.get("HF_DATASETS_CACHE", None)

    corpus_ds = load_dataset(
        "BeIR/nfcorpus", "corpus", trust_remote_code=True, cache_dir=cache_dir
    )["corpus"]
    queries_ds = load_dataset(
        "BeIR/nfcorpus", "queries", trust_remote_code=True, cache_dir=cache_dir
    )["queries"]

    if max_docs > 0 and max_docs < len(corpus_ds):
        corpus_ds = corpus_ds.shuffle(seed=SEED).select(range(max_docs))
    if max_queries > 0 and max_queries < len(queries_ds):
        queries_ds = queries_ds.shuffle(seed=SEED).select(range(max_queries))

    corpus_ids = corpus_ds["_id"]
    corpus_texts = [
        ((t or "") + " " + (x or "")).strip()
        for t, x in zip(corpus_ds["title"], corpus_ds["text"])
    ]

    query_ids = queries_ds["_id"]
    query_texts = queries_ds["text"]

    print(f"Loaded {len(corpus_texts)} NFCorpus docs and {len(query_texts)} queries.")

    print("Encoding NFCorpus corpus (documents) with SPLADE...")
    corpus_csr, vocab_size = encode_texts_sparse(
        splade, corpus_texts, batch_size=batch_size, is_query=False
    )

    print("Encoding NFCorpus queries with SPLADE...")
    query_csr, _ = encode_texts_sparse(
        splade, query_texts, batch_size=batch_size, is_query=True
    )

    print("Saving NFCorpus embeddings and IDs...")
    save_npz(corpus_path, corpus_csr)
    save_npz(query_path, query_csr)
    np.save(ids_corpus_path, np.array(corpus_ids))
    np.save(ids_query_path, np.array(query_ids))

    with open(meta_path, "w") as f:
        f.write(f"vocab_size={vocab_size}\n")
        f.write(f"num_docs={corpus_csr.shape[0]}\n")
        f.write(f"num_queries={query_csr.shape[0]}\n")
        f.write(f"model={MODEL_NAME}\n")
        f.write("dataset=BeIR/nfcorpus\n")

    print(f"NFCorpus done. Saved to {out_dir}.")


def embed_msmarco(
    splade: SparseEncoder,
    max_docs: int,
    max_queries: int,
    batch_size: int,
    base_output_dir: str,
):
    print("\n=== MSMARCO (ms_marco v2.1, validation split) ===")
    (
        out_dir,
        corpus_path,
        query_path,
        ids_corpus_path,
        ids_query_path,
        meta_path,
    ) = get_output_paths(base_output_dir, "msmarco-v2.1")

    if embeddings_exist(corpus_path, query_path, ids_corpus_path, ids_query_path):
        print(f"Embeddings already exist at {out_dir}, skipping MSMARCO.")
        return

    cache_dir = os.environ.get("HF_DATASETS_CACHE", None)
    rng = np.random.default_rng(SEED)

    # Load a slice of the validation split: n_queries + 100 (as in your example)
    if max_queries > 0:
        split_spec = f"validation[:{max_queries + 100}]"
    else:
        split_spec = "validation"

    print(f"Loading MSMARCO from Hugging Face with split='{split_spec}'...")
    ds = load_dataset(
        "ms_marco",
        "v2.1",
        split=split_spec,
        cache_dir=cache_dir,
    )

    # Queries
    all_queries = list(ds["query"])
    if max_queries > 0 and max_queries < len(all_queries):
        q_idx = rng.choice(len(all_queries), size=max_queries, replace=False)
        all_queries = [all_queries[i] for i in q_idx]

    # Docs: flatten all passages, deduplicate preserving order
    passages_column = ds["passages"]
    flat_passages = []
    for item in passages_column:
        for t in item["passage_text"]:
            flat_passages.append(t)

    docs_unique = list(dict.fromkeys(flat_passages))  # preserves order
    if max_docs > 0 and max_docs < len(docs_unique):
        d_idx = rng.choice(len(docs_unique), size=max_docs, replace=False)
        docs_unique = [docs_unique[i] for i in d_idx]

    print(
        f"Flattened {len(flat_passages)} passages -> {len(docs_unique)} unique docs "
        f"({len(all_queries)} queries)."
    )

    query_ids = np.arange(len(all_queries))
    corpus_ids = np.arange(len(docs_unique))

    print("Encoding MSMARCO corpus (docs) with SPLADE...")
    corpus_csr, vocab_size = encode_texts_sparse(
        splade, docs_unique, batch_size=batch_size, is_query=False
    )

    print("Encoding MSMARCO queries with SPLADE...")
    query_csr, _ = encode_texts_sparse(
        splade, all_queries, batch_size=batch_size, is_query=True
    )

    print("Saving MSMARCO embeddings and IDs...")
    save_npz(corpus_path, corpus_csr)
    save_npz(query_path, query_csr)
    np.save(ids_corpus_path, corpus_ids)
    np.save(ids_query_path, query_ids)

    with open(meta_path, "w") as f:
        f.write(f"vocab_size={vocab_size}\n")
        f.write(f"num_docs={corpus_csr.shape[0]}\n")
        f.write(f"num_queries={query_csr.shape[0]}\n")
        f.write(f"model={MODEL_NAME}\n")
        f.write("dataset=ms_marco v2.1 (validation)\n")

    print(f"MSMARCO done. Saved to {out_dir}.")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute SPLADE sparse embeddings for "
            "HotpotQA, Natural Questions, TREC-COVID, NFCorpus, and MSMARCO."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Base directory to write embeddings (datasets get their own subfolders).",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=500_000,
        help="Max number of corpus documents to encode (per dataset). Use -1 for all.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=10_000,
        help="Max number of queries to encode (per dataset). Use -1 for all.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for SPLADE encoding.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Device for encoding: "cuda", "cuda:0", "mps", or "cpu". Default: auto.',
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading SPLADE model ({MODEL_NAME})...")
    model_kwargs = {}
    if args.device is not None:
        model_kwargs["device"] = args.device

    splade = SparseEncoder(MODEL_NAME, **model_kwargs)

    embed_hotpotqa(
        splade=splade,
        max_docs=args.max_docs,
        max_queries=args.max_queries,
        batch_size=args.batch_size,
        base_output_dir=args.output_dir,
    )

    embed_natural_questions(
        splade=splade,
        max_docs=args.max_docs,
        max_queries=args.max_queries,
        batch_size=args.batch_size,
        base_output_dir=args.output_dir,
    )

    embed_trec_covid(
        splade=splade,
        max_docs=args.max_docs,
        max_queries=args.max_queries,
        batch_size=args.batch_size,
        base_output_dir=args.output_dir,
    )

    embed_nfcorpus(
        splade=splade,
        max_docs=args.max_docs,
        max_queries=args.max_queries,
        batch_size=args.batch_size,
        base_output_dir=args.output_dir,
    )

    embed_msmarco(
        splade=splade,
        max_docs=args.max_docs,
        max_queries=args.max_queries,
        batch_size=args.batch_size,
        base_output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
