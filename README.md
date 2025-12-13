# ANN-Stability

This repo contains the code for analyzing ANN search stability across sparse, multi-vector and filtered search settings.

## Getting Started

Install `uv` package manager:

```bash
pip install uv
```

Or follow the [official uv documentation](https://docs.astral.sh/uv/) for installation.

Then sync dependencies:

```bash
make install
```

## Multi-Vector Search

### Run synthetic multi-vector stability experiments

```bash
make synthetic-stability
```

### Run ColBERT stability analysis on MS MARCO

```bash
make colbert-stability
```

### Validate Theorem 5.9 on a specific dataset

```bash
make theorem-validation DATASET=msmarco
```

Available datasets: `msmarco`, `natural_questions`, `hotpotqa`, `trec_covid`, `nfcorpus`

### Validate on all datasets

```bash
make validate-all-datasets
```

## Filtered Search


### Run filtered search stability experiments

```bash
make filtered-stability
```

This analyzes the impact of filter mismatch penalties on stability across varying penalty settings and dimensions.

## Sparse Search

Sparse search experiments validate theoretical stability guarantees for sparse embeddings.

### Run synthetic sparse search experiments

```bash
make sparse-synthetics
```

### Compute SPLADE embeddings for datasets

```bash
make compute-splade OUTPUT=./splade_embeddings MAX_DOCS=500000 MAX_QUERIES=10000
```

Optional parameters:
- `OUTPUT` - Output directory (default: `./splade_embeddings`)
- `MAX_DOCS` - Maximum documents to embed (default: `500000`)
- `MAX_QUERIES` - Maximum queries to embed (default: `10000`)
- `BATCH_SIZE` - Batch size for encoding (default: `32`)
- `DEVICE` - Device for computation (default: `cuda`)

### Validate Theorem 7.4 on sparse embeddings

```bash
make validate-sparse-theorem CORPUS=corpus.npz QUERIES=queries.npz DATASET=msmarco
```

Required parameters:
- `CORPUS` - Path to corpus sparse matrix (NPZ format)
- `QUERIES` - Path to queries sparse matrix (NPZ format)
- `DATASET` - Dataset name

Optional parameters:
- `P` - p-norm (default: `2`)
- `SEED` - Random seed (default: `0`)
- `STABILITY_QUERIES` - Queries to sample for stability (default: `2000`)
- `STABILITY_DOCS` - Documents to sample for stability (default: `2000`)

Results are automatically written to `output/{DATASET}_results.csv`.

## Other Experiments

### Run main algorithmic stability experiments (HNSW vs IVF)

```bash
make algorithmic-stability
```

### Run all synthetic experiments

```bash
make all-experiments
```
