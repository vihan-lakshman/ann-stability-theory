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

`make install` pulls only the core dependencies (numpy, scipy, pandas, matplotlib,
tqdm). That is everything needed to run all of the synthetic experiments and to
re-render every figure in `figures/`.

The experiments that need an ANN index backend or a neural encoder pull their
dependencies from optional extras:

| Extra | Packages | Used by |
| --- | --- | --- |
| `ann` | `faiss-cpu`, `hnswlib` | `algorithmic_stability.py` |
| `models` | `torch`, `transformers`, `datasets`, `sentence-transformers`, `beir` | `colbert_stability.py`, `theorem_validation.py`, `compute_splade_embeddings.py` |

```bash
make install-all     # core + both extras
```

The `make` targets for those experiments request their extra automatically, so
`uv` installs it on first use. Note that `faiss-cpu` publishes no wheel for
macOS older than 14; on such a machine `make install` still works and only the
`ann` extra is unavailable.

## Reproducing the figures

The Python version (`.python-version`) and every package version (`uv.lock`) are
pinned, and each experiment seeds NumPy per dimension, so a run reproduces the
committed figure byte for byte.

Each experiment writes its raw numbers to `results/<name>.json` next to the
figure, and accepts `--plot-only` to re-render the figure from that JSON without
re-running the experiment:

```bash
make synthetic-stability                                  # run + plot (~40 s)
uv run python multi-vector/synthetic_stability.py --plot-only   # plot only (<1 s)
make plots                                                # re-render every figure from results/
```

The multi-vector `results/*.json` files are committed, so
`figures/multivector_stability.{png,pdf}` and the three
`figures/multivector_distribution_*.{png,pdf}` can be regenerated with no
compute at all. Figures whose `results/*.json` is not committed (ColBERT,
filtered) have to be produced by running their experiment once; `make plots`
skips them with a message until then.

## Multi-Vector Search

### Run synthetic multi-vector stability experiments

```bash
make synthetic-stability
```

### Run the stability sweep across data distributions

```bash
make distribution-stability                          # default: lowrank-modes
make distribution-stability DISTRIBUTION=antipodal
make all-distributions                               # all three, ~45 s total
```

Sweeps the dimension from 4 to 32768 and compares Chamfer distance against mean
pooling under three synthetic data distributions (`--distribution`):

| Distribution | Construction |
| --- | --- |
| `iid` | Every vector an i.i.d. standard Gaussian draw; the baseline where both aggregations lose contrast as `d` grows. |
| `antipodal` | Every set contains `v` and `-v`, so mean pooling cancels; queries are noisy copies of documents. |
| `lowrank-modes` | Fixed semantic modes on the unit sphere with locally low-rank Gaussian variation; the closest to a real encoder. |

Each writes `results/multivector_distribution_<name>.json` and
`figures/multivector_distribution_<name>.{png,pdf}`. Distances are squared
Euclidean here, so the stability ratios are not directly comparable with
`synthetic_stability.py`, which uses cosine.

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
