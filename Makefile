.PHONY: help install sync clean clean-plots algorithmic-stability filtered-stability \
        colbert-stability synthetic-stability theorem-validation sparse-synthetics \
        compute-splade validate-sparse-theorem all-experiments

.DEFAULT_GOAL := help

BLUE := \033[34m
GREEN := \033[32m
YELLOW := \033[33m
RESET := \033[0m

help:
	@echo "ANN-Stability Study for Sparse, Multi-vector and Filtered Search"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-25s$(RESET) %s\n", $$1, $$2}'

install: ## Install dependencies
	uv sync

sync: ## Sync dependencies
	uv sync

clean: ## Clean cache files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".DS_Store" -delete

clean-plots: ## Clean plot files
	find . -type f -name "*.png" -delete

# Real work begins here
# Target for running algorithmic stability experiments comparing
algorithmic-stability: ## Run main algorithmic stability experiments (HNSW vs IVF)
	@echo "$(BLUE)Running algorithmic stability experiments...$(RESET)"
	uv run python algorithmic_stability.py

filtered-stability: ## Run filtered search stability experiments
	@echo "$(BLUE)Running filtered stability experiments...$(RESET)"
	uv run python filtered/filtered_stability.py

colbert-stability: ## Run ColBERT real-world stability analysis
	@echo "$(BLUE)Running ColBERT stability analysis...$(RESET)"
	uv run python multi-vector/colbert_stability.py

synthetic-stability: ## Run multi-vector synthetic stability experiments
	@echo "$(BLUE)Running synthetic multi-vector stability experiments...$(RESET)"
	uv run python multi-vector/synthetic_stability.py

sparse-synthetics: ## Run CoI and overlap of importance analysis on synthetic embeddings
	@echo "$(BLUE)Running sparse synthetics experiments...$(RESET)"
	uv run python sparse/synthetics.py

# Theorem validation (requires dataset parameter)
theorem-validation: ## Validate Theorem 5.9 on real datasets (usage: make theorem-validation DATASET=msmarco)
ifndef DATASET
	@echo "$(YELLOW)Error: DATASET parameter required$(RESET)"
	@echo "Usage: make theorem-validation DATASET=<dataset>"
	@echo "Available datasets: msmarco, natural_questions, hotpotqa, trec_covid, nfcorpus"
	@exit 1
endif
	@echo "$(BLUE)Validating Theorem 5.9 on $(DATASET)...$(RESET)"
	uv run python multi-vector/theorem_validation.py $(DATASET)

compute-splade: ## Compute SPLADE embeddings (usage: make compute-splade OUTPUT=./embeddings MAX_DOCS=500000 MAX_QUERIES=10000)
	@echo "$(BLUE)Computing SPLADE embeddings...$(RESET)"
	uv run python sparse/compute_splade_embeddings.py \
		--output-dir $(or $(OUTPUT),./splade_embeddings) \
		--max-docs $(or $(MAX_DOCS),500000) \
		--max-queries $(or $(MAX_QUERIES),10000) \
		--batch-size $(or $(BATCH_SIZE),32) \
		--device $(or $(DEVICE),cuda)

# Sparse stability theorem validation (requires NPZ files)
validate-sparse-theorem: ## Validate Theorem 7.4 on sparse embeddings (usage: make validate-sparse-theorem CORPUS=corpus.npz QUERIES=queries.npz DATASET=msmarco)
ifndef CORPUS
	@echo "$(YELLOW)Error: CORPUS parameter required$(RESET)"
	@echo "Usage: make validate-sparse-theorem CORPUS=<corpus.npz> QUERIES=<queries.npz> DATASET=<name>"
	@echo "Optional parameters: P=2 SEED=0 STABILITY_QUERIES=2000 STABILITY_DOCS=2000"
	@echo "CSV output will be written to output/{DATASET}_results.csv"
	@exit 1
endif
ifndef QUERIES
	@echo "$(YELLOW)Error: QUERIES parameter required$(RESET)"
	@echo "Usage: make validate-sparse-theorem CORPUS=<corpus.npz> QUERIES=<queries.npz> DATASET=<name>"
	@echo "Optional parameters: P=2 SEED=0 STABILITY_QUERIES=2000 STABILITY_DOCS=2000"
	@echo "CSV output will be written to output/{DATASET}_results.csv"
	@exit 1
endif
	@echo "$(BLUE)Validating Theorem 7.4 on sparse embeddings...$(RESET)"
	uv run python sparse/sparse_stability.py \
		--corpus-npz $(CORPUS) \
		--queries-npz $(QUERIES) \
		--dataset-name $(or $(DATASET),DATASET) \
		$(if $(P),--p $(P),) \
		$(if $(SEED),--seed $(SEED),) \
		$(if $(STABILITY_QUERIES),--stability-queries $(STABILITY_QUERIES),) \
		$(if $(STABILITY_DOCS),--stability-docs $(STABILITY_DOCS),)

validate-all-datasets: ## Validate Theorem 5.9 on all available datasets
	@echo "$(BLUE)Validating on all datasets...$(RESET)"
	@for dataset in msmarco natural_questions hotpotqa trec_covid nfcorpus; do \
		echo "$(YELLOW)Validating on $$dataset...$(RESET)"; \
		uv run python multi-vector/theorem_validation.py $$dataset; \
	done
	@echo "$(GREEN)All dataset validations completed!$(RESET)"
