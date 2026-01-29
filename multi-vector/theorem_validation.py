import sys
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import json
import warnings
from beir import util
from beir.datasets.data_loader import GenericDataLoader

warnings.filterwarnings('ignore')

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"

@dataclass
class ValidationConfig:
    """Configuration for validation."""
    n_queries: int = 100
    n_docs: int = 10000
    model_name: str = "colbert-ir/colbertv2.0"
    max_query_len: int = 64
    max_doc_len: int = 128


class TheoremValidator:
    def __init__(self, config: ValidationConfig, dataset):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading model: {config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModel.from_pretrained(config.model_name).to(self.device)
        self.model.eval()
        self.dataset = dataset
    
    def load_msmarco(self):
        ds = load_dataset("ms_marco", "v2.1", split=f"validation[:{self.config.n_queries + 100}]")
        queries = ds["query"][:self.config.n_queries]
        passages = [t for item in ds['passages'] for t in item['passage_text']]
        docs = list(dict.fromkeys(passages))[:self.config.n_docs]
        return queries, docs
    
    def load_natural_questions(self):
        ds = load_dataset("sentence-transformers/natural-questions", split='train')
        queries = ds["query"][:self.config.n_queries]
        docs = ds["answer"][:self.config.n_docs]
        return queries, docs
    
    def load_hotpotqa(self):
        dataset = "hotpotqa"
        url = BEIR_URL.format(dataset)
        data_path = util.download_and_unzip(url, "datasets")

        # BEIR HotpotQA only has train and test. The test split is the evaluation split.
        corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

        # Extract queries as strings
        query_strings = list(queries.values())

        # Extract documents as strings (title + text)
        document_strings = []
        for doc in corpus.values():
            title = doc.get("title", "")
            text = doc.get("text", "")
            combined = (title + " " + text).strip()
            document_strings.append(combined)
        
        queries = query_strings[:self.config.n_queries]
        docs = document_strings[:self.config.n_docs]

        return queries, docs

    def load_trec_covid(self):
        dataset = "trec-covid"
        url = BEIR_URL.format(dataset)
        data_path = util.download_and_unzip(url, "datasets")

        corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
        query_strings = list(queries.values())
        document_strings = []
        for doc in corpus.values():
            title = doc.get("title", "")
            text = doc.get("text", "")
            combined = (title + " " + text).strip()
            document_strings.append(combined)
        
        queries = query_strings[:self.config.n_queries]
        docs = document_strings[:self.config.n_docs]
        return queries, docs
    
    def load_nfcorpus(self):
        dataset = "nfcorpus"
        url = BEIR_URL.format(dataset)
        data_path = util.download_and_unzip(url, "datasets")

        corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
        query_strings = list(queries.values())

        document_strings = []
        for doc in corpus.values():
            title = doc.get("title", "")
            text = doc.get("text", "")
            combined = (title + " " + text).strip()
            document_strings.append(combined)
        
        queries = query_strings[:self.config.n_queries]
        docs = document_strings[:self.config.n_docs]
        return queries, docs
    
    def encode_batch(self, texts, max_len):
        """Encode a batch of texts."""
        inputs = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=max_len, return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            embs = self.model(**inputs).last_hidden_state
        embs = torch.nn.functional.normalize(embs, p=2, dim=-1)
        
        return embs, inputs.attention_mask.bool()
    
    def prepare_data(self, queries, docs):
        
        queries = list(set(queries)) # Ensure all queries are unique

        print("\nEncoding queries...")
        q_embs, q_masks = self.encode_batch(queries, self.config.max_query_len)
        
        print("Encoding documents...")
        d_embs, d_masks = self.encode_batch(docs, self.config.max_doc_len)
        
        # Extract query sets (each query is a SET of token vectors)
        query_sets = []
        for i in range(q_embs.shape[0]):
            tokens = q_embs[i][q_masks[i]]  # [n_tokens, dim]
            query_sets.append(tokens)
        
        # Extract document sets (each document is a SET of token vectors)
        doc_sets = []
        for i in range(d_embs.shape[0]):
            tokens = d_embs[i][d_masks[i]]  # [n_tokens, dim]
            doc_sets.append(tokens)
        
        # Create D' = flattened database of all document tokens (for induced problem)
        # Note that this might create duplicates while mathematically we assume the elements of D_prime are unique
        # However, this doesn't matter in practice since it won't change the stability ratio.
        D_prime = torch.cat(doc_sets, dim=0)  # [total_tokens, dim]

        return query_sets, doc_sets, D_prime
    
    # =========================================================================
    # CONDITION 1: Strong Stability of Induced Single-Vector Problem
    # =========================================================================
    
    def check_condition_1_induced_stability(self, query_sets, D_prime):
        
        all_ratios = []
        
        for q_idx, Q in enumerate(tqdm(query_sets, desc="Processing query sets")):
            for token_idx in range(Q.shape[0]):
                q = Q[token_idx]  # Single query token [dim]
                
                # Compute distances to ALL document tokens in D'
                sims = torch.matmul(q, D_prime.T)  # [|D'|]
                dists = 1.0 - sims
                
                dmin = dists.min().item()
                dmax = dists.max().item()
                
                if dmin > 1e-8:
                    ratio = dmax / dmin
                    all_ratios.append(ratio)
        
        ratios = np.array(all_ratios)
        
        estimated_c = np.min(ratios)
        is_stable = estimated_c > 1
        
        assert is_stable

        return estimated_c
    # =========================================================================
    # CONDITION 2: Non-Degeneracy over Document Sets
    # =========================================================================
    
    def check_condition_2_non_degeneracy(self, query_sets, doc_sets, c):
        margins = []
        lhs_values = []
        rhs_values = []
        
        n_docs = len(doc_sets)
        
        # For each query token
        for Q in tqdm(query_sets, desc="Checking non-degeneracy"):
            for token_idx in range(Q.shape[0]):
                q = Q[token_idx]  # Single query vector [dim]
                min_dists_per_doc = []  # min dist to each document set
                max_dists_per_doc = []  # max dist to each document set
                
                for D_k in doc_sets:
                    sims = torch.matmul(q, D_k.T)  # [|D_k|]
                    dists = 1.0 - sims
                    
                    min_dists_per_doc.append(dists.min().item())
                    max_dists_per_doc.append(dists.max().item())
                
                # "The farthest nearest-neighbor across all document sets"
                lhs = c * max(min_dists_per_doc)
                
                # "The farthest token across all document sets"
                rhs = max(max_dists_per_doc)
                
                lhs_values.append(lhs)
                rhs_values.append(rhs)
                margins.append(lhs - rhs)
        
        margins = np.array(margins)
        lhs_arr = np.array(lhs_values)
        rhs_arr = np.array(rhs_values)
        
        satisfaction_rate = np.mean(margins >= 0)
        
        assert satisfaction_rate > 0.99

    # =========================================================================
    # CONDITION 3: Non-Negative Covariance
    # =========================================================================

    def check_condition_3_bounded_covariance(self, query_sets, doc_sets):
        n_docs = len(doc_sets)
        total_sum_cov = []
        
        for Q in tqdm(query_sets, desc="Computing covariances"):
            all_covariances = []
            k = Q.shape[0]  # Number of tokens in this query set
            
            A_matrix = torch.zeros(k, n_docs)
            
            for doc_idx, D in enumerate(doc_sets):
                # Distance from each query token to each token in D
                sims = torch.matmul(Q, D.T)  # [k, |D|]
                dists = 1.0 - sims
                if dists.dim() == 1:
                    dists = dists.unsqueeze(0) 
                A_matrix[:, doc_idx] = dists.min(dim=1)[0]
            
            A_matrix = A_matrix.cpu().numpy()  # [k, n_docs] 
            cov_matrix = np.cov(A_matrix)  # [k, k]
            
            for i in range(k):
                for j in range(i + 1, k):
                    cov_ij = cov_matrix[i, j]
                    all_covariances.append(cov_ij)
            
            sum_cov = sum(all_covariances) 
            total_sum_cov.append(sum_cov)
        
        # Check that the smallest of the sum of the covariances is non-negative which implies they all are
        is_positive = np.min(total_sum_cov) >= 0

        assert is_positive
        
    # =========================================================================
    # Full Validation
    # =========================================================================
    
    def run_validation(self) -> Dict:
        print("="*70)
        print(f"\nConfiguration:")
        print(f"  Queries: {self.config.n_queries}")
        print(f"  Documents: {self.config.n_docs}")
        
        # Load and prepare data
        if self.dataset == "natural_questions":
            queries, docs = self.load_natural_questions()
        elif self.dataset == "trec_covid":
            queries, docs = self.load_trec_covid()
        elif self.dataset == "nfcorpus":
            queries, docs = self.load_nfcorpus()
        elif self.dataset == "hotpotqa":
            queries, docs = self.load_hotpotqa()
        elif self.dataset == "msmarco":
            queries, docs = self.load_msmarco()
        else:
            raise ValueError("Invalid dataset choice")

        query_sets, doc_sets, D_prime = self.prepare_data(queries, docs)
        
        # Check Condition 1: Strong stability of induced problem
        estimated_c = self.check_condition_1_induced_stability(query_sets, D_prime)
        
        print(f"Strong Stability passed. The estimated constant c is: {estimated_c}")

        # Check Condition 2: Non-degeneracy over document sets
        self.check_condition_2_non_degeneracy(query_sets, doc_sets, estimated_c)
        
        print("Non-Degeneracy Passed")

        # Check Condition 3: Non-negative covariance within query sets
        self.check_condition_3_bounded_covariance(query_sets, doc_sets)
        
        print("Non-Negative Covariance Sum Passed")

def main():
    dataset = sys.argv[-1]
    config = ValidationConfig()
    
    validator = TheoremValidator(config, dataset)
    results = validator.run_validation()
    

if __name__ == "__main__":
    main()
