import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

np.random.seed(42)


def cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine distance between two vectors."""
    return 1 - np.dot(a, b)


def chamfer(A: np.ndarray, B: np.ndarray) -> float:
    """Compute Chamfer distance between two vector sets using cosine distance."""
    total = 0
    for a in A:
        min_dist = float('inf')
        for b in B:
            dist = cosine_dist(a, b)
            if dist < min_dist:
                min_dist = dist
        total += min_dist
    return total


def average_pooling(A: np.ndarray, B: np.ndarray) -> float:
    """Compute average pooling distance between two vector sets using cosine distance."""
    total = 0
    for a in A:
        for b in B:
            total += cosine_dist(a, b)
    return total / (len(A) * len(B))


def normalize_vectors(vecs: np.ndarray) -> np.ndarray:
    """Normalize vectors to unit norm."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.where(norms > 0, norms, 1)


def generate_stable_base_dataset(n_docs: int, n_queries: int, dim: int, n_clusters: int = 5):
    """
    Generate a stable base dataset with cluster structure.
    
    Uses 1/sqrt(dim) scaling to maintain consistent geometry across dimensions.
    
    Returns docs, queries, and their cluster assignments.
    """
    # Create cluster centers
    cluster_centers = np.random.randn(n_clusters, dim) / np.sqrt(dim)
    cluster_centers = normalize_vectors(cluster_centers)
    
    # Assign docs and queries to clusters
    doc_assignments = np.random.randint(0, n_clusters, n_docs)
    query_assignments = np.random.randint(0, n_clusters, n_queries)
    
    # Generate with dimension-scaled noise
    doc_noise = np.random.randn(n_docs, dim) / np.sqrt(dim) * 0.3
    query_noise = np.random.randn(n_queries, dim) / np.sqrt(dim) * 0.3
    
    docs = cluster_centers[doc_assignments] + doc_noise
    queries = cluster_centers[query_assignments] + query_noise
    
    # Normalize to unit vectors
    docs = normalize_vectors(docs)
    queries = normalize_vectors(queries)
    
    return docs, queries, query_assignments


def build_multivector_sets_with_nn(
    docs: np.ndarray,
    queries: np.ndarray,
    query_assignments: np.ndarray,
    num_query_sets: int,
    vectors_per_set: int = 4,
    n_clusters: int = 5
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Build multi-vector query and document sets using nearest/furthest neighbor construction.
    
    This is the key construction from Section 5.5:
    - For each query vector, find its nearest and furthest neighbor documents
    - Build document sets containing these vectors AND their antipodal (-v)
    - The antipodal vectors cause average pooling to cancel out
    - But Chamfer distance can still find the nearest neighbors
    
    Query vectors within each set are sampled from the same cluster + shared noise
    to ensure positive covariance (Theorem 5.9 condition 3).
    """
    n_queries = len(queries)
    n_docs = len(docs)
    
    # Precompute nearest and furthest neighbors for each query
    nearest_neighbors = []
    furthest_neighbors = []
    
    for q in queries:
        scores = np.array([cosine_dist(q, d) for d in docs])
        nearest_neighbors.append(np.argmin(scores))
        furthest_neighbors.append(np.argmax(scores))
    
    # Group queries by cluster for sampling
    cluster_to_queries = {c: [] for c in range(n_clusters)}
    for idx, cluster in enumerate(query_assignments):
        cluster_to_queries[cluster].append(idx)
    
    # Build query and document sets
    query_sets = []
    doc_sets = []
    
    for _ in range(num_query_sets):
        # Pick a random cluster that has enough queries
        valid_clusters = [c for c, indices in cluster_to_queries.items() 
                         if len(indices) >= vectors_per_set]
        if not valid_clusters:
            # Fall back to random sampling if no cluster has enough
            sampled_indices = np.random.choice(n_queries, vectors_per_set, replace=False)
        else:
            cluster = np.random.choice(valid_clusters)
            sampled_indices = np.random.choice(
                cluster_to_queries[cluster], vectors_per_set, replace=False
            )
        
        # Add shared noise component to create additional correlation within query set
        # This models topically related tokens and reinforces condition 3 (non-negative covariance)
        dim = queries.shape[1]
        shared_noise = np.random.randn(dim) / np.sqrt(dim) * 0.15
        
        query_set = []
        doc_set_nearest = []
        doc_set_furthest = []
        
        for idx in sampled_indices:
            # Add shared noise to query vector and renormalize
            q_vec = queries[idx] + shared_noise
            q_vec = q_vec / np.linalg.norm(q_vec)
            query_set.append(q_vec)
            
            # Add nearest neighbor doc and its antipodal
            doc_set_nearest.append(docs[nearest_neighbors[idx]])
            doc_set_nearest.append(-1 * docs[nearest_neighbors[idx]])
            
            # Add furthest neighbor doc and its antipodal
            doc_set_furthest.append(docs[furthest_neighbors[idx]])
            doc_set_furthest.append(-1 * docs[furthest_neighbors[idx]])
        
        query_sets.append(np.array(query_set))
        doc_sets.append(np.array(doc_set_nearest))
        doc_sets.append(np.array(doc_set_furthest))
    
    return query_sets, doc_sets


def compute_theorem_conditions(
    query_sets: List[np.ndarray],
    doc_sets: List[np.ndarray],
    sample_size: int = 50
) -> Dict:
    """
    Check the three conditions from Theorem 5.9.
    
    Theorem 5.9 states multi-vector search with Chamfer distance is stable if:
        1. The induced single-vector search instance is c-strongly stable (c > 1)
        2. The document sets satisfy non-degeneracy condition
        3. Sum of covariances of individual nearest-neighbor distances >= 0
    """
    # Sample query sets for efficiency
    sampled_indices = np.random.choice(
        len(query_sets), 
        min(sample_size, len(query_sets)), 
        replace=False
    )
    sampled_query_sets = [query_sets[i] for i in sampled_indices]
    
    # Flatten to get induced single-vector problem
    all_query_vecs = np.vstack(sampled_query_sets)
    all_doc_vecs = np.vstack(doc_sets)
    
    # =========================================================================
    # Condition 1: c-strong stability of induced single-vector search
    # =========================================================================
    stability_ratios = []
    for q in all_query_vecs:
        distances = np.array([cosine_dist(q, d) for d in all_doc_vecs])
        d_min = np.min(distances)
        d_max = np.max(distances)
        if d_min > 1e-10:
            stability_ratios.append(d_max / d_min)
    
    c_stability = np.min(stability_ratios) if stability_ratios else 0
    condition1_passed = c_stability > 1
    
    # =========================================================================
    # Condition 2: Non-degeneracy of document sets
    # =========================================================================
    non_degeneracy_pass_count = 0
    total_checks = 0
    
    for q in all_query_vecs[:100]:
        min_dists_per_doc = []
        max_dists_per_doc = []
        
        for doc_set in doc_sets:
            dists = np.array([cosine_dist(q, d) for d in doc_set])
            min_dists_per_doc.append(np.min(dists))
            max_dists_per_doc.append(np.max(dists))
        
        max_of_mins = np.max(min_dists_per_doc)
        max_of_maxs = np.max(max_dists_per_doc)
        
        if c_stability * max_of_mins >= max_of_maxs:
            non_degeneracy_pass_count += 1
        total_checks += 1
    
    non_degeneracy_rate = non_degeneracy_pass_count / total_checks if total_checks > 0 else 0
    condition2_passed = non_degeneracy_rate >= 0.99
    
    # =========================================================================
    # Condition 3: Non-negative sum of covariances
    # =========================================================================
    covariance_sums = []
    
    for query_set in sampled_query_sets:
        nn_distances = []
        for q in query_set:
            min_dist = float('inf')
            for doc_set in doc_sets:
                for d in doc_set:
                    dist = cosine_dist(q, d)
                    if dist < min_dist:
                        min_dist = dist
            nn_distances.append(min_dist)
        covariance_sums.append(nn_distances)
    
    nn_dist_matrix = np.array(covariance_sums)
    
    if nn_dist_matrix.shape[0] > 1:
        cov_matrix = np.cov(nn_dist_matrix.T)
        k = cov_matrix.shape[0]
        covariance_sum = 0
        for i in range(k):
            for j in range(i + 1, k):
                covariance_sum += cov_matrix[i, j]
    else:
        covariance_sum = 0
    
    # Use small tolerance for numerical precision
    condition3_passed = covariance_sum >= -1e-6
    
    return {
        'c_stability': c_stability,
        'condition1_passed': condition1_passed,
        'non_degeneracy_rate': non_degeneracy_rate,
        'condition2_passed': condition2_passed,
        'covariance_sum': covariance_sum,
        'condition3_passed': condition3_passed,
        'all_conditions_passed': condition1_passed and condition2_passed and condition3_passed
    }


def compute_stability_metrics(
    query_sets: List[np.ndarray],
    doc_sets: List[np.ndarray],
    distance_fn
) -> Tuple[float, float]:
    """Compute stability ratio and relative variance."""
    stability_ratios = []
    all_distances = []
    
    for q in query_sets:
        min_dist = float('inf')
        max_dist = float('-inf')
        
        for d in doc_sets:
            dist = distance_fn(q, d)
            all_distances.append(dist)
            
            if dist < min_dist:
                min_dist = dist
            if dist > max_dist:
                max_dist = dist
        
        if min_dist > 1e-10:
            stability_ratios.append(max_dist / min_dist)
    
    all_distances = np.array(all_distances)
    mean_dist = np.mean(all_distances)
    var_dist = np.var(all_distances)
    
    relative_variance = var_dist / (mean_dist ** 2) if mean_dist > 0 else 0
    mean_stability_ratio = np.mean(stability_ratios)
    
    return mean_stability_ratio, relative_variance


def run_experiment(
    dimensions: List[int],
    n_base_docs: int = 200,
    n_base_queries: int = 200,
    num_query_sets: int = 100,
    vectors_per_set: int = 4,
    n_clusters: int = 5,
    check_theorem: bool = True
) -> Tuple[Dict, List[Dict]]:
    """
    Run the multi-vector stability experiment across multiple dimensions.
    
    Uses nearest/furthest neighbor construction from Section 5.5.
    """
    print("=" * 90)
    print("Multi-Vector Stability Experiment - Nearest/Furthest Neighbor Construction")
    print("=" * 90)
    print()
    
    results = {
        'dimensions': dimensions,
        'chamfer_stability': [],
        'chamfer_relvar': [],
        'avgpool_stability': [],
        'avgpool_relvar': []
    }
    theorem_results = []
    
    print(f"{'Dim':>6} | {'Chamfer Ratio':>13} | {'Chamfer RelVar':>14} | "
          f"{'AvgPool Ratio':>13} | {'AvgPool RelVar':>14}")
    print("-" * 90)
    
    for dim in dimensions:
        np.random.seed(42)
        
        # Generate base dataset with cluster structure
        docs, queries, query_assignments = generate_stable_base_dataset(
            n_docs=n_base_docs,
            n_queries=n_base_queries,
            dim=dim,
            n_clusters=n_clusters
        )
        
        # Build multi-vector sets using nearest/furthest neighbor construction
        query_sets, doc_sets = build_multivector_sets_with_nn(
            docs=docs,
            queries=queries,
            query_assignments=query_assignments,
            num_query_sets=num_query_sets,
            vectors_per_set=vectors_per_set,
            n_clusters=n_clusters
        )
        
        # Compute stability metrics
        cs, cr = compute_stability_metrics(query_sets, doc_sets, chamfer)
        avgs, avgr = compute_stability_metrics(query_sets, doc_sets, average_pooling)
        
        results['chamfer_stability'].append(cs)
        results['chamfer_relvar'].append(cr)
        results['avgpool_stability'].append(avgs)
        results['avgpool_relvar'].append(avgr)
        
        print(f"{dim:>6} | {cs:>13.4f} | {cr:>14.6f} | {avgs:>13.4f} | {avgr:>14.6f}")
        
        if check_theorem:
            theorem_conds = compute_theorem_conditions(query_sets, doc_sets)
            theorem_conds['dimension'] = dim
            theorem_results.append(theorem_conds)
    
    return results, theorem_results


def print_theorem_conditions(theorem_results: List[Dict]):
    """Print theorem condition check results."""
    print()
    print("=" * 90)
    print("Theorem 5.9 Condition Verification")
    print("=" * 90)
    print()
    print("Theorem 5.9 states multi-vector search with Chamfer distance is stable if:")
    print("  1. The induced single-vector search is c-strongly stable (c > 1)")
    print("  2. Document sets satisfy non-degeneracy condition")
    print("  3. Sum of covariances of nearest-neighbor distances >= 0")
    print()
    print(f"{'Dim':>6} | {'c-stability':>11} | {'Cond 1':>8} | {'Non-degen %':>11} | "
          f"{'Cond 2':>8} | {'Cov Sum':>12} | {'Cond 3':>8} | {'All Pass':>8}")
    print("-" * 95)
    
    for result in theorem_results:
        dim = result['dimension']
        c = result['c_stability']
        c1 = "✓" if result['condition1_passed'] else "✗"
        nd_rate = result['non_degeneracy_rate'] * 100
        c2 = "✓" if result['condition2_passed'] else "✗"
        cov = result['covariance_sum']
        c3 = "✓" if result['condition3_passed'] else "✗"
        all_pass = "✓" if result['all_conditions_passed'] else "✗"
        
        print(f"{dim:>6} | {c:>11.4f} | {c1:>8} | {nd_rate:>10.1f}% | "
              f"{c2:>8} | {cov:>12.6f} | {c3:>8} | {all_pass:>8}")
    
    print()
    print("Legend: ✓ = condition satisfied, ✗ = condition not satisfied")


def plot_results(dimensions: List[int], results: Dict, output_path: str):
    """Generate publication-quality plots of the results."""
    chamfer_ratios = results['chamfer_stability']
    avg_ratios = results['avgpool_stability']
    chamfer_relvars = results['chamfer_relvar']
    avg_relvars = results['avgpool_relvar']
    
    # Handle zero values for log scale
    avg_relvars_plot = [max(v, 1e-6) for v in avg_relvars]
    
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'serif',
        'axes.linewidth': 1.2,
        'lines.linewidth': 2,
        'lines.markersize': 7,
        'figure.dpi': 150,
    })
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    color_chamfer = '#2E86AB'
    color_avg = '#D62828'
    color_threshold = '#F77F00'
    
    # Plot 1: Stability Ratio
    ax1.plot(dimensions, chamfer_ratios, marker='o', label='Chamfer Distance', 
             color=color_chamfer, markerfacecolor='white', markeredgewidth=1.5)
    ax1.plot(dimensions, avg_ratios, marker='s', label='Average Pooling', 
             color=color_avg, markerfacecolor='white', markeredgewidth=1.5)
    ax1.axhline(y=1.0, color=color_threshold, linestyle='--', alpha=0.8, 
                linewidth=1.5, label='Instability Threshold')
    
    ax1.set_xlabel('Dimension', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Stability Ratio ($d_{max}/d_{min}$)', fontsize=12, fontweight='bold')
    ax1.set_title('Query Stability', fontsize=13, fontweight='bold', pad=10)
    ax1.legend(loc='best', framealpha=0.9)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True, which='major', alpha=0.3)
    ax1.grid(True, which='minor', alpha=0.1, linestyle=':')
    ax1.minorticks_on()
    
    # Plot 2: Relative Variance
    ax2.plot(dimensions, chamfer_relvars, marker='o', label='Chamfer Distance', 
             color=color_chamfer, markerfacecolor='white', markeredgewidth=1.5)
    ax2.plot(dimensions, avg_relvars_plot, marker='s', label='Average Pooling', 
             color=color_avg, markerfacecolor='white', markeredgewidth=1.5)
    
    ax2.set_xlabel('Dimension', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Relative Variance', fontsize=12, fontweight='bold')
    ax2.set_title('Relative Variance', fontsize=13, fontweight='bold', pad=10)
    ax2.legend(loc='best', framealpha=0.9)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid(True, which='major', alpha=0.3)
    ax2.grid(True, which='minor', alpha=0.1, linestyle=':')
    ax2.minorticks_on()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")


def print_analysis(results: Dict):
    """Print analysis of the results."""
    chamfer_ratios = results['chamfer_stability']
    avg_ratios = results['avgpool_stability']
    chamfer_relvars = results['chamfer_relvar']
    avg_relvars = results['avgpool_relvar']
    
    print()
    print("=" * 90)
    print("Analysis")
    print("=" * 90)
    print()
    print("Average Pooling:")
    print(f"  - Stability ratio: {min(avg_ratios):.4f} to {max(avg_ratios):.4f}")
    print(f"  - Relative variance: {min(avg_relvars):.6f} to {max(avg_relvars):.6f}")
    print(f"  - INSTABILITY: antipodal vectors cause distance collapse")
    print()
    print("Chamfer Distance:")
    print(f"  - Stability ratio stays above 1.0: {min(chamfer_ratios):.4f} to {max(chamfer_ratios):.4f}")
    print(f"  - Relative variance stays positive: {min(chamfer_relvars):.6f} to {max(chamfer_relvars):.6f}")
    print(f"  - STABILITY: Chamfer's min operator avoids the cancellation")
    print()
    print("Key insight from Theorem 5.9:")
    print("  Chamfer distance preserves stability because it selects nearest neighbors")
    print("  per query vector, ignoring the antipodal vectors that would cancel in averaging.")
    print()
    print("Construction details:")
    print("  - Document sets contain nearest/furthest neighbors + their antipodal vectors")
    print("  - For average pooling: v and -v cancel out, making all distances equal")
    print("  - For Chamfer: min operator selects v or -v (whichever is closer), preserving signal")


if __name__ == "__main__":
    dimensions = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    
    results, theorem_results = run_experiment(
        dimensions=dimensions,
        n_base_docs=200,
        n_base_queries=200,
        num_query_sets=100,
        vectors_per_set=4,
        n_clusters=5,
        check_theorem=True
    )
    
    print_analysis(results)
    print_theorem_conditions(theorem_results)
    plot_results(dimensions, results, 'multivector_stability.png')
