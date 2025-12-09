import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
import girg_sampling
from scipy.sparse import csr_matrix
from torch_sparse import SparseTensor
from torch.utils.cpp_extension import load_inline

def graph_to_csr_transition(G: nx.Graph):
    n = G.number_of_nodes()
    A = nx.to_scipy_sparse_array(G, nodelist=range(n), format="csr", dtype=float)
    row_sums = np.array(A.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    D_inv = 1.0 / row_sums
    A = A.multiply(D_inv[:, None])
    A = A.tocsr()
    return A.indptr.astype(np.uint32), A.indices.astype(np.uint32), A.data.astype(np.float32)

def _sample_neighbor(indices, weights, rng):
    weight_sum = weights.sum()
    if weight_sum <= 0:
        return int(rng.choice(indices))
    cdf = np.cumsum(weights)
    draw = rng.random() * cdf[-1]
    idx = np.searchsorted(cdf, draw)
    return int(indices[idx])

def method_1_numpy_walks(G, walk_length, num_walks_per_node):
    indptr, indices, data = graph_to_csr_transition(G)
    start_nodes = np.repeat(np.arange(G.number_of_nodes()), num_walks_per_node).astype(np.uint32)
    n_walks = len(start_nodes)
    walks = np.empty((n_walks, walk_length), dtype=np.uint32)
    rng = np.random.default_rng(42)

    t0 = time.perf_counter()
    
    for i in range(n_walks):
        step = int(start_nodes[i])
        walks[i, 0] = step

        for k in range(1, walk_length):
            row_start = indptr[step]
            row_end = indptr[step + 1]

            if row_start == row_end:
                walks[i, k] = step
                continue

            neigh_idx = indices[row_start:row_end]
            weights = data[row_start:row_end]
            
            next_step = _sample_neighbor(neigh_idx, weights, rng)
            step = int(next_step)
            walks[i, k] = step
            
    t1 = time.perf_counter()
    return walks, t1 - t0


def method_2_pyg_walks(G, walk_length, num_walks_per_node):
    adj_scipy = nx.to_scipy_sparse_array(G)
    rowptr = torch.from_numpy(adj_scipy.indptr).long()
    col = torch.from_numpy(adj_scipy.indices).long()
    adj = SparseTensor(rowptr=rowptr, col=col, sparse_sizes=(len(G), len(G)))
    start_nodes = torch.arange(G.number_of_nodes()).repeat_interleave(num_walks_per_node)
    
    t0 = time.perf_counter()
    
    walks_tensor = adj.random_walk(start_nodes, walk_length - 1)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    t1 = time.perf_counter()
    
    return walks_tensor.numpy(), t1 - t0

with open('random_walks.cpp', 'r') as f:
    cpp_source = f.read()

cpp_module = load_inline(
    name='cpp_walk_module',
    cpp_sources=cpp_source,
    functions=['walk_cpp_impl'],
    extra_cflags=['-O3', '-fopenmp'], # Optimization flags
    extra_ldflags=['-lgomp'],         # Link OpenMP
    verbose=True
)

def method_3_cpp_walks(G, walk_length, num_walks_per_node):
    # Prepare data in PyTorch format (CSR)
    adj_scipy = nx.to_scipy_sparse_array(G)
    rowptr = torch.from_numpy(adj_scipy.indptr).long()
    col = torch.from_numpy(adj_scipy.indices).long()
    start_nodes = torch.arange(G.number_of_nodes()).repeat_interleave(num_walks_per_node)

    t0 = time.perf_counter()
    
    walks = cpp_module.walk_cpp_impl(rowptr, col, start_nodes, walk_length, 42)
    
    t1 = time.perf_counter()
    return walks.numpy(), t1 - t0


def run_benchmark():
    walk_length = 100 
    num_walks_per_node = 10
    avg_degree = 15
    node_counts = [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000] 
    
    results = {"numpy": [], "pyg": [], "cpp": []}
    
    print(f"{'Nodes':<10} | {'NumPy':<12} | {'Pytorch (CPU)':<12} | {'C++ (OpenMP)':<12} | {'Speedup (PT/C++)'}")
    print("-" * 80)

    for n in node_counts:
        G = girg_sampling.girgs.generate_networkx_girg(n=n, ple=2.5, dim=2, deg=avg_degree, alpha=1000, seed=41)

        #_, t1 = method_1_numpy_walks(G, walk_length, num_walks_per_node)
        t1 = 1 # Placeholder for Method 1 timing (uncommenting method 1 is slow)
        _, t2 = method_2_pyg_walks(G, walk_length, num_walks_per_node)
        _, t3 = method_3_cpp_walks(G, walk_length, num_walks_per_node)
        
        results["numpy"].append(t1)
        results["pyg"].append(t2)
        results["cpp"].append(t3)
        
        speedup = t2 / t3
        print(f"{n:<10} | {t1:<12.4f} | {t2:<12.4f} | {t3:<12.4f} | {speedup:<10.1f}")

    # PLOTTING
    plt.figure(figsize=(10, 6))
    plt.plot(node_counts, results["numpy"], marker='o', label='Method 1: NumPy', linewidth=2)
    plt.plot(node_counts, results["pyg"], marker='s', label='Method 2: Pytorch (CPU)', linewidth=2)
    plt.plot(node_counts, results["cpp"], marker='^', label='Method 3: C++ (OpenMP)', linewidth=2)
    
    plt.title(f"Random Walk Benchmark\n(Walk Length: {walk_length}, Walks/Node: {num_walks_per_node})")
    plt.xlabel("Number of Nodes in Graph")
    plt.ylabel("Time (seconds)")
    plt.yscale("log")
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_benchmark()