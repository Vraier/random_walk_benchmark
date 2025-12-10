# pylint: disable=no-name-in-module,import-error
import time
import numpy as np
import networkx as nx

from random_walks import random_walks as _random_walks
from random_walks import random_walks_restart as _random_walks_restart
from random_walks import random_walks_no_backtrack as _random_walks_no_backtrack
from random_walks import random_walks_restart_no_backtrack as _random_walks_restart_no_backtrack
from random_walks import random_walks_periodic_restart as _random_walks_periodic_restart
from random_walks import random_walks_periodic_restart_no_backtrack as _random_walks_periodic_restart_no_backtrack
from random_walks import n2v_random_walks as _node2vec_random_walks

from random_walks import anonymize as _anonymize
from random_walks import anonymize_neighbors as _anonymize_neighbors

from preprocessing import get_normalized_adjacency, get_normalized_minimum_degree


def transition_probs(G, min_degree=False, sub_sampling=0.):
    assert not nx.is_directed(G), "Graph must be undirected"
    if min_degree:
        A, _ = get_normalized_minimum_degree(G)
    else:
        A, _ = get_normalized_adjacency(G, sub_sampling=sub_sampling)

    indptr = A.indptr.astype(np.uint32)
    indices = A.indices.astype(np.uint32)
    data = A.data.astype(np.float32)
    return indptr, indices, data


def _start_nodes(G, start_nodes):
    if start_nodes is None:
        start_nodes = np.arange(len(G.nodes)).astype(np.uint32)
    else:
        start_nodes = np.array(start_nodes, dtype=np.uint32)
    return start_nodes


def _seed(seed):
    if seed is None:
        seed = int(np.random.rand() * (2**32 - 1))
    return seed


def random_walks(
    G,
    n_walks=10,
    walk_len=10,
    min_degree=False,
    sub_sampling=0.,
    p=1, q=1, alpha=0, k=None,
    no_backtrack=False,
    start_nodes=None,
    seed=None,
    verbose=True
):
    """Generate random walks on a graph.

    Args:
        G (nx.Graph): Graph to walk on.
        n_walks (int): Number of walks per node.
        walk_len (int): Length of each walk.
        min_degree (bool): Whether to use minimum degree local rule.
        sub_sampling (float): Subsampling parameter for normalized adjacency.
        p (float): Return parameter.
        q (float): In-out parameter.
        alpha (float): Restart probability.
        k (int): Restart period.
        no_backtrack (bool): Whether to disallow backtracking.
        start_nodes (list): List of nodes to start walks from.
        seed (int): Random seed.
        verbose (bool): Whether to print progress.

    Returns:
        walks (np.ndarray): Random walk matrix of shape (n_start_nodes * n_walks, walk_len).
        restarts (np.ndarray): Restart matrix of shape (n_start_nodes * n_walks, walk_len).
    """
    start_time = time.time()

    indptr, indices, data = transition_probs(G, min_degree, sub_sampling)
    start_nodes = _start_nodes(G, start_nodes)
    seed = _seed(seed)

    if p == 1 and q == 1:
        if alpha == 0:
            if k is None:
                if no_backtrack:
                    walks = _random_walks_no_backtrack(indptr, indices, data, start_nodes, seed, n_walks, walk_len)
                else:
                    walks = _random_walks(indptr, indices, data, start_nodes, seed, n_walks, walk_len)
                restarts = np.zeros(walks.shape, dtype=bool)
            else:
                if no_backtrack:
                    walks = _random_walks_periodic_restart_no_backtrack(indptr, indices, data, start_nodes, seed, n_walks, walk_len, k)
                else:
                    walks = _random_walks_periodic_restart(indptr, indices, data, start_nodes, seed, n_walks, walk_len, k)
                restarts = np.zeros(walks.shape, dtype=bool)
                restarts[:, k::k] = True
        else:
            assert k is None, "Periodic restarts are not implemented for randomly restarting walks"
            if no_backtrack:
                walks, restarts = _random_walks_restart_no_backtrack(indptr, indices, data, start_nodes, seed, n_walks, walk_len, alpha)
            else:
                walks, restarts = _random_walks_restart(indptr, indices, data, start_nodes, seed, n_walks, walk_len, alpha)
    else:
        assert alpha == 0, "Restarts are not implemented for node2vec walks"
        assert k is None, "Periodic restarts are not implemented for node2vec walks"
        if no_backtrack:
            raise NotImplementedError("Non-backtracking is not implemented for node2vec walks")
        walks = _node2vec_random_walks(indptr, indices, data, start_nodes, seed, n_walks, walk_len, p, q)
        restarts = np.zeros(walks.shape, dtype=bool)

    if verbose:
        duration = time.time() - start_time
        print(f"Random walks - T={duration:.2f}s")

    return walks, restarts


def stationary_distribution(G, min_degree=False, sub_sampling=0.):
    """Compute the stationary distribution of a graph.

    Args:
        G (nx.Graph): Graph to compute stationary distribution for.
        min_degree (bool): Whether to use minimum degree local rule.
        sub_sampling (float): Subsampling parameter for normalized adjacency.

    Returns:
        probs (np.ndarray): Stationary distribution of shape (n_nodes,).
    """
    assert not nx.is_directed(G), "Graph must be undirected"
    if min_degree:
        _, probs = get_normalized_minimum_degree(G)
    else:
        _, probs = get_normalized_adjacency(G, sub_sampling=sub_sampling)
    return probs
