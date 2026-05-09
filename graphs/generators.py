import networkx as nx


def erdos_renyi(n: int, avg_degree: int, seed: int = 42) -> nx.Graph:
    p = avg_degree / (n - 1)
    return nx.fast_gnp_random_graph(n, p, seed=seed)
