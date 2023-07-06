import numpy as np
import random
import time

from functools import reduce
from .io_utils import *
from itertools import combinations
from multiprocessing import Pool
from queue import Queue
from tqdm import tqdm


def matrixfy_graph_dict(graph):
    """ Converts dictionary graph to matrix form with indexing
    Parameters
    ----------
    graph : Dict[int, Dict[int, Any]]

    Returns
    -------
    matrix: 2D numpy array
    index : Dict[int,int]
        mapping of vocab id to new index
    """
    sz = len(graph.keys())
    mat = np.zeros((sz, sz), dtype="bool")
    idx = {w: i for i, w in enumerate(graph.keys())}
    for w1, s1 in graph.items():
        mat[idx[w1], [idx[w2] for w2 in s1.keys()]] = 1
    return mat, idx


def DAGify_graph_mat(matrix):
    """ Converts a matrix into directed acyclic form
    Parameters
    ----------
    matrix: 2D numpy array

    Returns
    -------
    matrix: 2D numpy array
    """
    for i in range(len(matrix)):
        matrix[i, :i+1] = 0
    return matrix


def pre_core(graph, clique_size):
    """ Removes redudant vertices and edges
    Parameters
    ----------
    graph: Dict[int, Dict[int, Any]]
    clique_size : int

    Returns
    -------
    graph: Dict[int, Dict[int, Any]]
        with redundant vertices & edges removed

    Note
    ----
    Adapted from Yuan et al., 2022. Self implemented.
    """
    q = Queue()
    f = set()
    deg = {}
    for u in graph:
        deg[u] = len(graph[u])
        if len(graph[u]) < clique_size - 1:
            q.put(u)
            f.add(u)

    while not q.empty():
        u = q.get()
        for v in graph[u]:
            deg[v] -= 1
            if deg[v] < clique_size - 1 and v not in f:
                f.add(v)
                q.put(v)

    for u in f:
        graph[u] = {}
    for u in graph:
        graph[u] = {v: w for v, w in graph[u].items() if v not in f}
    return graph


def s_degreelist(remaining, clique, candidates, dag):
    """ Recursive depth first search for clique
    Parameters
    ----------
    remaining : int
        number of vertices left to be added
    clique : Set[int]
        current vertices in clique
    candidates : boolean 1d array
        potential candidates in clique
    dag: 2d array
        Directed acyclic matrix
    Returns
    -------
    clique: Set[int]
        Set of vocab ids (clique)

    Note
    ----
    Adapted from Yuan et al., 2022. Self implemented.
    """
    for u in np.random.permutation(np.nonzero(candidates)[0]):

        next_candidates = np.bitwise_and(dag[u], candidates)
        next_candidates_id = np.nonzero(next_candidates)[0]

        if len(candidates) <= remaining - 2:
            continue
        if remaining < 2:
            return []

        if remaining == 2 and len(next_candidates_id) > 0:
            output = clique.union([u])
            output.add(random.choices(next_candidates_id)[0])
            for o1, o2 in combinations(output, 2):
                dag[o1, o2] = 0
                dag[o2, o1] = 0
            return output

        elif len(next_candidates) > remaining - 2:
            travel = s_degreelist(
                remaining - 1, clique.union([u]), next_candidates, dag)
            if len(travel) != 0:
                return travel

    return set()


def s_degree(graph, clique_size, target=0):
    """ Recursive depth first search for cliques
    Parameters
    ----------
    graph: Dict[int, Dict[int, Any]]
    clique_size : int
    target : int
        number of cliques to be sampled

    Returns
    -------
    cliques: List[List[int]]
        List of topics (list of vocab ids)

    Note
    ----
    Adapted from Yuan et al., 2022. Self implemented.
    """
    graph = pre_core(graph, clique_size)
    dag, index = matrixfy_graph_dict(graph)
    dag = DAGify_graph_mat(dag)

    cliques = []
    vertices = list(range(len(dag)))
    random.shuffle(vertices)
    for u in vertices:
        result = s_degreelist(clique_size - 1, set([u]), dag[u], dag)
        if len(result) == clique_size:
            cliques.append(result)
        if target != 0 and len(cliques) >= target:
            break

    index_r = {v: k for k, v in index.items()}
    cliques = set([tuple(sorted([index_r[i] for i in s])) for s in cliques])
    return cliques


def prune_graph_using_edge_value(graph, condition):
    """
    Parameters
    ----------
    graph: Dict[int, Dict[int, Any]]
    condition: function(v) -> bool
        determines which edges to keep in graph

    Returns
    -------
    graph: Dict[int, Dict[int, Any]]
    """
    graph = {k1: {k2: v for k2, v in s.items() if condition(v)}
             for k1, s in graph.items()}
    graph = {k1: s for k1, s in graph.items() if len(s) > 0}
    return graph


def prune_graph_using_vertex(graph, keep):
    """
    Parameters
    ----------
    graph: Dict[int, Dict[int, Any]]
    keep: List[int]
        vertices to keep in graph

    Returns
    -------
    graph: Dict[int, Dict[int, Any]]
    """
    graph = {k1: {k2: v for k2, v in s.items() if k2 in keep}
             for k1, s in graph.items() if k1 in keep}
    return graph


def sample(key, graph_dir, clique_size, edge_condition, target=1):
    """ Sample cliques from sub-graph constrained by edge condition
    Parameters
    ----------
    key : int
        vocab id to shortlist sub-graph
    graph_dir : str
        directory path to all the co-occurence scored graph
    clique_size : int
    edge_condition : function(v) -> bool
        that takes in a value and returns bool
    target : int
        number of cliques to be sampled

    Returns
    -------
    List[List[int]]
        List of topics (list of vocab ids)
    """
    print('START:', key)
    start = time.time()
    keys = [k for k, v in load_graph(f"{graph_dir}/{key}.pkl").items()
            if edge_condition(v)] + [key]
    paths = [os.path.join(graph_dir, f'{k}.pkl') for k in keys]
    g = reduce(aggregate_loaded, tqdm(iload_id_and_graph(paths)), {})
    print(f'{key}\tLoaded:', time.time()-start)
    g = prune_graph_using_vertex(g, keys)
    g = prune_graph_using_edge_value(g, edge_condition)
    print(f'{key}\tPruned:', time.time()-start)
    sampled = s_degree(g, clique_size, target)
    print(f"{key}\t@{time.time()-start}s\t Sampled: {len(sampled)}")
    return sampled
