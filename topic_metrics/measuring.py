import pandas as pd
import numpy as np
import os

from .counting import load_graph, load_id_and_graph, iload_id_and_graph
from functools import reduce, partial
from math import log
from multiprocessing import Pool
from tqdm import tqdm

EPS = EPSILON = 1e-12


def get_total_windows(histogram_path, window_size):
    """ Calculate number of sliding windows based on document lengths

    Parameters
    ----------
    histogram_path : str
        path of histogram file, csv format
    window_size : int
        hyper-parameter for counting, sliding window size

    Return
    ------
    num_windows: int
        total number of windows in corpus

    Note
    ----
    as per Palmetto (Röder et al., 2015)
    """
    histogram = pd.read_csv(histogram_path, header=None).values
    if window_size == 0:
        return histogram[:, 1].sum()
    return (histogram[:, 1] * np.maximum([1], histogram[:, 0] - (window_size - 1))).sum()


def pmi(p_a_b, p_a, p_b, smooth=True, default_to=0):
    """ Pointwise-mutual Information

    Parameters
    ----------
    p_a_b : float
        joint co-occurence probability for word a & b
    p_a : float
        probability for word a
    p_b : float
        probability for word b
    smooth : bool
        decides the use of epsilon=1e-12 or default
    default_to : float
        value used if smooth == False
    Return
    ------
    score : float
    """
    if smooth:
        p_a_b += EPS
    if p_a_b == 0 or p_a == 0 or p_b == 0:
        return default_to
    return log(p_a_b/(p_a * p_b))


def log_cond(p_a_b, p_a, p_b, smooth=True, default_to=0):
    """ Log. conditional, used in UMass (Mimno et al., 2011)

    Parameters
    ----------
    p_a_b: float
        joint co-occurence probability for word a & b
    p_a: float
        probability for word a, not used
    p_b: float
        probability for word b
    smooth : bool
        decides the use of epsilon=1e-12 or default
    default_to : float
        value used if smooth == False

    Return
    ------
    score : float

    Note
    ----
    word b is W* per Röder et al., 2015 for C_UMass
    """
    if smooth:
        p_a_b += EPS
    if p_a_b == 0 or p_b == 0:
        return default_to
    return log(p_a_b/p_b)


def fitelson(p_a_b, p_a, p_b, **kwargs):
    """ Fitelson's coherence (Fitelson, 2003)

    Parameters
    ----------
    p_a_b: float
        joint co-occurence probability for word a & b
    p_a: float
        probability for word a, not used
    p_b: float
        probability for word b

    Return
    ------
    score : float

    """
    p_a_given_b = p_a_b / p_b if p_b > 0 else 0
    p_a_given_not_b = (p_a - p_a_b) / (1 - p_b) if p_b < 1 else 0
    if p_a_given_b > 0 or p_a_given_not_b > 0:
        return (p_a_given_b - p_a_given_not_b) / (p_a_given_b + p_a_given_not_b)
    else:
        return 0


def npmi(p_a_b, p_a, p_b, smooth=True, default_to=0):
    """ Normalised Pointwise-mutual information (Bouma, 2009)

    Parameters
    ----------
    p_a_b: float
        joint co-occurence probability for word a & b
    p_a: float
        probability for word a
    p_b: float
        probability for word b
    smooth : bool
        decides the use of epsilon=1e-12 or default
    default_to : float
        value used if smooth == False
    Return
    ------
    score : float

    """
    if smooth:
        p_a_b += EPS
    if p_a == 0 or p_b == 0 or p_a_b == 0:
        return default_to
    return log(p_a_b / (p_a * p_b)) / -log(p_a_b)


def create_prob_graph(graph, num_window, min_freq):
    """ Create prob. graph from count graph fulfilling minimum freq. count

    Parameters
    ----------
    graph : Dict[int, int]
        count graph
    num_window : int
        num_window number of sliding windows in corpus
    min_freq : int
        lower bound to exclude rare words with counts less than

    Return
    ------
    probability graph : Dict[int, float]   
    """
    return {k: v/num_window if v >= min_freq else 0 for k, v in graph.items()}


def create_joint_prob_graph(graph, num_window, min_freq):
    """ Create joint co-occurrence prob. graph fulfilling minimum freq. count

    Parameters
    ----------
    graph : Dict[int, Dict[int, Any]]
        joint co-occurence count graph
    num_window : int
        num_window number of sliding windows in corpus
    min_freq : int
        lower bound to exclude rare word-pairs with counts less than

    Return
    ------
    joint co-occurence probability graph : Dict[int, Dict[int, float]]
    """
    return {
        i: create_prob_graph(j, num_window, min_freq)
        for i, j in graph.items()
    }


def create_graph_with(score_func, co_occ, occ, smooth=True, shortlist=[]):
    """ create a scored graph from probability graphs

    Parameters
    ----------
    score_func : function -> float
        Defined as f(p_a_b, p_a, p_b, **kwargs)
        Function takes in joint co-occ and prior word probabilities
        Generates a graph based on your scoring function
    co_occ: Dict[int, Dict[int, float]]
        joint co-occurence probability graph
    occ: Dict[int, float]
        prior probability graph
    smooth : bool
        Decides the use of epsilon=1e-12 or default if required
    shortlist : List[int]
        list of shortlisted vocab ids

    Return
    ------
    scored co-occurrence graph : Dict[int, Dict[int, float]]

    Benchmark
    ---------
    ~33 minutes to calculate 40K Wikipedia graphs using AMD EPYC 7502 @ 2.50GHz
    """
    if len(shortlist) > 0:
        co_occ = {s: {k2: v for k2, v in co_occ[s].items()
                      if k2 in shortlist}
                  for s in shortlist}
    graph = {
        i: {j: score_func(co_occ[i][j] if j in co_occ[i] else 0,
                          occ[i], occ[j], smooth=smooth)
            for j in co_occ.keys()}
        for i in co_occ.keys()
    }
    return graph


def direct_avg(topic, graph, error_default_to=0):
    """ Mean scores of ONE-PRE segmentation (Röder et al., 2015)

    Parameters
    ----------
    topic : List[int]
        list of vocab ids
    graph : Dict[int, Dict[int, float]]
        scored co-occurrence graph
    error_default_to: float
        how to score when co-occurence count is 0
        KeyError implies a 0 co-occurence count

    Returns
    -------
    score: float
        score of selected topics from graph

    Note
    ----
    topics sorted in descending manner based on p(w|topic)
    ONE-ONE segmentation can be transformed to ONE-PRE
    """
    scores = []
    for i in range(len(topic)):
        for j in range(i + 1, len(topic)):
            try:
                scores.append(graph[topic[j]][topic[i]])
            except KeyError:
                scores.append(error_default_to)
    return np.array(scores).mean()


def indirect_cv(topic, npmi_graph, gamma=1, error_default_to=0):
    """ Calculate C_V with gamma graph per Röder et al., 2015

    Parameters
    ----------
    topic : List[int]
        list of vocab ids
    npmi_graph : Dict[int, Dict[int, float]]
        scored NPMI co-occurrence graph
    gamma : int
        hyper-parameter affecting scaling of scores
    error_default_to: float
        how to score when co-occurence count is 0
        KeyError implies a 0 co-occurence count

    Returns
    -------
    score: float
        C_V score of selected topics from graph

    Note
    ----
    ONE-ONE segmentation can be transformed to ONE-PRE
    """
    graph = np.zeros((len(topic), len(topic)))
    for i in range(len(topic)):
        for j in range(i+1, len(topic)):
            try:
                graph[i][j] = npmi_graph[topic[j]][topic[i]] ** gamma
            except KeyError:
                graph[i][j] = error_default_to
            graph[j][i] = graph[i][j]
    graph += np.eye(len(graph))
    g2 = graph.sum(axis=0)  # for diag co-occ npmi
    return (graph.dot(g2) / (np.linalg.norm(graph, 2, axis=1) * np.linalg.norm(g2, 2,))).mean()

def simple_aggregate(graph, item):
    """ 
    Parameters
    ----------
    graph : Dict[int, Dict[int, float]]
    item : Tuple[int, Dict[int, float]]
        vocab id : count co-occurrences
    Returns
    -------
    graph: Dict[int, Dict[int, float]]
    """
    graph[item[0]] = item[1]
    return graph


def load_scored_graph(shortlist, graph_dir, agg_func, existing_graph={}):
    """ Directly load graph and score without additional processing

    Parameters
    ----------
    shortlist : List[int]
        list of shortlisted vocab ids
    graph_dir : str
        path to directory cotaining only .pkl joint graphs
    agg_func : function
        Defined as f(topics, graph, **kwargs)
        Aggregate the scores your way from the graph subset of selected topics
    existing_graph : Dict[int, Dict[int, float]]

    Returns
    -------
    joint co-occurence graph : Dict[int, Dict[int, float]]
    """

    if len(shortlist) == 0:
        paths = [f"{graph_dir}/{f}" for f in os.listdir(graph_dir)]
    else:
        paths = [f"{graph_dir}/{f}.pkl" for f in shortlist if os.path.exists(
            os.path.join(graph_dir, f"{f}.pkl"))]
    
    return agg_func(shortlist, reduce(simple_aggregate, iload_id_and_graph(paths),
                       existing_graph))

def calculate_scored_graphs(topics, graph_dir, agg_func, num_processes=1):
    """ Directly load graph and score topics without additional processing
    topics : List[int]
        list of shortlisted vocab ids
    graph_dir : str
        path to directory cotaining only .pkl joint graphs
    agg_func : function
        Defined as f(topics, graph, **kwargs)
        Aggregate the scores your way from the graph subset of selected topics
    num_processes : int
        number of wokers in Pool
    """
    with Pool(processes=num_processes) as pool:
        scores = pool.map(partial(load_scored_graph, 
                                  graph_dir=graph_dir, agg_func=agg_func),
            tqdm(topics))
    return scores
            


def aggregate_prob_graph(graph, item, num_windows, min_freq):
    """ 
    Parameters
    ----------
    graph : Dict[int, Dict[int, float]]
    item : Tuple[int, Dict[int, float]]
        vocab id : count co-occurrences
    num_windows: int
        total number of windows in corpus
    min_freq : int 
        lower bount to consider co-occurrence counts

    Returns
    -------
    graph: Dict[int, Dict[int, float]]
    """
    graph[item[0]] = create_prob_graph(item[1], num_windows, min_freq)
    return graph


def load_joint_prob_graph(graph_dir, num_windows, min_freq,
                          shortlist=[], num_processes=1,
                          existing_graph={}):
    """ Load and build probability graphs from count graphs

    Parameters
    ----------
    graph_dir : str
        path to directory cotaining only .pkl joint graphs
    num_windows: int
        total number of windows in corpus
    min_freq : int 
        lower bount to consider co-occurrence counts
    shortlist : List[int]
        list of shortlisted vocab ids
    num_processes : int
        number of wokers in Pool
    existing_graph : Dict[int, Dict[int, float]]

    Returns
    -------
    joint co-occurence probability graph : Dict[int, Dict[int, float]]

    Benchmark
    ---------
    Benchmarked using Wiki large graphs:
    150 graphs/s using AMD EPYC 7502 @ 2.50GHz with 20 workers
    100 graphs/s using ~ but with multiprocessing.Manager.dict()
    """
    if len(shortlist) == 0:
        paths = [f"{graph_dir}/{f}" for f in os.listdir(graph_dir)]
    else:
        paths = [f"{graph_dir}/{f}.pkl" for f in shortlist if os.path.exists(
            os.path.join(graph_dir, f"{f}.pkl"))]
    if num_processes == 1:
        graph = reduce(partial(aggregate_prob_graph, num_windows=num_windows,
                               min_freq=min_freq), iload_id_and_graph(paths),
                       existing_graph)
    else:
        with Pool(num_processes) as pool:
            graph = reduce(partial(aggregate_prob_graph, num_windows=num_windows,
                                   min_freq=min_freq), pool.imap(load_id_and_graph,
                                                                 tqdm(paths)), existing_graph)
    return graph


def C_NPMI():
    """
    {"score_func": npmi,
    'window_size': 10,
    "agg_func": direct_avg}
    """
    return {"score_func": npmi,
            'window_size': 10,
            "agg_func": direct_avg}


def C_P():
    """
    {"score_func": fitelson,
    'window_size': 70,
    "agg_func": direct_avg}
    """
    return {"score_func": fitelson,
            'window_size': 70,
            "agg_func": direct_avg}


def C_UMass():
    """
    {"score_func": log_cond,
    'window_size': 0,
    "agg_func": direct_avg}
    """
    return {"score_func": log_cond,
            'window_size': 0,
            "agg_func": direct_avg}


def C_V(gamma: int = 1):
    """
    {"score_func": npmi,
    'window_size': 110,
    "agg_func":  partial(indirect_cv, gamma=gamma)
    """
    return {"score_func": npmi,
            'window_size': 110,
            "agg_func":  partial(indirect_cv, gamma=gamma)}


def single_count_setup(histogram_path, single_count_path,
                       window_size, min_freq):
    """ 
    Parameters
    ----------
    histogram_path : str
        path of histogram file, csv format
    single_count_path : str
        path to .pkl containing prior counts
    window_size : int 
        hyper-parameter for counting, sliding window size
    min_freq : int 
        lower bount to consider co-occurrence counts
        auto uses window_size to determine
    Returns
    -------
    num_windows: int
        total number of windows in corpus
    single_prob: Dict[int,float]
        prior probability graph
    """
    num_windows = get_total_windows(histogram_path, window_size)
    single_prob = create_prob_graph(load_graph(
        single_count_path), num_windows, min_freq)
    return num_windows, single_prob


def calculate_score_from_counts(topic, single_prob, joint_count_path,
                    num_windows, min_freq, score_func,
                    agg_func, smooth):
    """ Caculate one topic score your way from count graphs

    Parameters
    ----------
    topic : List[int]
        list of vocab ids
    single_prob : Dict[int,float]
        prior probability graph
    joint_count_path : str
        path to directory cotaining only .pkl joint co-occurrence counts
    num_windows: int
        total number of windows in corpus
    min_freq : int 
        lower bount to consider co-occurrence counts
        auto uses window_size to determine
    score_func : function
        Defined as f(p_a_b, p_a, p_b, **kwargs)
        Function takes in joint co-occ and prior word probabilities
        Generates a graph based on your scoring function
    agg_func : function
        Defined as f(topics, graph, **kwargs)
        Aggregate the scores your way from the graph subset of selected topics
    smooth : bool
        Decides the use of epsilon=1e-12 or default

    Returns
    -------
    score: float
        Output of computation of your scoring and aggregate function
    """
    joint_prob = load_joint_prob_graph(joint_count_path,
                                       num_windows, min_freq, topic)
    graph = create_graph_with(
        score_func, joint_prob, single_prob, smooth, shortlist=set(topic))
    return agg_func(topic, graph)


def calculate_scores_from_counts(topics, histogram_path, single_count_path,
                     joint_count_path, score_func,
                     agg_func,
                     window_size, smooth=True, min_freq='auto',
                     num_processes=10):
    """ Caculate topics scores your way from count graphs

    Parameters
    ----------
    topics: List[List[int]]
        List of list of vocab ids
    histogram_path : str
        path of histogram file, csv format
    single_count_path : str
        path to .pkl containing prior counts
    joint_count_path : str
        path to directory cotaining only .pkl joint co-occurrence counts
    score_func : function
        Defined as f(p_a_b, p_a, p_b, **kwargs)
        Function takes in joint co-occ and prior word probabilities
        Generates a graph based on your scoring function
    agg_func : function
        Defined as f(topics, graph, **kwargs)
        Aggregate the scores your way from the graph subset of selected topics
    window_size : int 
        hyper-parameter for counting, sliding window size
    smooth : bool
        Decides the use of epsilon=1e-12 or default
    min_freq : int 
        lower bount to consider co-occurrence counts
        auto uses window_size to determine
    num_processes : int
        number of wokers in Pool

    Returns
    -------
    scores: List[float]
        Outputs of computation of your scoring and aggregate function

    Benchmark
    ---------
    Likely bottlenecked by OS I/O. 
    Loading graphs + calculation. Benchmarked using Wiki large graphs:
    60 topics/s using AMD EPYC 7502 @ 2.50GHz with 10 workers
    80 topics/s using AMD EPYC 7502 @ 2.50GHz with 20 workers
    60 topics/s using AMD EPYC 7502 @ 2.50GHz with 25 workers
    """

    if min_freq == 'auto':
        min_freq = max(10, window_size*10)
    num_windows, single_prob = single_count_setup(histogram_path, single_count_path,
                                                  window_size, min_freq)

    with Pool(processes=num_processes) as pool:
        scores = pool.map(partial(calculate_score_from_counts, single_prob=single_prob,
                                  joint_count_path=joint_count_path,
                                  num_windows=num_windows, min_freq=min_freq,
                                  score_func=score_func, agg_func=agg_func,
                                  smooth=smooth),
                          tqdm(topics))
    return scores


