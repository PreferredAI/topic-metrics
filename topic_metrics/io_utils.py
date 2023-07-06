import os
import pickle

from functools import partial
from multiprocessing import Pool
from tqdm import tqdm


def aggregate_loaded(graph, item):
    graph[item[0]] = item[1]
    return graph


def load_graph(path):
    """ load .pkl dictionary graph

    Parameters
    ----------
    path : str
        filepath

    Return
    ------
    dict
        mapping of key,value
    """
    return pickle.load(open(path, 'rb'))


def iload_graph(paths):
    """ iterative loader for .pkl graph

    Parameters
    ----------
    paths : List[str]
        filepaths

    Yield
    ------
    dict
        mapping of key,value
    """
    for f in paths:
        yield load_graph(f)


def load_id_and_graph(path):
    """ load .pkl dictionary graph and name

    Parameters
    ----------
    path : str
        filepath

    Return
    ------
    str
        name of graph
    dict
        mapping of key,value
    """
    return int(path.split('/')[-1].rstrip('.pkl')), load_graph(path)


def iload_id_and_graph(paths):
    """ iterative loader for name and .pkl graph

    Parameters
    ----------
    paths : List[str]
        filepaths

    Yield
    ------
    str
        name of graph
    dict
        mapping of key,value
    """
    for f in paths:
        yield int(f.split('/')[-1].rstrip('.pkl')), load_graph(f)


def write_split_file(f, directory, destination, batch=100):
    """ Split one file into smaller batches

    Parameters
    ----------
    f : str
        filename in directory
    directory : str
    destination : str
        final destination directory
    batch : int
        determines number of documents in a batch

    Return
    ------
    Saves files split from main file
    """

    with open(f'{directory}/{f}', 'r') as r:
        docs = [line.rstrip() for line in r]
        batched_docs = [docs[i*batch:(i+1)*batch]
                        for i in range(len(docs)//batch+1)]
        for i, docs in enumerate(batched_docs):
            with open(f'{destination}/{i}_{f}', 'w') as w:
                w.write("\n".join(docs))


def split_corpus(directory, destination, batch, num_processes=1):
    """ Split many files into smaller batches

    Parameters
    ----------
    directory : str
    destination : str
        final destination directory
    batch : int
        determines number of documents in a batch
    num_processes : int
        number of wokers in Pool

    Return
    ------
    Saves files split from files in directory (corpus)
    """
    files = [f for f in os.listdir(directory) if f.endswith('.txt')]
    with Pool(processes=num_processes) as pool:
        pool.map(partial(write_split_file,
                         directory=directory,
                         destination=destination,
                         batch=batch),
                 tqdm(files))
