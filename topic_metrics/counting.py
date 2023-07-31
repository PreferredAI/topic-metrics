import os
import math
import numpy as np
import pandas as pd
import pickle
import shutil

from collections import defaultdict
from .io_utils import *
from functools import reduce, partial
from itertools import combinations
from multiprocessing import Pool
from tqdm import tqdm


def aggregate_count(d1, d2):
    for k, v in d2.items():
        d1[k] += v
    return d1


def aggregate_count_nested(d1, d2):
    for k1, s in d2.items():
        for k2, v in s.items():
            d1[k1][k2] += v
    return d1


def build_histogram(f, directory):
    """ Builds histogram from file

    Parameters
    ----------
    f : str
        filename in directory
    directory : str

    Return
    ------
    histogram: Dict[int, int]
        mapping of len to count
    """
    hist = defaultdict(int)
    with open(os.path.join(directory, f), 'r') as f:
        for line in f:
            hist[len(line.rstrip().split())] += 1
    return dict(hist)


def count_histogram(directory, destination, num_processes):
    """ Build & save histogram from directory (corpus)

    Parameters
    ----------
    directory : str
    destination : str
        final destination directory
    num_processes : int
        number of wokers in Pool

    Return
    ------
    Save a mapping of length:count in .csv format
    """
    files = [f for f in os.listdir(directory) if f.endswith('.txt')]

    with Pool(processes=num_processes) as pool:
        histograms = pool.map(
            partial(build_histogram, directory=directory), tqdm(files))
    histograms = dict(
        reduce(aggregate_count, tqdm(histograms), defaultdict(int)))

    dest = os.path.join(destination, "histogram.csv")
    pd.DataFrame(histograms.items()).to_csv(dest, header=None, index=None)
    print("histogram saved:", dest)


def build_vocab_counts(f, directory):
    """ Count vocabulary from file

    Parameters
    ----------
    f : str
        filename in directory
    directory : str

    Return
    ------
    vocab_counts : Dict[int, int]
        mapping of vocab to count
    """
    vocab = defaultdict(int)
    with open(f"{directory}/{f}", 'r', encoding="utf8", errors='ignore') as f:
        for line in f:
            for t in line.rstrip().split(" "):
                vocab[t] += 1
    return dict(vocab)


def count_vocab(directory, destination, num_processes=1):
    """ Count & save vocabulary from directory (corpus)
    Parameters
    ----------
    directory : str
    destination : str
        final destination directory
    num_processes : int
        number of wokers in Pool

    Return
    ------
    Save a mapping of vocab_id:count in .pkl format
    """
    files = [f for f in os.listdir(directory) if f.endswith('.txt')]

    with Pool(processes=num_processes) as pool:
        vocab = pool.map(partial(build_vocab_counts,
                                 directory=directory), tqdm(files))

    vocab = dict(reduce(aggregate_count, tqdm(vocab), defaultdict(int)))
    vocab = {k: v for k, v in sorted(vocab.items(), key=lambda x: x[0])}
    pickle.dump(vocab, open(f"{destination}/vocab_count.pkl", 'wb'))
    print("vocab saved:", f"{destination}/vocab_count.pkl")


def shortlist_vocab(vocab_count_location, auto=True, upper_threshold=None, lower_threshold=None):
    """ Shortlist vocab based on count, mode 'auto' is similar to Hoyle et al., 2021
    Parameters
    ----------
    vocab_count_location : str
    auto : bool
        Set to False if thresholds are defined
    upper_threshold : int
        ignore vocabulary with counts higher than
    lower_thresohld : int
        ignore vocabulary with counts lower than

    Return
    ------
    vocab_counts : Dict[int, int]
        sorted map of vocab_id:count 
    """
    vocab_count = pickle.load(open(vocab_count_location, 'rb'))
    max_count = max(list(vocab_count.values()))

    if upper_threshold == None:
        upper_threshold = max_count+1
    if lower_threshold == None:
        lower_threshold = 0
    if auto:
        upper_threshold = int(max_count * 0.9)
        lower_threshold = int(2 * (0.02*max_count) ** (1/math.log(10)))
    vocab_count = {k: v for k, v in vocab_count.items()
                   if v < upper_threshold and v > lower_threshold}

    return vocab_count


def count_windows_helper(f, directory, dest_single, dest_joint, window_size, vocab2id):
    """ Count sliding windows from a file, each line is a document
    Parameters
    ----------
    f : str
        filename in directory
    directory : str
    dest_single : str
        final destination file for prior count .pkl
    dest_joint : str
        final destination file for joint co-occurence .pkl
    window_size : int
        hyper-parameter for counting, sliding window size
    vocab2id : Dict[str, int]
        mapping of vocabulary word to its id

    Return
    ------
    Save outputs to respective locations
    """
    single_path = os.path.join(dest_single, f"single_count_{f}.pkl")
    joint_path = os.path.join(dest_joint, f"joint_count_{f}.pkl")
    if os.path.exists(joint_path) and os.path.exists(single_path):
        print(f"EXISTS: {joint_path}")
        return 0

    single_count = defaultdict(int)
    pair_count = defaultdict(lambda: defaultdict(int))
    MASK = np.ones(window_size, dtype=bool)

    with open(f'{directory}/{f}', 'r') as f:
        docs = [line.rstrip() for line in f]

    for i in range(len(docs)):
        tokens = docs[i].split()
        length = len(tokens)
        doc = defaultdict(list)
        for i, t in enumerate(tokens):
            if t in vocab2id:
                doc[t].append(i)
        words = list(doc.keys())

        if window_size >= length or window_size == 0:
            for w in words:
                single_count[vocab2id[w]] += 1
            for w1, w2 in combinations(words, 2):
                pair_count[vocab2id[w1]][vocab2id[w2]] += 1
        else:
            convolved_bms = {}
            for w in words:
                bm = np.zeros(length, dtype=bool)
                bm[np.array(doc[w])] = True
                convolved_bms[w] = np.convolve(bm, MASK, mode='valid')
                single_count[vocab2id[w]] += convolved_bms[w].sum()

            for w1, w2 in combinations(words, 2):
                pair_count[vocab2id[w1]][vocab2id[w2]] += np.logical_and(
                    convolved_bms[w1], convolved_bms[w2]
                ).sum()

    single_count = {k1: v for k1, v in single_count.items() if v != 0}
    pair_count = {k1: {k2: v for k2, v in s.items() if v != 0}
                  for k1, s in pair_count.items()}

    words = list(single_count.keys())
    for w in words:
        if w not in pair_count:
            pair_count[w] = {}
    for w1, w2 in combinations(words, 2):
        rhs = pair_count[w2][w1] if w1 in pair_count[w2] else 0
        lhs = pair_count[w1][w2] if w2 in pair_count[w1] else 0
        if rhs + lhs == 0:
            continue
        pair_count[w2][w1] = pair_count[w1][w2] = rhs + lhs

    pickle.dump(pair_count, open(joint_path, "wb"))
    pickle.dump(single_count, open(single_path, "wb"))


def count_windows(directory, destination, window_size, vocab2id, count_processes=4, load_processes=4):
    """ Pipeline for counting sliding windows in a directory (corpus)
    Parameters
    ----------
    directory : str
    destination : str
        final destination directory
    window_size : int
        hyper-parameter for counting, sliding window size
    vocab2id : Dict[str, int]
        mapping of vocabulary word to its id
    num_processes : int
        number of wokers in Pool

    Return
    ------
    Save outputs to respective locations
    """
    destination = os.path.join(destination, str(window_size))
    dest_temp_single = destination + '_single_temp'
    dest_temp_joint = destination + '_joint_temp'
    dest_joint = os.path.join(destination, 'joint')

    os.makedirs(destination, exist_ok=True)
    os.makedirs(dest_temp_single, exist_ok=True)
    os.makedirs(dest_temp_joint, exist_ok=True)
    os.makedirs(dest_joint, exist_ok=True)

    files = [f for f in os.listdir(directory) if f.endswith('.txt')]

    with Pool(processes=count_processes) as pool:
        pool.map(partial(count_windows_helper,
                         directory=directory,
                         dest_single=dest_temp_single,
                         dest_joint=dest_temp_joint,
                         window_size=window_size, vocab2id=vocab2id),
                 tqdm(files))

    print('pre-counting completed, post-processing...')

    single_graph_paths = [f'{dest_temp_single}/{f}' for f in os.listdir(dest_temp_single)
                          if f.endswith('.pkl')]

    with Pool(processes=load_processes) as pool:
        single_graph = reduce(aggregate_count,
                              pool.imap_unordered(
                                  load_graph, tqdm(single_graph_paths)),
                              defaultdict(int))
    pickle.dump(single_graph, open(f"{destination}/single.pkl", 'wb'))

    print('single counts completed, joint counting...')

    joint_graph_paths = [f'{dest_temp_joint}/{f}' for f in os.listdir(dest_temp_joint)
                         if f.endswith('.pkl')]

    with Pool(processes=load_processes) as pool:
        joint_graph = reduce(aggregate_count_nested,
                             pool.imap_unordered(
                                 load_graph, tqdm(joint_graph_paths)),
                             defaultdict(lambda: defaultdict(int)))

    print('joint counts completed, dumping...')

    for k1, s in joint_graph.items():
        pickle.dump(dict(s), open(os.path.join(dest_joint, f"{k1}.pkl"), "wb"))

    print('cleaning...')
    shutil.rmtree(dest_temp_single)
    shutil.rmtree(dest_temp_joint)
    print('completed.')
    print(f'Single prior counts saved to: {destination}/single.pkl')
    print(f'Joint co-occurrence counts saved to: {dest_joint}')
