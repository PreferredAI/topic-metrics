import ctypes
import os
import math
import numpy as np
import pickle

from collections import defaultdict
from functools import reduce, partial
from itertools import combinations
from multiprocessing import Pool, Array
from time import time
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
        number of workers in Pool

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
    open(dest, 'w').write("\n".join(f"{item[0]},{item[1]}" for item in histograms.items()))
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
        number of workers in Pool

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
    pickle.dump(vocab, open(destination, 'wb'))
    print(f"vocab saved: {destination}")


def shortlist_vocab(vocab_count_location, upper_threshold=None, lower_threshold=None):
    """ Shortlist vocab based on count, mode 'auto' is similar to Hoyle et al., 2021
    Parameters
    ----------
    vocab_count_location : str
    upper_threshold : int
        ignore vocabulary with counts higher than
    lower_threshold : int
        ignore vocabulary with counts lower than

    Return
    ------
    vocab_counts : Dict[int, int]
        sorted map of vocab_id:count 
    """
    vocab_count = pickle.load(open(vocab_count_location, 'rb'))
    max_count = max(list(vocab_count.values()))
    
    ut = int(max_count * 0.9)
    lt = int(2 * (0.02*max_count) ** (1/math.log(10)))
    if upper_threshold != None:
        ut = upper_threshold
    if lower_threshold != None:
        lt = lower_threshold
        
    vocab_count = {k: v for k, v in vocab_count.items()
                   if v < ut and v > lt}
    if '' in vocab_count:
        del vocab_count['']

    return vocab_count


def count_windows_helper(f, directory, window_size, vocab2id):
    """ Count sliding windows from a file, each line is a document
    Parameters
    ----------
    f : str
        filename in directory
    directory : str
    window_size : int
        hyper-parameter for counting, sliding window size
    vocab2id : Dict[str, int]
        mapping of vocabulary word to its id

    Return
    ------
    single_count : Dict[int, int]
    joint_count : Dict[int, Dict[int, int]]
    """

    single_count = defaultdict(int)
    joint_count = defaultdict(lambda: defaultdict(int))
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
                joint_count[vocab2id[w1]][vocab2id[w2]] += 1
        else:
            convolved_bms = {}
            for w in words:
                bm = np.zeros(length, dtype=bool)
                bm[np.array(doc[w])] = True
                convolved_bms[w] = np.convolve(bm, MASK, mode='valid')
                single_count[vocab2id[w]] += convolved_bms[w].sum()

            for w1, w2 in combinations(words, 2):
                joint_count[vocab2id[w1]][vocab2id[w2]] += np.logical_and(
                    convolved_bms[w1], convolved_bms[w2]
                ).sum()

    single_count = {k1: v for k1, v in single_count.items() if v != 0}
    joint_count = {k1: {k2: v for k2, v in s.items() if v != 0}
                   for k1, s in joint_count.items()}

    words = list(single_count.keys())
    for w in words:
        if w not in joint_count:
            joint_count[w] = {}
    for w1, w2 in combinations(words, 2):
        rhs = joint_count[w2][w1] if w1 in joint_count[w2] else 0
        lhs = joint_count[w1][w2] if w2 in joint_count[w1] else 0
        if rhs + lhs == 0:
            continue
        joint_count[w2][w1] = joint_count[w1][w2] = rhs + lhs

    return single_count, joint_count


def init_bases(single_base, joint_base, size):
    """Initializer for shared array between pool workers
    Parameters
    ----------
    single_base : Multiprocessing.Array
    joint_base : Multiprocessing.Array
    size : int
        size x size matrix
    """
    global shared_single_array
    global shared_joint_array

    shared_single_array = np.ctypeslib.as_array(single_base.get_obj())
    shared_joint_array = np.ctypeslib.as_array(joint_base.get_obj())
    shared_joint_array = shared_joint_array.reshape(size, size)


def count_windows_multiprocessing(f, directory, window_size, vocab2id):
    """ helper for count_windows, require init
    Parameters
    ----------
    refer to count_windows_helper
    """
    single, joint = count_windows_helper(
        f, directory, window_size, vocab2id)
    for k1, v in single.items():
        shared_single_array[k1] += v

    for k1, s in joint.items():
        inc = np.zeros(len(shared_joint_array), dtype=np.int64)
        for k2, v in s.items():
            inc[k2] += v
        shared_joint_array[k1, :] += inc


def count_windows(directory, destination, window_size, vocab2id,
                  count_processes=4):
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
    count_processes : int
        number of workers in Pool to generate count graphs

    Return
    ------
    Save outputs to respective locations

    Note
    ----
    multiprocessing reference:
    https://stackoverflow.com/questions/5549190/
    """
    inittime = time()
    destination = os.path.join(destination, str(window_size))
    dest_joint = os.path.join(destination, 'joint')

    os.makedirs(destination, exist_ok=True)
    os.makedirs(dest_joint, exist_ok=True)

    files = [f for f in os.listdir(directory) if f.endswith('.txt')]
    shared_single_base = Array(ctypes.c_long, len(vocab2id))
    shared_joint_base = Array(ctypes.c_long, len(vocab2id)**2)

    with Pool(processes=count_processes, initializer=init_bases,
              initargs=(shared_single_base, shared_joint_base, len(vocab2id))
              ) as pool:
        pool.map(partial(count_windows_multiprocessing,
                         directory=directory,
                         window_size=window_size, vocab2id=vocab2id),
                 tqdm(files))

    print(f'counting completed, {time()-inittime} seconds, dumping...')
    shared_single_array = np.ctypeslib.as_array(shared_single_base.get_obj())
    shared_joint_array = np.ctypeslib.as_array(shared_joint_base.get_obj())
    shared_joint_array = shared_joint_array.reshape(
        len(vocab2id), len(vocab2id))

    pickle.dump({i: v for i, v in enumerate(shared_single_array)},
                open(f"{destination}/single.pkl", "wb"))
    for k1 in range(len(vocab2id)):
        pickle.dump({k2: v for k2, v in enumerate(shared_joint_array[k1, :])
                     if v > 0},
                    open(os.path.join(dest_joint, f"{k1}.pkl"), "wb"))

    print('completed.', time()-inittime, 'seconds')
    print(f'Single prior counts saved to: {destination}/single.pkl')
    print(f'Joint co-occurrence counts saved to: {dest_joint}')
