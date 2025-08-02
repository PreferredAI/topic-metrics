import ctypes
import os
import math
import numpy as np
import pickle


from collections import defaultdict
from functools import reduce, partial
from multiprocessing import Pool, Array, Manager
from numpy.lib.stride_tricks import sliding_window_view
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
    None (edits shared array)
    """

    with open(f'{directory}/{f}', 'r') as f:
        for line in f:
            tokens = [vocab2id[w] for w in line.rstrip().split() if w in vocab2id]
            if len(tokens) <= window_size or window_size == 0:
                with glock:
                    shared_joint_array[np.ix_(tokens,tokens)] += 1
            else:
                ixs = [np.ix_(arr,arr) for arr in sliding_window_view(tokens, window_shape=window_size)]
                with glock:
                    for ix in ixs:
                        shared_joint_array[ix] += 1
            

def init_base(joint_base, size, lock):
    """Initializer for shared array between pool workers
    Parameters
    ----------
    single_base : Multiprocessing.Array
    joint_base : Multiprocessing.Array
    size : int
        size x size matrix
    """
    global glock
    global shared_joint_array
    glock = lock
    shared_joint_array = np.ctypeslib.as_array(joint_base.get_obj())
    shared_joint_array = shared_joint_array.reshape(size, size)


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
    shared_joint_base = Array(ctypes.c_long, len(vocab2id)**2)

    mgr = Manager()
    lock = mgr.Lock()
    with Pool(processes=count_processes, initializer=init_base,
              initargs=(shared_joint_base, len(vocab2id), lock)
              ) as pool:
        pool.map(partial(count_windows_helper,
                         directory=directory,
                         window_size=window_size, vocab2id=vocab2id),
                 tqdm(files))

    print(f'counting completed, {time()-inittime} seconds, dumping...')
    
    shared_joint_array = np.ctypeslib.as_array(shared_joint_base.get_obj())
    shared_joint_array = shared_joint_array.reshape(
        len(vocab2id), len(vocab2id))
    pickle.dump({i: v for i, v in enumerate(np.diag(shared_joint_array))},
                open(f"{destination}/single.pkl", "wb"))
    for k1 in range(len(vocab2id)):
        pickle.dump({k2: v for k2, v in enumerate(shared_joint_array[k1, :])
                     if v > 0},
                    open(os.path.join(dest_joint, f"{k1}.pkl"), "wb"))

    print('completed.', time()-inittime, 'seconds')
    print(f'Single prior counts saved to: {destination}/single.pkl')
    print(f'Joint co-occurrence counts saved to: {dest_joint}')
