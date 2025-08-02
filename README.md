## Preferred Topic Metrics
**Large-Scale Correlation Analysis of Automated Metrics for Topic Models, ACL'23**

Accompanying code that made mining and evaluating **millions** of topic representations possible. 
For larger corpora, it is probably more efficient to compute counts once.

**Only using NPMI: see downloadables and snippet below.**

**Calculating from scratch:**
Most of the codebase was refactored and lightly tested on python 3.10 (in theory it should work on >=3.6).
Some functions were benchmarked for speed, using AMD EPYC 7502 @ 2.50GHz, using large Wikipedia graphs:
  1. 2 minutes to calculate 40K Wikipedia NPMI graphs from count graphs (see tutorial)
  2. 80 topics evaluated on NPMI / second from lazily loading count graphs (great for evaluating few topics)
  3. 30s to load 40K Wikipedia count graphs
  4. Very fast evaluation when count graphs are pre-loaded (300 topics/s with pre-loaded count graphs, see tutorial)
  5. 7-8 Hours to count Wikipedia in sliding windows (1B+ tokens total, 5M documents)
 
More found in docstrings.

---
### Goals
Hackable: hopefully readable and extendable for **your own** use cases.

Lightweight: only numpy and tqdm dependencies.

Speed: some attempts at computation efficiency.

---
### Features
<ol>
  <li>Topic evaluations</li>
  <li>Creating count statistics from corpus</li>
  <li>Mining Topic representations from corpora</li>
</ol> 

---
### To install

    pip install git+https://github.com/PreferredAI/topic-metrics.git

--- 
### Recommendations

We recommend setting a low window size (e.g 10) and minimum frequency (e.g. 0) for large corpora.

--- 
### Releasable Resources

Download matrix values in float16, window size 10, select minimum frequency (mf) and use easily.

Wiki (~40K Vocabulary) NPMI values (3.2GB): 

vocab_index [original](https://static.preferred.ai/jiapeng/npmi_matrices/vocab2id.pkl), [lemma](https://static.preferred.ai/jiapeng/npmi_matrices/vocab2id_lemma.pkl)

mf=0 [original](https://static.preferred.ai/jiapeng/npmi_matrices/wiki_npmi_wsz10_mf0.npy), [lemma](https://static.preferred.ai/jiapeng/npmi_matrices/wiki_lemma_npmi_wsz10_mf0.npy)

mf=100 [original](https://static.preferred.ai/jiapeng/npmi_matrices/wiki_npmi_wsz10_mf100.npy), [lemma](https://static.preferred.ai/jiapeng/npmi_matrices/wiki_lemma_npmi_wsz10_mf100.npy)

Wiki (~60K Vocabulary) NPMI values (8 GB) :

vocab_index [original](https://static.preferred.ai/jiapeng/npmi_matrices/wiki-large-vocab2id.pkl), [lemma](https://static.preferred.ai/jiapeng/npmi_matrices/wiki-large-vocab2id_lemma.pkl)

mf=0 [original](https://static.preferred.ai/jiapeng/npmi_matrices/wiki-large_npmi_wsz10_mf0.npy), [lemma](https://static.preferred.ai/jiapeng/npmi_matrices/wiki-large_lemma_npmi_wsz10_mf0.npy)

mf=100 [original](https://static.preferred.ai/jiapeng/npmi_matrices/wiki-large_npmi_wsz10_mf100.npy), [lemma](https://static.preferred.ai/jiapeng/npmi_matrices/wiki-large_lemma_npmi_wsz10_mf100.npy)


Example to use:
```
import numpy
import pickle
vocab2id = pickle.load(open(f'{data_dir}/vocab2id_lemma.pkl','rb'))
joint_npmi_mat = numpy.load(f"{data_dir}/wiki_lemma_npmi_wsz10_mf100.npy")
topic = ['apple','pear','banana','fruit']
from itertools import combinations # avoiding using code from this repo
print(numpy.mean([joint_npmi_mat[vocab2id[x1],vocab2id[x2]] for x1,x2 in combinations(topic,2) if x1 in vocab2id and x2 in vocab2id]))
# > 0.37
```

Original Counts: [Dropbox Link](https://www.dropbox.com/scl/fo/be5r4y9g76hlxnfvd4bqg/h?dl=0&rlkey=bbnnnxe9w8h77ln8vv7pfc8lx)

count_graph indices are mapped to alphabetically-sorted vocabulary while vocab_count maps are sorted by vocab count.

Example from Wiki's vocab-index:

    ...
    'addison': 724,
    'addition': 725,
    'additional': 726,
    'additionally': 727,
    'additions': 728,
    ...
    
---
[Anthology Link](https://aclanthology.org/2023.acl-long.776/)

If you had found the resources helpful, we'd appreciate a citation!

    @inproceedings{lim-lauw-2023-large,
        title = "Large-Scale Correlation Analysis of Automated Metrics for Topic Models",
        author = "Lim, Jia Peng  and
          Lauw, Hady",
        booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
        month = jul,
        year = "2023",
        address = "Toronto, Canada",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/2023.acl-long.776",
        pages = "13874--13898",
    }
    
    @article{10.1162/coli_a_00518,
      author = {Lim, Jia Peng and Lauw, Hady W.},
      title = {Aligning Human and Computational Coherence Evaluations},
      journal = {Computational Linguistics},
      volume = {50},
      number = {3},
      pages = {893-952},
      year = {2024},
      month = {09},
      issn = {0891-2017},
      doi = {10.1162/coli_a_00518},
      url = {https://doi.org/10.1162/coli\_a\_00518},
      eprint = {https://direct.mit.edu/coli/article-pdf/50/3/893/2471052/coli\_a\_00518.pdf},
}




