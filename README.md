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

Wiki (~40K Vocabulary) NPMI values: 

Download matrix values in float16, window size 10, select minimum frequency (mf) and use easily.

vocab_index [original](https://static.preferred.ai/jiapeng/npmi_matrices/vocab2id.pkl), [lemma](https://static.preferred.ai/jiapeng/npmi_matrices/vocab2id_lemma.pkl)

mf=0 [original](https://static.preferred.ai/jiapeng/npmi_matrices/wiki_npmi_wsz10_mf0.npy), [lemma](https://static.preferred.ai/jiapeng/npmi_matrices/wiki_lemma_npmi_wsz10_mf0.npy)

mf=100 [original](https://static.preferred.ai/jiapeng/npmi_matrices/wiki_npmi_wsz10_mf100.npy), [lemma](https://static.preferred.ai/jiapeng/npmi_matrices/wiki_lemma_npmi_wsz10_mf100.npy)

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
      title = "{Aligning Human and Computational Coherence Evaluations}",
      journal = {Computational Linguistics},
      pages = {1-60},
      year = {2024},
      month = {08},
      abstract = "{Automated coherence metrics constitute an efficient and popular way to evaluate topic models. Previous work presents a mixed picture of their presumed correlation with human judgment. This work proposes a novel sampling approach to mining topic representations at a large scale while seeking to mitigate bias from sampling, enabling the investigation of widely used automated coherence metrics via large corpora. Additionally, this article proposes a novel user study design, an amalgamation of different proxy tasks, to derive a finer insight into the human decision-making processes. This design subsumes the purpose of simple rating and outlier-detection user studies. Similar to the sampling approach, the user study conducted is extensive, comprising 40 study participants split into eight different study groups tasked with evaluating their respective set of 100 topic representations. Usually, when substantiating the use of these metrics, human responses are treated as the gold standard. This article further investigates the reliability of human judgment by flipping the comparison and conducting a novel extended analysis of human response at the group and individual level against a generic corpus. The investigation results show a moderate to good correlation between these metrics and human judgment, especially for generic corpora, and derive further insights into the human perception of coherence. Analyzing inter-metric correlations across corpora shows moderate to good correlation among these metrics. As these metrics depend on corpus statistics, this article further investigates the topical differences between corpora, revealing nuances in applications of these metrics.}",
      issn = {0891-2017},
      doi = {10.1162/coli_a_00518},
      url = {https://doi.org/10.1162/coli\_a\_00518},
      eprint = {https://direct.mit.edu/coli/article-pdf/doi/10.1162/coli\_a\_00518/2467767/coli\_a\_00518.pdf},
  }




