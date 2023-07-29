## Preferred Topic Metrics
**Large-Scale Correlation Analysis of Automated Metrics for Topic Models, ACL'23**

Accompanying code that made mining and evaluating **millions** of topic representations possible. 
For larger corpora, it is probably more efficient to compute counts once.

Most of the codebase was refactored and lightly tested on python 3.10 (in theory it should work on >=3.6).
Some functions were benchmarked for speed, using AMD EPYC 7502 @ 2.50GHz, using large Wikipedia graphs:
  1. 33 minutes to calculate 40K Wikipedia NPMI graphs from count graphs 
  2. 80 topics evaluated on NPMI / second from count graphs
 
More found in docstrings.

---
### Goals
Hackable: hopefully readable and extendable for **your own** use cases.

Lightweight: only numpy and pandas dependencies.

Speed: some attempts at computation efficiency.

---
### Features
<ol>
  <li>Topic evaluations</li>
  <li>Creating count statistics from corpus</li>
  <li>Mining Topic representations from corpora</li>
</ol> 

---
### To-do
<ol>
  <li>make it pip-able</li>
  <li>Some convenience functions</li>
</ol> 

--- 
### Releasable Resources
[Dropbox Link](https://www.dropbox.com/scl/fo/be5r4y9g76hlxnfvd4bqg/h?dl=0&rlkey=bbnnnxe9w8h77ln8vv7pfc8lx)

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
        abstract = "Automated coherence metrics constitute an important and popular way to evaluate topic models. Previous works present a mixed picture of their presumed correlation with human judgement. In this paper, we conduct a large-scale correlation analysis of coherence metrics. We propose a novel sampling approach to mine topics for the purpose of metric evaluation, and conduct the analysis via three large corpora showing that certain automated coherence metrics are correlated. Moreover, we extend the analysis to measure topical differences between corpora. Lastly, we examine the reliability of human judgement by conducting an extensive user study, which is designed as an amalgamation of different proxy tasks to derive a finer insight into the human decision-making processes. Our findings reveal some correlation between automated coherence metrics and human judgement, especially for generic corpora.",
    }
