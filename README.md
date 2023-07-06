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

Lightweight: only numpy and pandas dependences.

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
Count graphs of Wiki, Wiki-lemma.
Due to some outstanding issues with our S3, the current best way is to grab the large files via a meetup at ACL.
Hopefully, we can resolve the issues soon.

---

Citation to be generated
