This repository lists the prototype algorithms our group developed for data collection/analysis under local differential privacy and is maintained by [Tianhao Wang](https://tianhao.wang/).  



## Protocols


### OLH
Frequency Oracle (primitive to estimate the histograms)

Related Paper: [Locally Differentially Private Protocols for Frequency Estimation](https://www.usenix.org/system/files/conference/usenixsecurity17/sec17-wang-tianhao.pdf)


### Square Wave
Density Oracle (for numerical/ordinal values)

Related Paper: [Estimating Numerical Distributions under Local Differential Privacy](https://arxiv.org/pdf/1912.01051)

Clarification: Citation 33 should be Ning Wang et al. Collecting and Analyzing Multidimensional Data with Local Differential Privacy. ICDE 2019.

### SVSM
Frequent Itemset Mining under LDP

Related Paper: [Locally Differentially Private Frequent Itemset Mining](https://ieeexplore.ieee.org/document/8418600)

Errata: In Equation (10) of Section V, there are three terms, two of them misses the coefficient $\ell$.

Clarification: To find top-k itemsets, we also consider singleton estimates from SVIM (the method for singleton mining).


### Post-Porcessing
A list of Post-Porcess Methods for LDP

Related Paper: [Locally Differentially Private Frequency Estimation with Consistency](https://www.ndss-symposium.org/wp-content/uploads/2020/02/24157.pdf)



### HDG
Multi-Dimensional Range Query under LDP

Related Paper: [Answering Multi-Dimensional Range Queries under Local Differential Privacy](https://arxiv.org/pdf/2009.06538.pdf)

Code available at [this repository](https://github.com/YangJianyu-bupt/privmdr).

### I am slowly cleaning code for the protocols below:


### PEM
Heavy Hitter Identification

Related Paper: [Locally Differentially Private Heavy Hitter Identification](https://arxiv.org/pdf/1708.06674.pdf)

Errata: For the AOL dataset, there are 0.12M, instead of 0.2M unique queries.  This is a typo that does not change any result.

Clarification: For the QUANTCAST data, I downloaded the data for one month (which contains 10 billion clicks), and sample for 5min (i.e., divide the # clicks by 30 * 24 * 5 * 60).  The dataset is available upon request.


### HIO 
Multi-Dimensional Analytics 

Related Paper: [Multi-Dimensional Analytics Related Paper: Answering Multi-Dimensional Analytical Queries under Local Differential Privacy](https://dl.acm.org/citation.cfm?id=3319891)


### CALM
Marginal Estimation 

The source code is not opened yet, but the similar code (plus a data synthesizing component) for the central DP setting is opened at [DPSyn](https://github.com/usnistgov/PrivacyEngCollabSpace/tree/master/tools/de-identification/Differential-Privacy-Synthetic-Data-Challenge-Algorithms/DPSyn) by [@Zhangzhk0819](https://github.com/Zhangzhk0819) (related info at [nist challenge 1](
https://www.nist.gov/communications-technology-laboratory/pscr/funding-opportunities/open-innovation-prize-challenges-2) and [nist challenge 2](https://www.nist.gov/communications-technology-laboratory/pscr/funding-opportunities/open-innovation-prize-challenges-1)).

Related Paper: [CALM: Consistent Adaptive Local Marginal for Marginal Release under Local Differential Privacy](https://dl.acm.org/citation.cfm?id=3243742)


### MURS
Shuffler Model

Related Paper: [Improving Utility and Security of the Shuffler-based Differential Privacy](http://www.vldb.org/pvldb/vol13/p3545-wang.pdf)


## Environment
I mainly used Python 3 with numpy.  Although not tested, the code should be compatible with any recent version.  I also use a special package called xxhash for high hashing throughput.  It can be changed to the builtin hash function.  In case your local environment does not work, this is the package versions I used from my local side.

Python 3.6

xxhash 1.0.1

numpy 1.11.3

pytest 3.4.0

Or, run
```
pip install -r requirements.txt
pytest
```


