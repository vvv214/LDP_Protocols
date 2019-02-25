[![Build Status](https://travis-ci.org/vvv214/LDP_Protocols.png?branch=master)](https://travis-ci.org/vvv214/LDP_Protocols)

## Protocols

### OLH
Sample OLH implementation in Python

Related Paper: Locally Differentially Private Protocols for Frequency Estimation 
([link](https://www.usenix.org/system/files/conference/usenixsecurity17/sec17-wang-tianhao.pdf))

### PEM (under construction)
Sample PEM implementation in Python

Related Paper: Locally Differentially Private Heavy Hitter Identification
([link](https://arxiv.org/pdf/1708.06674.pdf))

### SVIM/SVSM (under construction)
Sample SVIM/SVSM implementation in Python

Related Paper: Locally Differentially Private Frequent Itemset Mining
([link](https://www.computer.org/csdl/proceedings/sp/2018/4353/00/435301a578-abs.html))
Errata: In Equation (10) of Section V, there are three terms, two of them misses the coefficient $\ell$.
Clarification: To find top-k itemsets, we also consider singleton estimates from SVIM.

### CALM (under construction)
Sample CALM implementation in Python

Related Paper: CALM: Consistent Adaptive Local Marginal for Marginal Release under Local Differential Privacy
([link](https://dl.acm.org/citation.cfm?id=3243742))


## Environment
Python 2.7.10 (can also work for Python 3 by chaning the print statements)

xxhash 1.0.1

numpy 1.11.3

pytest 3.4.0

Or, run
```
pip install -r requirements.txt
pytest
```
