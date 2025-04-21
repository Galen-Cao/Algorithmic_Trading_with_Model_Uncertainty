# Algorithmic_Trading_with_Model_Uncertainty
This repository contains the implementation of the robust optimal strategies proposed in the paper: [Cartea, Á., Donnelly, R., &amp; Jaimungal, S. (2017). Algorithmic Trading with Model Uncertainty. SIAM Journal on Financial Mathematics, 8(1), 635–671](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2310645). 

## Overview
The code in this repository reproduces the robust market making strategies derived in the paper and the associated numerical simulations. It also includes a comparison between the robust strategy and a plain optimal (non-robust) strategy. 

## Features

- Solves the dynamic programming equations for both robust and non-robust market making strategies.
- Implements the numerical scheme for solving the HJB/HJBI equations as described in the paper.
- Simulates a realistic trading environment to evaluate performance.
- Compares the Sharpe ratios of robust and plain strategies using Monte Carlo simulations.
 
## Results
> The robust optimal strategy consistently yields a higher Sharpe ratio than the plain optimal strategy, demonstrating improved performance under model uncertainty.