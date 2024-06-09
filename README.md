# OSwithtrees

This repository is the official implementation of the PhD thesis by Xin Zhi.

# List of the algorithms available

1. The file Deltaalgo.py is the official implementation of the $\Delta$-algorithm in the paper [Optimal Stopping with Trees: The Details](https://arxiv.org/pdf/2210.04645.pdf). The manual is stated in the appendix of the thesis
2. The file boundary.py is the implementation of approximation method of the optimal stopping boundary of American put options described in [Yue-Kuen Kwok (2008)](https://link.springer.com/book/10.1007/978-3-540-68688-0)
3. The file LSM.py is the implementation of the Longstaff-Schwartz (LS) method for valuing American options in [Longstaff and Schwartz (2001)](https://people.math.ethz.ch/~hjfurrer/teaching/LongstaffSchwartzAmericanOptionsLeastSquareMonteCarlo.pdf)
4.  The file GPR_PCA_pockets.py is the implementation of the algorithm 'Gaussian Process Regression with PCA' in the thesis
5.  The file GPR.py is the implementation of the algorithm 'Gaussian Process Regression method' in the thesis

# Numerical applications

1. Americanputs.py : Section 3.1 American put options using the LS method and the $\Delta$-algorithm
2. Bermudanmaxcall.py : Section 3.2 High-Dimensional Bermudan Max-Call Options using the $\Delta$-algorithm
3. Brownianmotion_Delta.py : Section 4.3 Brownian Motion using the $\Delta$-algorithm
4. DOS_Delta.py : Section 3.2 High-Dimensional Bermudan Max-Call Options using the Neural network algorithm in [Becker et al (2019)](https://www.jmlr.org/papers/volume20/18-232/18-232.pdf)
5. DampedBM_DOS.py : Section 4.3 Damped Brownian Motion using the Neural network algorithm in [Becker et al (2019)](https://www.jmlr.org/papers/volume20/18-232/18-232.pdf)
6. FBM_Delta.py : Section 4.3 Fractional Brownian Motion using the $\Delta$-algorithm
7. FBM_DOS.py : Section 4.3 Fractional Brownian Motion using the Neural network algorithm in [Becker et al (2019)](https://www.jmlr.org/papers/volume20/18-232/18-232.pdf)
8. InterpretableOS_multiprocessing.py : Section 3.3 Pricing Bermudan Max-Call Options with Barrier using the algorithm in [Ciocan and Misic (2019)](https://arxiv.org/pdf/1812.07211.pdf)
9. Interpretable_Delta.py : Section 3.3 Pricing Bermudan Max-Call Options with Barrier using the $\Delta$-algorithm
10. SDDE.py : Chapter 5 and 6 Stochastic differential equation with delay using the LS method and the $\Delta$-algorithm
11. alphastable.py Section 4.1.5 $\alpha$-Stable Processes using the LS method and the $\Delta$-algorithm
12. levy.py : Section 4.1.6 Compound Poisson Process using the LS method and the $\Delta$-algorithm
