# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 14:00:33 2023

@author: xin
"""
from scipy.stats import norm 
from scipy import optimize

import numpy as np
def boundaryfunc(x, boundary, C, sigma, r,  k, tau_s , delta_tau , q=0 ):
    '''
    

    Parameters
    ----------
    x : numpy array
        stock prices.
    boundary : numpy array
        boundary from previous step.
    K : int
        strike price.
    sigma : float
        volatility.
    r : float
        interest rate.
    k : int
        time step.
    tau_s : numpy array
        list of time steps.
    delta_tau : float
        period between two adjacent time steps.
    q : float, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    numpy array
        DESCRIPTION.

    '''

    d1 = (np.log(x/C) + (r-q + sigma**2/2) * tau_s[k] ) / (sigma *np.sqrt(tau_s[k]))
    d2 = d1 - sigma * np.sqrt(tau_s[k])
    p =C* np.exp(- r * tau_s[k]) * norm.cdf( - d2) - \
        x * np.exp( - q * tau_s[k]) * norm.cdf( - d1)
    if k == 1:

        f1 = 0
        eta = tau_s[1]
        d1_hat = (np.log(x/boundary[0])+(r - q + sigma**2/2) * eta)/(sigma * np.sqrt(eta))
        d2_hat = d1_hat - sigma * np.sqrt(eta)
        f2 = r * C * np.exp(-r * eta) * norm.cdf(-d2_hat) 
        f_integral = delta_tau/2 * (f1 + f2)
    else:

        f1 = 0
        eta = tau_s[k]
        d1_hat = (np.log(x/boundary[0])+(r - q + sigma**2/2) * eta)/(sigma * np.sqrt(eta))
        d2_hat = d1_hat - sigma * np.sqrt(eta)
        f2 = r *C * np.exp(-r * eta) * norm.cdf(-d2_hat)
        f_integral = delta_tau/2 * (f1 + f2)

        for i in range(1, k):
            eta = tau_s[i]
            d1_hat = (np.log(x/boundary[k-i])+(r - q + sigma**2/2) * eta) / (sigma * np.sqrt(eta))
            d2_hat = d1_hat - sigma * np.sqrt(eta)
            f_temp = r *C * np.exp(-r * eta) * norm.cdf(-d2_hat) 
            f_integral += delta_tau * f_temp
    return C - x- p - f_integral

def boundarycurve(C, sigma, r ,tau_s, T,N):
    '''
    optimize the theoretical boundary of American put options
    '''
    optimized =  np.zeros(N+1).astype(object)
    boundary = np.zeros(N+1).astype(np.float32)
    boundary[0] =C
    for k in range(1,N+1):
        optimized[k]= optimize.root(boundaryfunc, float(boundary[k-1]) , 
                                    args = (boundary,C, sigma, r, k ,tau_s, T/N,))
        boundary[k]  = float(optimized[k].x)    
    return np.flip(boundary)
