# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 11:49:38 2022

@author: xin
"""
import math
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

#Longstaff method


def generatebasis(S, degree =2):
    '''
    return a polynomial basis array of S
    '''
    poly = PolynomialFeatures(degree = degree )
    X = poly.fit_transform(S)
    return X


def LSM(S, S_test, N, T, r,K, K_test,h, h2, features = False, paths=None, 
        paths_test=None, in_money= True):

    """
    Longstaff Schwartz algorithm
    
    Parameters
    ----------
    S : numpy array
        training data.
    S_test : numpy array
        testing data.
    N : int
        number of exercise opportunities.
    T : int
        maturity.
    r : float
        interest rate.
    k : int
        sample size.
    h : numpy array
        payoff of training data.
    h2 : numpy array
        payoff of testing data.

    features :  bool
        if we use paths and paths_test as features. The default is False.
    paths : numpy array
        features input of training data. The default is None.
    paths_test: numpy array
        features input of testing data. The default is None.
    in_money : bool
        if we use only in the money paths for training. The default is True.


    Returns
    -------
    value using training data.
    value using testing data.
    model

    """
    
    value_LS  = h[:,-1]
    value2_LS = h2[:,-1]
    model =[None] *(N+1)
    coef = [None] * (N+ 1)
    for t in reversed(range(1,N)):

        if in_money == True:
            idx = h[:,t] > 0
            if idx.sum() > 0:
                
                C = np.zeros(K)
                if features == False:        
                    basisf =  S[:,t,:] 
                else: 
                    basisf = paths[:,t,:]            
                
                X_  = generatebasis(basisf, 2)
                y = value_LS * math.exp(-r * T/N)
                clf = LinearRegression( fit_intercept = False )
                clf.fit(X_[idx], y[idx])
                coef[t] = clf.coef_
                C[idx] =  clf.predict(X_[idx])
                model[t] =clf
                value_LS = np.where(h[:,t] > C, h[:,t], value_LS * math.exp(-r * T/N))
                
                
                C_test = np.zeros(K_test)
                
                
                if features == False: 
                    basisf2 =  S_test[:,t,:]
                else: 
                    basisf2 = paths_test[:,t,:]            
                X2 = generatebasis(basisf2, 2)
                idx2 = h2[:,t] > 0
                if h2.sum()>0:
                    C_test[idx2]  =  clf.predict(X2[idx2] )
                else:
                    C_test  =  clf.predict(X2 )
                value2_LS = np.where(h2[:,t] > C_test, h2[:,t], value2_LS * math.exp(-r * T/N))
      
        
            else:
                value_LS = np.where(h[:,t] >value_LS * math.exp(-r * T/N) , h[:,t], value_LS * math.exp(-r * T/N))
                value2_LS = np.where(h2[:,t] > value2_LS * math.exp(-r * T/N), h2[:,t], value2_LS * math.exp(-r * T/N))
        else:

            C = np.zeros(K)
            if features == False:        
                basisf =  S[:,t,:] 
            else: 
                basisf = paths[:,t,:]            
            
            X_  = generatebasis(basisf, 2)
            y = value_LS * math.exp(-r * T/N)
            clf = LinearRegression( fit_intercept = False )
            clf.fit(X_, y)
            coef[t] = clf.coef_
            C=  clf.predict(X_)
            model[t] =clf
            value_LS = np.where(h[:,t] > C, h[:,t], value_LS * math.exp(-r * T/N))
            
            
            C_test = np.zeros(K_test)
            
            
            if features == False: 
                basisf2 =  S_test[:,t,:]
            else: 
                basisf2 = paths_test[:,t,:]            
            X2 = generatebasis(basisf2, 2)
            
            C_test  =  clf.predict(X2 )
            value2_LS = np.where(h2[:,t] > C_test, h2[:,t], value2_LS * math.exp(-r * T/N))
           
                
        # print(value2_LS.mean())

    value_LS  = np.mean(math.exp(-r * T/N) * value_LS)
    value2_LS = np.mean(math.exp(-r * T/N) * value2_LS)    
        
    
    return value_LS, value2_LS, model

