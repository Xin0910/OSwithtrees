# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 10:34:10 2023

@author: xin
"""

import numpy as np
from fbm import FBM
from Deltaalgo import  trainingDelta, testDelta
import time
import pandas as pd

#%%
class FractualBM:
    
    def __init__(self, T,   N, K, d, hurst, initial = 0,  payofftype ='identity'):

        self.T = T                   #terminal time
        self.initial = initial          #initial price
        self.N = N                     #number of exercise oppotunities
        self.K = K                    #number of paths
        self.d = d                     #number of assets
        self.dt=self.T/self.N
        self.hurst = hurst
        self.payofftype = payofftype
        if self.hurst != 1:
            self.fBM  = FBM(n=self.N, hurst=self.hurst, length=self.T, method='cholesky')
    
    def fractualBM1(self):
        """fractional Brownian Motion when hurst paramter H=1"""
        return np.linspace(0, self.T, self.N+1) * np.random.randn(1)
    def simulatepaths(self):
        paths = np.zeros((self.K, self.N + 1, self.d),dtype= np.float32)
        if self.hurst != 1:
            for m in range(self.K):
                for i in range(self.d):
                    paths[m,:,i] = self.fBM.fbm()  + self.initial
        else:
            paths[:,0,:] = self.initial
            paths[:,1:,:] = np.random.normal(scale  =np.sqrt( self.T/ self.N), size = (self.K, self.N, self.d))
            paths =  self.initial + np.cumsum(paths, axis = 1)
        return paths


    def dis_PAYOFF(self, X):
        X= X[:,:,0]
        if self.payofftype == 'identity':
            return X
        elif self.payofftype == 'sin':
            return np.sin(X)
        elif self.payofftype == 'square':
            return X ** 2
        else:
            print('Choose the correct payoff type')



#%% 
#parameters
T = 1
K = 20000
K_test = 20000
d = 1
min_node_size = 10
numFolds= 10
N =int( 100 * T)

eps = 0
depth = 10
initial = 0
window_size = N
kwargs = {'min_node_size':min_node_size, 'depth':depth ,'eps' : eps}
features = 'S'
pathDep = True

hursts =  [i/100 for i in range(5,105,5)]
hursts.insert(0,0.01)

hursts = [0.05]

foldsize= int(K / numFolds)
trained_models= []


columnname =['hurst_param','t_train','value_train','t_test','value_test']

value=pd.DataFrame(columns=columnname)
value['hurst_param'] = hursts

#%%

for hurst in hursts:
    print('hurst:', hurst)
    # training
    np.random.seed(42)
    S = FractualBM(T,  N, K, d, hurst,initial, payofftype ='identity')     
    paths = S.simulatepaths()
    all_payoff = S.dis_PAYOFF(paths)
    
    start_train = time.time()
    time_mat_all, V_est , estimators = trainingDelta(paths, N, K, numFolds,
                                                     all_payoff, 
                                                     features = features,
                                                     pathDep = pathDep,
                                                     window_size = window_size,
                                                     **kwargs)
    finish_train = time.time()
    print('test value  %.4f.' %V_est )
    
    print('train time %.4f' %(finish_train - start_train))
    
    trained_models.append(estimators)
    #testing
    S = FractualBM(T,  N, K_test, d, hurst,initial, payofftype ='identity')     
    paths_test=S.simulatepaths().astype(np.float32)
    all_payoff_test = S.dis_PAYOFF(paths_test)
    
    start_test = time.time()
    
    
    time_mat_test, value_test = testDelta(paths_test, N, K_test, estimators,
                                          all_payoff_test, features = features, 
                                          pathDep = pathDep,
                                          window_size = window_size)
    
    
    finish_test = time.time()
    print('test time %.4f' %(finish_test - start_test))
    print('test value  %.4f.' % value_test )
    value.loc[(value.hurst_param==hurst),'t_train'] = finish_train - start_train
    value.loc[(value.hurst_param==hurst),'value_train'] = V_est
    value.loc[(value.hurst_param==hurst),'t_test']=finish_test - start_test
    value.loc[(value.hurst_param==hurst),'value_test']=value_test