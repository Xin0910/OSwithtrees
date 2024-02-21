# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 15:08:55 2023

@author: xin
"""
import time
from Deltaalgo import trainingDelta, testDelta
import numpy as np
import pandas as pd

class BM:
    def __init__(self, T, r, N, K, d, initial = 0,
                 payofftype ='identity'):
        self.T = T                   #terminal time
        self.initial = initial          #initial price
        self.N = N                     #number of exercise oppotunities
        self.K = K                   #number of paths
        self.d = d                     #number of assets
        self.dt=self.T/self.N
        self.r = r
        self.payofftype = payofftype

    def simulatepaths(self):
        paths = np.zeros((self.K, self.N + 1, self.d),dtype= np.float32)
        paths[:,0,:] = self.initial
        paths[:,1:,:] = np.random.normal(loc = 0,
                                         scale = np.sqrt( self.T/ self.N),
                                         size = (self.K, self.N, self.d))
        paths = np.cumsum(paths, axis = 1)
        return paths

    def dis_PAYOFF(self, X, discount = True):
        X= X[:,:,0]
        if self.payofftype == 'identity':
            payoff = X
        elif self.payofftype == 'sin':
            payoff = np.sin(X)
        elif self.payofftype == 'square':
            payoff = X ** 2
        else:
            print('Choose the correct payoff type')
        
        if discount == False:
            return payoff
        else:
            timematrix=np.ones((self.K,self.N+1))*np.arange(0,self.N+1,1)
            
            return np.exp(-self.r*(self.dt)*timematrix)* payoff
            # return np.exp(-self.r*(self.T))* payoff

#%%
# Brownina motion example
'''
min_node_size = 10
numFolds= 10
eps = 0
depth = 0
kwargs = {'min_node_size':min_node_size, 'depth':depth ,'eps' : eps}
T = 1
K = 20000
K_test = 20000
r = 0
d = 1
N =int( 100 * T)
features = 'S'


np.random.seed(42)
S = BM(T, r, N, K, d, initial= 0)
paths = S.simulatepaths()
all_payoff = S.dis_PAYOFF(paths)


start=time.time()
time_mat_all , V_est ,estimators = trainingDelta(paths, N, K, numFolds,
                                                 all_payoff, 
                                                 features = features,
                                                 **kwargs)
print(V_est)

S = BM(T, r, N, K_test, d,initial= 0)
paths_test = S.simulatepaths()
all_payoff_test = S.dis_PAYOFF(paths_test)

time_mat_test, value = testDelta(paths_test, N, K_test, estimators, 
                                 all_payoff_test, 
                                 features = features)
print(value)
'''

#%%
# damped Brownian Motion, i.e., adding a discount factor

min_node_size = 10
numFolds= 10
eps = 0
depth = 10
kwargs = {'min_node_size':min_node_size, 'depth':depth ,'eps' : eps}
T = 1
K = 20000
K_test = 20000
r = 1
d = 1
N =int( 100 * T)
features = 'S'
# initial = 0

foldsize= int(K / numFolds)

initials = [-1, -0.1 ,-0.2, -0.3,-0.4, -0.5,0, 1, 0.1, 0.2, 0.3, 0.4 , 0.5]


columnname =['initial_param','value','Literature','t_test']
value=pd.DataFrame(columns=columnname)
value['initial_param'] = initials

#%%

for initial in initials:    
    np.random.seed(42)
    S = BM(T, r, N, K, d, initial= initial)
    paths = S.simulatepaths()
    all_payoff = S.dis_PAYOFF(paths)
    
    
    start=time.time()
    time_mat_all , V_est ,estimators = trainingDelta(paths, N, K, numFolds,
                                                     all_payoff, 
                                                     features = features,
                                                     **kwargs)
    # print(V_est)
    
    S = BM(T, r, N, K_test, d,initial= initial)
    
    paths_test = S.simulatepaths()
    all_payoff_test = S.dis_PAYOFF(paths_test)
    
    time_mat_test, value_test = testDelta(paths_test, N, K_test, estimators, 
                                     all_payoff_test, 
                                     features = features)
    print(value_test)
    value.loc[(value.initial_param==initial),'value']= value_test