# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 16:06:56 2021

@author: 41546

implementation of the paper 'deep optimal stopping'
citation: https://github.com/erraydin/Deep-Optimal-Stopping
"""

#%%

import time
import pandas as pd 
from fbm import FBM

import numpy as np
import torch
import torch.nn as nn
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



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




# Creates neural network 
def create_model():
    """
    Creates a neural network with 2 hidden layers of 40+d units
    Includes batch norm layers
    """
    model = nn.Sequential(
    nn.Linear(d, d+40),
    nn.BatchNorm1d(40+d),
    nn.ReLU(),
    nn.Linear(d+40, d+40),
    nn.BatchNorm1d(d+40),
    nn.ReLU(),
    nn.Linear(d+40, 1),
    nn.Sigmoid()
    )
    return model

# initiates dictionaries that will contain functions F (soft stopping decision),
# f (stopping decision) and l (stopping time) from the paper
def fN(x):
    return 1
def FN(x):
    return 1.0
def lN(x):    #can take input a vector of values
    """
    Argument:
    x: a tensor of shape (k,d,1) which contains Nth values of brownian paths for k samples
    Outputs:
    Stopping times as a tensor of shape (k, ). (in this case it will just output [N-1, N-1, ..., N-1])
    """
    ans = N  * np.ones(shape = (x.shape[0], ))
    ans = ans.astype(int)
    return ans



def train(model, i, optimizer,l,S,number_of_training_steps):
    for j in range(number_of_training_steps):
        # print(j)
        np.random.seed( j + 10000)
        batch  = S.simulatepaths()
        # batch_now = batch[:, i, :]
        batch_gvalues = S.dis_PAYOFF(batch)
        batch_gvalues_now = batch_gvalues[:, i].reshape(1, batch_size)
        batch = torch.from_numpy(batch).float().to(device)
        # X = np.hstack((batch_now,  batch_gvalues[:, i].reshape(batch_size,1)))
        X =  batch_gvalues[:, i].reshape(-1,1)
        # temp = np.dstack((batch, batch_gvalues))
        temp = np.expand_dims(batch_gvalues, 2)
        temp = torch.from_numpy(temp).float().to(device)
        
        Z = batch_gvalues[range(batch_size), l[i+1](temp)].reshape(1, batch_size)
        
        X = torch.from_numpy(X).float().to(device)
        batch_gvalues_now = torch.from_numpy(batch_gvalues_now).float().to(device)
        Z = torch.from_numpy(Z).float().to(device)
      
        #compute loss
        z = model(X)
        ans1 = torch.mm(batch_gvalues_now, z)
        ans2 = torch.mm( Z, 1.0 - z)
        loss = - 1 / batch_size * (ans1 + ans2)
        
        #apply updates
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # print("one step done")


#%%

T = 1
d = 1
N =int( 100 * T)

eps = 0
depth = 10
initial = 0

hursts =  [i/100 for i in range(5,105,5)]
hursts.insert(0,0.01)

hursts = [0.05]
learning_rate = 0.001

batch_size = 2048
trainingstep = 6000
testsize = 4096
roundtime = 1000



columnname =['hurst_param','t_train','value_train','t_test','value_test']

value=pd.DataFrame(columns=columnname)
value['hurst_param'] = hursts

#%%

for hurst in hursts:
    print('hurst:', hurst)
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


    total_paths = batch_size * (trainingstep + d)
    number_of_training_steps = int(trainingstep + d)        
    f = {N : fN}   #dictionary containing little f functions from the paper  (Decision functions to stop)
    F = {N : FN}   #dictionary containing big F functions  (Soft decision functions i.e models)
    l= {N : lN}  #dictionary containing little l functions (Stopping times) 
    
    S = FractualBM(T,  N, batch_size, d, hurst,initial, payofftype ='identity')    
    start_train = time.time()
    for i in range(N-1, 0, -1):   
        
        model = create_model().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    
        train(model, i, optimizer,l,S,number_of_training_steps)
        F.update({i  : model})
        
        def fi(x, i=i):
            func = F[i].eval()
            ans = torch.ceil(func(x) - 1/2)
            return ans
        
        f.update({i : fi})
        
        def li(x, i=i):
            a= f[i](x[:,i,:]).cpu().detach().numpy().reshape(x[:,i,:].size()[0], )
            ans = (i)*a + np.multiply(l[i+1](x), (1-a))
            ans = ans.astype("int32")
            return ans
        
        l.update({i : li})
        
    finish_train= time.time()
    print('training time:',finish_train - start_train)


    def evaluate(F, i ,N, test_X):
        # test_X: shape (M ,N+1,  d + 1) , stockprices + current payoff
        
        if i == N:
            return np.ones(test_X.shape[0]) * N 
        func = F[i].eval()
        a = torch.ceil(func(test_X[:,i,:])- 1/2).cpu().detach().numpy().reshape(test_X[:,i,:].size()[0],)
        

        ans = (i)*a + np.multiply(evaluate(F, i +1,N,test_X), (1-a))
        ans = ans.astype("int32")
   
        return ans
        
    S = FractualBM(T,  N, testsize, d, hurst,initial, payofftype ='identity')    
    
    price = 0
    time_test = 0

    for i in range(roundtime):
        X = S.simulatepaths()
        g_val = S.dis_PAYOFF(X)
        temp = np.expand_dims(g_val, 2)
        X = torch.from_numpy(temp).float().to(device)
        start_test = time.time()
        stopping = evaluate(F, 1, N,X)
        finish_test = time.time()
        Z = g_val[range(testsize), stopping]
        price += np.mean(Z)
        time_test += finish_test - start_test
   
    print('test value:', price/roundtime)
    value.loc[(value.hurst_param==hurst),'t_train'] = finish_train - start_train
    value.loc[(value.hurst_param==hurst),'t_test'] =finish_test - start_test
    value.loc[(value.hurst_param==hurst),'value_test'] = price/roundtime
        
        
print(value)

