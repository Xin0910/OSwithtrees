# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 12:34:30 2023

@author: xin
"""

#%%

import time
import pandas as pd 
import numpy as np
import torch
import torch.nn as nn
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)


#%%

class BM:
    def __init__(self, T, r, N, K, d, initial = 0,
                 payofftype ='identity'):
        self.T = T                   #terminal time
        self.initial = initial          #initial price
        self.N = N                     #number of exercise oppotunities
        self.K = K                  #number of paths
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


# Creates neural network 
def create_model():
    """
    Creates a neural network with 2 hidden layers of 40+d units
    Includes batch norm layers
    """
    model = nn.Sequential(
    nn.Linear(d+1, d+40),
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
        np.random.seed( j + 10000)
        batch  = S.simulatepaths()
        batch_now = batch[:, i, :]
        batch_gvalues = S.dis_PAYOFF(batch)
        batch_gvalues_now = batch_gvalues[:, i].reshape(1, batch_size)
        batch = torch.from_numpy(batch).float().to(device)
        X = np.hstack((batch_now,  batch_gvalues[:, i].reshape(batch_size,1)))
        temp = np.dstack((batch, batch_gvalues))
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
r = 1
d = 1
N =int( 100 * T)

learning_rate = 0.001

batch_size = 8192
trainingstep = 3000
testsize = 4096
roundtime = 1000


d_s = [1] 
initials = [-1, -0.1 ,-0.2, -0.3,-0.4, -0.5,0, 1, 0.1, 0.2, 0.3, 0.4 , 0.5]
batch_size = 4096
trainingstep = 50
testsize = 100000
roundtime = 1

initial = 0
columnname =['initial_param', 'value','t_train','t_test']

value=pd.DataFrame(columns=columnname)
value['initial_param'] = initials

#%%

print('start training!')

for initial in initials:
    print('initial =',initial)

    # # initiates dictionaries that will contain functions F (soft stopping decision),
    # # f (stopping decision) and l (stopping time) from the paper
    # def fN(x):
    #     return 1
    # def FN(x):
    #     return 1.0
    # def lN(x):    #can take input a vector of values
    #     """
    #     Argument:
    #     x: a tensor of shape (k,d,1) which contains Nth values of brownian paths for k samples
    #     Outputs:
    #     Stopping times as a tensor of shape (k, ). (in this case it will just output [N-1, N-1, ..., N-1])
    #     """
    #     ans = N  * np.ones(shape = (x.shape[0], ))
    #     ans = ans.astype(int)
    #     return ans
        
    total_paths = batch_size * (trainingstep + d)
    number_of_training_steps = int(trainingstep + d)        
    f = {N : fN}   #dictionary containing little f functions from the paper  (Decision functions to stop)
    F = {N : FN}   #dictionary containing big F functions  (Soft decision functions i.e models)
    l= {N : lN}  #dictionary containing little l functions (Stopping times) 
    S = BM(T, r, N, batch_size, d, initial= initial)
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
        
    np.random.seed(42)
    S = BM(T, r, N, testsize, d, initial= initial)
    
    price = 0
    time_test = 0

    for i in range(roundtime):
        X = S.simulatepaths()
        g_val = S.dis_PAYOFF(X)
        temp = np.dstack((X, g_val))
        X = torch.from_numpy(temp).float().to(device)
        start_test = time.time()
        stopping = evaluate(F, 1, N,X)
        finish_test = time.time()
        Z = g_val[range(testsize), stopping]
        price += np.mean(Z)
        time_test += finish_test - start_test
   
    print(price/roundtime)

    value.loc[(value.initial_param==initial),'value']= price/roundtime
    value.loc[(value.initial_param==initial),'t_train']= finish_train- start_train
    
    value.loc[(value.initial_param==initial),'t_test']= time_test
        
print(value)

