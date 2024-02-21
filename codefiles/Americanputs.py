# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 18:48:13 2021

@author: xin
This script is to create early exercise boundary by solving the volterra integral
and to create numerical results in the case of American put options.
"""
#%%
from Deltaalgo import trainingDelta, testDelta
import numpy as np
from LSM import LSM
from boundary import boundarycurve
from collections import Counter
import time
import matplotlib.pyplot as plt
np.random.seed(42)

#%%
class stock:
    def __init__(self, T, C, sigma, So, r, N, K, d, delta=0, symmetric = True):
        """
        

        Parameters
        ----------
        T : int/float
            terminal time.
        C : int/float
            strike price.
        sigma : float
            initial volatility.
        So : int/float
            initial price.
        r : float
            riskless interest rate   .
        N : int
            number of exercise oppotunities.
        K : int
            number of paths.
        d : int
            number of stocks.
        delta : float, optional
            DESCRIPTION. The default is 0.
        symmetric : bool, optional
            set the symmetric feature of the volatility of stocks. 
            The default is True.
        """
        
        self.T = T                   #terminal time
        self.C= C                     #strike price         
        self.delta=delta             #dividend
        self.So=So                   #initial price
        self.r=r                     #riskless interest rate   
        self.N=N                     #number of exercise oppotunities
        self.K=K                     #number of paths
        self.d=d                     #number of assets
        self.symmetric = symmetric
        self.sigma=self.cal_sigma(sigma)
        self.dt=self.T/self.N
    
    def cal_sigma(self, sigma):
        """
        

        Parameters
        ----------
        sigma : float
            initial volatility.

        Returns
        -------
        array
            volatility.

        """
        
        if self.symmetric == True:
            return sigma
        else:
            if self.d > 5:
                return 0.1+np.arange(1,self.d+1)/(2*self.d)
            else:
                return 0.08+0.32*(np.arange(1,self.d+1)-1)/(self.d-1)
    def simulatepaths(self):
        """
        

        Returns
        -------
        s : numpy array
            stock price matrix with size (N,M,d).

        """
        S = np.zeros((self.K, self.N + 1, self.d),dtype= np.float32)
        S[:,0,:] = self.So        
        Z=np.random.standard_normal((self.K, self.N , self.d)) .astype(np.float32)

        S[:,1:,]=self.So*np.exp(np.cumsum((self.r-self.delta
                                           -0.5*self.sigma**2)*self.dt
                                          +self.sigma*np.sqrt(self.dt)*Z, axis=1))
        
        return S



def dis_payoff(S, C, r, K , T, N):
    '''
    

    Parameters
    ----------
    S : TYPE
        stock price.
    C : TYPE
        strike price.
    r : TYPE
        interest rate.
    K : TYPE
        sample size.
    T : TYPE
        maturity.
    N : TYPE
        number of exercise opportunities.

    Returns
    -------
    TYPE
        discounted payoff.

    '''
    payoff = np.maximum(C - S[:,:,0], 0) 
    timematrix = np.ones((K, N+1))*np.arange(0,N+1,1)
    return  np.exp(-r*(T/N)*timematrix)*payoff.astype(np.float32)


def boundarytest(S, C , T, N , r, K, boundary):
    boundary_matrix = np.repeat(boundary.reshape(1,-1), K, axis = 0)
    value_true = S[:,:,0] <= boundary_matrix
    value_true[:,-1] = True
    stoppingtime = np.argmax(value_true,axis = 1)
    value_fitted = np.exp(-r * stoppingtime *T/N) * np.maximum(C- S[np.arange(K), stoppingtime,0], 0)
    return value_fitted.mean()

def averageSboundary(C, tau, S, N):
    S_t_mean = np.zeros(N+1)
    S_t_mean[-1] = C
    for t in reversed(range(1,N)):
        idx = np.where(tau == t)[0]
        col_t_idx = tau[idx]
        if len(idx) == 0:
            S_t_mean[t] = S_t_mean[t+1]
        else:
            S_t_mean[t] = S[idx, col_t_idx, 0].mean()
    S_t_mean[0] = S_t_mean[1]
    
    return S_t_mean


#%%

#parameter setting
#T, K, sigma, So, r, N, M, d, delta
min_node_size = 10
rho = 1
d= 1
numFolds= 10
S0_hat = 85
sigma = 0.2
r =  0.05
T = 1
lambda_ = 6
rho = 1
C = 100
K  = 200000
K_test = 200000
delta = 0
N =int( 50 * T)
foldsize=int(K/numFolds)
eps = 0
depth = 10
q = 0
features = 'S'
# print theoretical boundary

tau_s = np.linspace(0,T,N+1)

boundary_flip = boundarycurve(C, sigma, r,tau_s, T,N)

#%%
kwargs = {'min_node_size':min_node_size, 'depth':depth ,'eps' : eps}
# training the Delta algorithm
np.random.seed(42)

X = stock(T, C, sigma, S0_hat , r, N, K, d, delta)   
S = X.simulatepaths().astype(np.float32)#[:,:,0] #because d = 1
all_payoff = dis_payoff(S, C, r, K , T, N)

np.random.seed(100)
X = stock(T, C, sigma, S0_hat , r, N, K_test, d, delta)
S_test = X.simulatepaths().astype(np.float32)
all_payoff_test = dis_payoff(S_test, C, r, K_test , T, N) 
start=time.time()
time_mat_all , V_est ,estimators = trainingDelta(S, N, K, numFolds,
                                                 all_payoff, 
                                                 features = features,
                                                 **kwargs)
finish=time.time()
print('The training value calculated using our method is %.3f.' %V_est )
# print('computation time %.3f' %(finish-start))


# out of sample test the Delta algorithm
time_mat_test, value = testDelta(S_test, N, K_test, estimators, 
                                 all_payoff_test, 
                                 features = features)
print('The testing value calculated using our method is %.3f.' %value)


#%%

tau = time_mat_test[:,0]
idx_train = np.where((tau!=N) & (tau!=0))[0]
col_idx = tau[idx_train]



value_theory= boundarytest(S_test, C , T, N , r, K_test, boundary_flip)

print('The value calculated using the theoretical boundary is %.3f.' %value_theory )

S_t_mean = averageSboundary(C, time_mat_all[:,0], S, N)
value_mean= boundarytest(S_test, C , T, N , r, K_test, S_t_mean)

print('The value calculated using the mean value boundary is %.3f.' %value_mean)
#%%

#Longstaff Schwartz method

h  = np.maximum(C - S, 0)[:,:,0]
h2 = np.maximum(C - S_test, 0)[:,:,0]
value_LS  ,value2_LS,model_LS =LSM(S, S_test, N, T, r,K, K_test,h, h2)
print('The training value calculated using the LS method is %.3f.' %value_LS)
print('The testing value calculated using the LS method is %.3f.' %value2_LS)



#%%

#create figure 7-12 in the paper

# first create S_t_mean using test sample
tau = time_mat_test[:,0]
idx_train = np.where((tau!=N) & (tau!=0))[0]
col_idx = tau[idx_train]


#figure 7a
boundary_matrix = np.repeat(boundary_flip.reshape(1,-1),K_test, axis = 0)
value_true = S_test[:,:,0] <= boundary_matrix
value_true[:,-1] = True
stoppingtime = np.argmax(value_true,axis = 1)
idx_train2 = np.where((stoppingtime!=N) & (stoppingtime!=0))[0]
col_idx2 = stoppingtime[idx_train2]

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(np.arange(N+1),boundary_flip , label = 'theoretical boundary $b(n)$')

ax.scatter(col_idx2, S_test[idx_train2, col_idx2,0], s = 5, c='black',label='stopping at '+'$b(n)$')
ax.set_xlim(0, N)
ax.set_ylim(55, 100)
ax.legend()
ax.set_xlabel('$n$')
ax.set_ylabel(r'$\tilde{X}_{n}$',rotation=0)


#figure 8a


boundary_matrix = np.repeat(boundary_flip.reshape(1,-1),K_test, axis = 0)

xx = S_test[idx_train2, col_idx2,0] -  boundary_matrix[idx_train2, col_idx2]
fig = plt.figure()
ax = plt.subplot()
ax.hist(xx,  density = True, bins = 200,label = 'residuals')
ax.grid()
ax.set_xlim(-12,4)
ax.set_ylim(0, 0.6)



# figure 9a and 10a



col_idx2 = stoppingtime[idx_train2]
count = Counter(col_idx2)
count_n = np.array(list(count.keys()))
count_val = np.array(list(count.values()))
count_idx = np.argsort(count_n)

fig = plt.figure()
ax = plt.subplot()
ax.scatter(col_idx2, S_test[idx_train2, col_idx2,0], s = 5, c='black',label='stopping at '+'$b(n)$')
ax.set_xlim(0, N)
ax.set_ylim(55, 100)
ax.set_xlabel('$n$')
ax.set_ylabel(r'$\tilde{X}_{n}$',rotation=0)
ax.plot(np.arange(N+1),boundary_flip , label = 'theoretical boundary $b(n)$')
# ax.plot(np.arange(N+1),S_t_mean , label =  'averaged boundary '+r'$\bar{b}(n)$')

ax2 = ax.twinx()
ax2.set_ylim(0,20000)
ax2.set_ylabel('counts')
ax2.plot(count_n[count_idx], count_val[count_idx],label = 'number of stoppings',color ='green')

# ax.legend(loc='upper center')
# ax2.legend()

h1, l1 = ax.get_legend_handles_labels()
h2 ,l2= ax2.get_legend_handles_labels()
ax.legend(h1+h2, l1+l2 ,  loc='upper center')



#figure 7b

tau = time_mat_test[:,0]
idx_train = np.where((tau!=N) & (tau!=0))[0]
col_idx = tau[idx_train]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.arange(N+1),boundary_flip , label = 'theoretical boundary $b(n)$')
ax.scatter(col_idx, S_test[idx_train, col_idx, 0], s = 5, c='black',label=r'$\Delta$'+'-algorithm')
ax.set_xlim(0, N)
ax.set_ylim(55, 100)
ax.legend()
ax.set_xlabel('$n$')
ax.set_ylabel(r'$\tilde{X}_{n}$',rotation=0)






#figure 8b
tau = time_mat_test[:,0]
idx_train = np.where((tau!=N) & (tau!=0))[0]
col_idx = tau[idx_train]
xx = S_test[idx_train, col_idx, 0] -  boundary_matrix[idx_train, col_idx]
fig = plt.figure()
ax = plt.subplot()
ax.hist(xx,  density = True, bins = 200,label = 'residuals')
ax.grid()
ax.set_xlim(-12,4)
ax.set_ylim(0, 0.6)


#figure 9b and 10b


tau = time_mat_test[:,0]
idx_train = np.where((tau!=N) & (tau!=0))[0]
col_idx = tau[idx_train]

count = Counter(col_idx)
count_n = np.array(list(count.keys()))
count_val = np.array(list(count.values()))
count_idx = np.argsort(count_n)
#The following is to plot 3 boundaries in one graph.
fig = plt.figure()
ax = plt.subplot()
ax.scatter(col_idx, S_test[idx_train, col_idx, 0], s = 5, c='black', label=r'$\Delta$'+'-algorithm')
ax.set_xlim(0, N)
ax.set_ylim(55, 100)
ax.set_xlabel('$n$')
ax.set_ylabel(r'$\tilde{X}_{n}$',rotation=0)
ax.plot(np.arange(N+1),boundary_flip , label = 'theoretical boundary $b(n)$')
ax.plot(np.arange(N+1),S_t_mean , label =  'averaged boundary '+r'$\bar{b}(n)$')

ax2 = ax.twinx()
ax2.set_ylim(0,20000)
ax2.set_ylabel('counts')
ax2.plot(count_n[count_idx], count_val[count_idx],label = 'number of stoppings',color ='green')

# ax.legend(loc='upper center')
# ax2.legend()

h1, l1 = ax.get_legend_handles_labels()
h2 ,l2= ax2.get_legend_handles_labels()
ax.legend(h1+h2, l1+l2 ,  loc='upper center',fontsize = 'small')


#figure 11
# first create S_t_mean using test sample
tau = time_mat_test[:,0]
idx_train = np.where((tau!=N) & (tau!=0))[0]
col_idx = tau[idx_train]
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(col_idx, S_test[idx_train, col_idx, 0], s = 5, c='black',label=r'$\Delta$'+'-algorithm')
ax.set_xlim(0, N)
ax.set_ylim(55, 100)
ax.set_xlabel('$n$')
ax.set_ylabel(r'$\tilde{X}_{n}$',rotation=0)
ax.plot(np.arange(N+1),boundary_flip , label = 'theoretical boundary $b(n)$')
ax.plot(np.arange(N+1),S_t_mean , label = 'averaged boundary '+r'$\bar{b}(n)$')
ax.legend()


#figure 11 in black&white

cm =2.54 # centimeters in inches

tau = time_mat_test[:,0]
idx_train = np.where((tau!=N) & (tau!=0))[0]
col_idx = tau[idx_train]
# fig = plt.figure(figsize=(8.5/cm,6/cm), dpi = 100)
fig= plt.figure()
ax = fig.add_subplot(111)
ax.scatter(col_idx, S_test[idx_train, col_idx, 0], marker='.',  s = 0.8,alpha= 0.2,c= 'gray',label=r'$\Delta$'+'-algorithm')
ax.set_xlim(0, N)
ax.set_ylim(55, 100)
ax.set_xlabel('$n$')
ax.set_ylabel(r'$\tilde{X}_{n}$',rotation=0)
ax.plot(np.arange(N+1),boundary_flip ,c='blue',ls = 'solid', lw =1,label = 'theoretical boundary $b(n)$')
ax.plot(np.arange(N+1),S_t_mean ,c='red',ls = 'dashed',lw = 1, label = 'averaged boundary '+r'$\bar{b}(n)$')
# leg = ax.legend()
# # for lh in leg.legendHandles: 
# #     lh.set_alpha(1)




#%%


# plot some stopped paths when k = 30 and observe that the paths are dropping

col_idx = tau[idx_train]
t = 30
idx = np.where(col_idx == t)[0]
fig = plt.figure()
ax = fig.add_subplot(111)

for i in idx[:10]:
    ax.plot(S[idx_train[i], :])
ax.set_xlim(0, N)
ax.set_xlabel('n')
ax.set_ylabel(r'$S$',rotation=0)
ax.grid()

#The following is to plot 3 boundaries in one graph.
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(col_idx, S[idx_train, col_idx], s = 5, c='black',label='the tree model')
ax.set_xlim(0, N)
ax.set_ylim(55, 100)
ax.set_xlabel('$n$')
ax.set_ylabel(r'$S_{k}$',rotation=0)
ax.plot(np.arange(N+1),boundary_flip , label = 'theoretical boundary $b(t)$')
ax.plot(np.arange(N+1),S_t_mean , label = r'$S_{mean}(k)$')
ax.legend()


previous = S[idx_train[idx], t-1]
current  = S[idx_train[idx],  t ]
diff = previous - current 
print(diff.mean())
plt.hist(diff, density = True, bins = 50)

