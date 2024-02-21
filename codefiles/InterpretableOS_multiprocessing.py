# -*- coding: utf-8 -*-
"""
Created on Wed May 26 11:18:38 2021

@author: 41546
Code is is to reproduce numerical results in the paper ' Interpretable optimal stopping'
"""


#%%
import pandas as pd
import numpy as np
import math
import time
from pandas.core.common import flatten
import multiprocessing
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
            riskless interest rate.
        N : int
            number of exercise oppotunities.
        K : int
            sample size.
        d : int
            number of stocks.
        delta : float, optional
            DESCRIPTION. The default is 0.
        symmetric : bool, optional
            set the symmetric feature of the volatility of stocks. 
            The default is True.
        """
        self.T = T                   #terminal time
        self.C=C                     #strike price 
        self.delta=delta             #dividend
        self.So=So                   #initial price
        self.r=r                     #riskless interest rate   
        self.N=N                     #number of exercise oppotunity 
        self.K=K                     #number of paths
        self.d=d                     #number of assets
        self.symmetric = symmetric
        self.sigma=self.cal_sigma(sigma) #volatility
        self.dt=self.T/(self.N - 1)
    
    def cal_sigma(self, sigma):
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
        S : numpy array
            stock price matrix with size (K,N+1,d).

        """       
        S = np.zeros((self.K, self.N + 1, self.d),dtype= np.float32)
        S[:,0:2,:] = self.So  
        Z=np.random.standard_normal((self.K, self.N-1, self.d)).astype(np.float32)
        S[:,2:,] = self.So*np.exp(np.cumsum((self.r-self.delta-0.5*self.sigma**2)*self.dt+self.sigma*np.sqrt(self.dt)*Z, axis=1))
        return S
    

#%%


def basisfunction(prices,K,idx,C,B,timestep,beta):
    '''
    generate basis functions and discounted payoff given simulated paths of assets 
    
    '''
    maxprice=np.amax(prices,axis=2)
    KOind=np.cumprod(maxprice<B,axis=1)#y_t  
    PAYOFF=np.maximum(0,maxprice - C)*KOind #g_t
    timematrix=np.ones((K,timestep))*np.arange(1,timestep+1,1)
    dis_PAYOFF=PAYOFF*(beta**(timematrix - 1))
    if idx==0:
        basisf=np.stack((timematrix,PAYOFF,timematrix),axis=2)
    if idx==1:
        
        basisf=np.dstack((timematrix.reshape((K,timestep,1)),prices))
    if idx==2:
        basisf=np.dstack((timematrix.reshape((K,timestep,1)),
                                  prices,PAYOFF))
    if idx==3:
        basisf=np.dstack((timematrix.reshape((K,timestep,1)),
                                  prices,timematrix.reshape((K,timestep,1))))
    if idx==4:
        basisf=np.dstack((timematrix.reshape((K,timestep,1)),
                                  prices,np.stack((timematrix,PAYOFF),axis=2)))
    if idx==5:
        basisf=np.dstack((timematrix.reshape((K,timestep,1)),prices,
                                  np.stack((timematrix,PAYOFF,KOind),axis=2)))

    return basisf,dis_PAYOFF



#%%
def get_nostoptime(leaf,leavesdata,i,data,action):
    """
    to calculate equation (13) in the paper

    """
    #paper: equation (13)
    TIM=[]#no-stop time

    for key,value in leavesdata.items():
        if (key==leaf) or (action[key]=='go'):
            continue
        elif len(value[i])==0:
            continue
        else:
            TIM.append(value[i][0,0])#currentnodedata[i,t,0] is time index

    if len(TIM)==0:
        return 10000
    else:
        return min(TIM)

    
def one_process_gira_solution(x, b_wilst, f_nslst, di):

    def stopF_w(x, b_wi, f, di):
        if di == 'left':
            return f[np.searchsorted(-b_wi, -x, side='left')]
        else:
            return f[np.searchsorted(b_wi, x, side='right')]

    a = np.zeros(x.shape[0])
    for b_wi, f in zip(b_wilst, f_nslst):
        a += stopF_w(x, np.asarray(b_wi), np.asarray(f), di)
    return a


def multi_process_gira_solution(b_wilst, f_nslst, di):
    """
    multiprocessing process

    """
    def error_handler(e):
        print("-->{}<--".format(e.__cause__))

    intval = np.unique(list(flatten(b_wilst)))
    x2 = np.concatenate(([-10000], (intval[:-1] + intval[1:]) / 2, [10000])).astype(np.float32)
    pool = multiprocessing.Pool()
    process_num = multiprocessing.cpu_count()
    step = int(len(f_nslst) / process_num) + 1
    result = []
    for i in range(process_num):
        pool.apply_async(one_process_gira_solution,
                         args=(x2, b_wilst[(step * i):(step * (i + 1))], f_nslst[(step * i):(step * (i + 1))], di),
                         callback=lambda ele: result.append(ele),
                         error_callback=error_handler)
    pool.close()
    pool.join()
    return sum(result)

def OptimizeSplitPoint(v,l,di,leaves,data,leavesdata,action,I,dis_PAYOFF):
    """
    Algorithm 3 in the paper

    Parameters
    ----------
    v : int
        column index of feature.
    l : int
        index of leaf.
    di : string
        direction: left or right.
    leaves : dictionary
        dictionary that stores indices of each leaf.
    data : array
         training data.
    leavesdata : dictionary
        dictionary that stores data in each leaf.
    action : dictionary
        dictionary that stores action in each leaf.
    I : int
        number of paths.

    Returns
    -------
    float
        optimal value.
    float
        optimal split value.

    """
    #need to return: theta,objective, leftchild dataset, rightchild dataset
    f_nslst=[];b_wilst=[]
    for i in range(0,I):
        #computer no-stop time ,equation (13):stoptime is a number
        stoptime=int(get_nostoptime(l,leavesdata,i,data[i],action))
        #no-stop value, equation (14):
        if stoptime==10000:
            f_ns=0
        else:
            f_ns=dis_PAYOFF[i,stoptime-1]
        #calculate in-leaf period, equation (15):
        S_w=data[i][data[i][:,0]<stoptime,0].astype(int)
        if di=='left':
            #P_w, left-stop permissible stopping period, equation (17):
            P_w=[] # dont nieed to reorder P_w, because it is sorted already
            b_wi=[] #corresponding breakpoint of P_w
            if (len(S_w)==0) &(stoptime!=10000):
                f=[f_ns,f_ns]
                b_wi.append(10000)
                f_nslst.append(f)       
                b_wilst.append(b_wi)

            if len(S_w)!=0:
                for t in S_w:
                    idx = int(np.where(data[i][:,0]==t)[0][0])
                    if t==S_w[0]:
                        P_w.append(t)
                        b_wi.append(data[i][idx,1])
                    elif data[i][idx,1]<b_wi[-1]:
                        P_w.append(t)
                        b_wi.append(data[i][idx,1])
                        
                f=np.ndarray.tolist(dis_PAYOFF[i,np.asarray(P_w)-1])
                f.append(f_ns)
     
                f_nslst.append(f)
                b_wilst.append(b_wi)
        if di=='right':
            #calculate permissible stopping period, equation (16) in the paper:
            P_w=[] # dont need to re-order P_w, because it is sorted already
            b_wi=[] #corresponding breakpoint of P_w
            if (len(S_w)==0) &(stoptime!=10000):
                f=[f_ns,f_ns]
                b_wi.append(-10000)
                f_nslst.append(f)       
                b_wilst.append(b_wi)

            if len(S_w)!=0:
                for t in S_w:
                    idx = int(np.where(data[i][:,0]==t)[0][0])
                    if t==S_w[0]:
                        P_w.append(t)
                        b_wi.append(data[i][idx,1])
                    elif data[i][idx,1]>b_wi[-1]:
                        P_w.append(t)
                        b_wi.append(data[i][idx,1])
                        #f: discounted payoff
                f=np.ndarray.tolist(dis_PAYOFF[i,np.asarray(P_w)-1])
                f.append(f_ns)
            
                f_nslst.append(f)
                b_wilst.append(b_wi)
    #median value of interval
    intval=np.sort( np.unique(list(flatten(b_wilst))))
    

    F_w=[]        

    F_w = multi_process_gira_solution(b_wilst,f_nslst,di)/I 
    if F_w.shape!=(): 
        obj=max(F_w)
        if F_w[0]==obj:
            theta=-10000
        elif F_w[-1]==obj:
            theta=10000
        else:
            idx=np.where(F_w==obj)[0].tolist()
            theta=(intval[idx[0]-1]+intval[idx[-1]])/2
        return obj,theta
    else:
        return -10000,-10000

def test_split(v, theta, data):
    """
    

    Parameters
    ----------
    v : int
        feature index.
    theta : float
        optimal spilt value.
    data : array
        data in the current node.

    Returns
    -------
    left : array
        data allocated in the left child of the current node.
    right : array
        data allocated in the right child of the current node.

    """
    
    left, right=list(),list()
    for i in data:
        left.append(i[i[:,v]<=theta,:])
        right.append(i[i[:,v]>theta,:])
    return left, right

def GrowTree(N,leaves,splits,leftchild,rightchild,l,data,leavesdata,theta,v):
    """
    Algorithm 2 in the paper

    """
    
    leaves.remove(l)
    del leavesdata[l]
    leaves.extend((len(N)+1,len(N)+2))
    splits.append(l)
    leftchild[l]=len(N)+1
    rightchild[l]=len(N)+2
    left,right=test_split(v,theta,data)
    leavesdata[len(N)+1]=left
    leavesdata[len(N)+2]=right
    N.extend((len(N)+1,len(N)+2))
    return N,leaves,splits,leftchild,rightchild,leavesdata

def predict(N,leaves,splits,leftchild,rightchild,currentnodedata,
            splitvar,splitindex,action,leavesdata,node=1):
    """
    use the trained model to split test data
    """
    
    if node in leaves:
        leavesdata[node]=currentnodedata
    else:
        del leavesdata[node]
        leavesdata[leftchild[node]],leavesdata[rightchild[node]]=test_split(
            splitindex[node],splitvar[node],currentnodedata)
    if (node+1) in N:
        return predict(N,leaves,splits,leftchild,rightchild,
                       leavesdata[node+1],splitvar,splitindex,
                       action,leavesdata,node+1)
    else:
        return leavesdata

def outofsample(leavesdata,action,dis_PAYOFF,I):
    """
    use the data ,which is split by funtion 'predict', to predict the value of options

    """
    x=np.zeros(I);I_list=[]
    for i in range(I):
        idx=[]
        for key,data in leavesdata.items():
            if action[key]=='go':
                continue 
            else:
                if len(data[i])==0:
                    continue
                else:
                    I_list.append(i)
                    idx.append(int(data[i][0,0]))
        if len(idx)!=0:
            x[i]=dis_PAYOFF[i,min(idx)-1]
    idx=np.in1d(np.arange(I),np.array(I_list))
    x[~idx]=0#x[~idx]=dis_PAYOFF[~idx,-1]
    value=np.mean(x)
    return value
    
def algorithm1(dis_PAYOFF,basisf,I,gamma=0.005):
    """
    algorithm 1 in the 'interpretable optimal stopping' paper

    Parameters
    ----------
    dis_PAYOFF : array
        discounted payoff.
    basisf : array
        training data.
    I : int
        number of paths.
    gamma : float, optional
        user-specified tolerance factor. The default is 0.005.

    Returns
    -------
    N : list
        indices of all nodes and leaves.
    leaves : dictionary
        indices of leaves.
    splits : list
        indices of splits.
    leftchild : dictionary
        indices of left child of each split node.
    rightchild : dictionary
         indices of right child of each split node.
    splitindex : dictionary
        split index of each node.
    splitvar : dictionary
        optimal split value of each node.
    action : dictionary
        action of each leaf.
    leavesdata : dictionary
        data in each node and leaf.
    Z : float
        optimal value.

    """
    leavesdata={};N=[1];leaves=[1];splits=[];leftchild={};
    rightchild={}
    splitindex={};splitvar={};action={1:'go'}
    leavesdata={1:basisf}
    existimprovement=True
    Z=0
    while existimprovement:
        #action: dictionary--dictionary,keys: node index;values: {go, stop}
        #splitpoint is to save the optimizesplit point;similar to objective(dictionary)
        splitpoint={};objective={}
    
        for l in leaves:
            for v in range(1,basisf.shape[2]):
                #first column of basisfunction as time index, easier to manipulate numpy array
                for di in direction:
                    currentnodedata=leavesdata[l]
                    testdata=[i[:,(0,v)] for i in currentnodedata]
                    obj,theta=OptimizeSplitPoint(
                            v,l,di,leaves,testdata,leavesdata,action,I, dis_PAYOFF)


                    objective[l,v,di]=obj
                    splitpoint[l,v,di]=theta
    
        existimprovement=max(objective.values())>=(1+gamma)*Z
        if max(objective.values())>Z:
            l =max(objective,key=objective.get)[0]
            v =max(objective,key=objective.get)[1]
            di=max(objective,key=objective.get)[2]
            N,leaves,splits,leftchild,rightchild,leavesdata=GrowTree(
                    N,leaves,splits,leftchild,rightchild,l,leavesdata[l],
                    leavesdata,splitpoint[l,v,di],v)       
            splitindex[l]=v
            splitvar[l]=splitpoint[l,v,di]
            if di=='left':
                action[leftchild[l]]='stop'
                action[rightchild[l]]='go'
            else:
                action[leftchild[l]]='go'
                action[rightchild[l]]='stop'
            Z=max(objective.values())
    return (N,leaves,splits,leftchild,rightchild,splitindex,
            splitvar,action,leavesdata,Z)


#%%

#initialization # Section 5 in 'interpretable optimal stopping' paper
if __name__ == '__main__':

    K = 20000      # number of in-sample trajectories
    K_test = 100000   # number of out-of-sample trajectories
    r = 0.05              # riskless interest rate of underlying assets
    timestep = 54         # number of time steps     
    T = 3          # maturity time
    C = 100               # strike price
    B = 170               # barrier of Bermudan max-call option
    n_s =[ 4,8,16]             # set of possible number of underlying assets
    sigma = 0.2           # volatility of assets
    S0_s = [90,100,110]   # set of possible initial prices of assets
    
    reptimes = 1         #number of replications
    beta=math.exp(-r*T/timestep) # discount factor
    columnname=['d','Method','state variables',
                'p=90','SD90','p=100','SD100','p=110',
                'SD110','time_train90','time_train100','time_train110',
                'time_test90','time_test100','time_test110']
    value=pd.DataFrame(columns=columnname)
    value['Method']='Tree'
    statevariable_s=['payoff time','prices','prices payoff',
                     'prices time','prices time payoff',
                     'prices time payoff KOind']
    value['d']=np.repeat(n_s,len(statevariable_s))
    value['state variables']=statevariable_s*len(n_s)
    value['Method']='Tree'
    direction=['left','right']

    for d in n_s:
        print('d=',d)
        for S0 in S0_s:
            print('S0=',S0)
            v=np.zeros((reptimes,len(statevariable_s)))
            comptime=np.zeros((reptimes,len(statevariable_s)))
            comptime_test=np.zeros((reptimes,len(statevariable_s)))

            for reptime in range(reptimes):
                filename = 'd_' + str(d) + 'So_' + str(S0) + 'rep_' + str(reptime)+'.obj'
                S=stock(T, C, sigma, S0, r, timestep, K, d)
                np.random.seed(d + S0 +reptime)
                paths= S.simulatepaths()
    
                S=stock(T, C, sigma, S0, r, timestep, K_test, d)
                np.random.seed(42 + reptime)
                paths_test = S.simulatepaths()     
                
                for i in range(1):
                    print('i=',i)
                    basisf_train,dis_PAYOFF_train=basisfunction(paths[:,1:,:],K,i,C,B,timestep,beta)
                    start_train=time.time()
                    (N,leaves,splits,leftchild,rightchild,
                     splitindex,splitvar,action,leavesdata,Z)=algorithm1(
                             dis_PAYOFF_train,basisf_train,K)
                    finish_train=time.time()
                    
                    basisf_test,dis_PAYOFF_test=basisfunction(paths_test[:,1:,:],K_test,i,C,B,timestep,beta)
                    leavesdata={};leavesdata[1] = basisf_test
                    leavesdata=predict(N,leaves,splits,leftchild,
                                       rightchild,leavesdata[1],splitvar,splitindex,
                                       action,leavesdata,node=1) 
                    start_test = time.time()
                    Z=outofsample(leavesdata,action,dis_PAYOFF_test,K_test)
                    finish_test = time.time()
                    v[reptime,i]=Z
                    comptime[reptime,i]=round(finish_train-start_train,1)
                    comptime_test[reptime,i]=round(finish_test-start_test,1)


            value.loc[value['d']==d,'p='+str(S0)]=np.mean(v,axis=0)
            value.loc[value['d']==d,'SD'+str(S0)]=np.std(v,axis=0)
            value.loc[value['d']==d,'time_train'+str(S0)]=np.mean(comptime,axis=0)
            value.loc[value['d']==d,'time_test'+str(S0)]=np.mean(comptime_test,axis=0)
    
            print('value:',np.mean(v,axis=0) )
            print('train time:',np.mean(comptime,axis=0) )

