#!/usr/bin/env python
# coding: utf-8

import numpy as np
import random
import math
import time
from sklearn.linear_model import LogisticRegression
from concurrent.futures import ProcessPoolExecutor as Pool
from mrandom import Rand
from mRGUCB import RGUCB
from mDBGD import DBGD
from mELOMLE import EloSMLE
from MaxInELO import MaxInELO
import os
import matplotlib.pyplot as plt

class GridSearch:
    def __init__(self, paras,name,payoff,melo,K,T,iter,rep,dim=8,save_path=''):
        self.paras = paras
        self.name=name#model_name
        self.payoff=payoff
        self.melo=melo #whether melo
        self.K=K#number of players
        self.T=T#rounds
        self.iter=iter#step size
        self.rep=rep#repeat how many times
        self.dim=dim#dimision of melo
        self.save_path=save_path+name+'/'
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)




    def sgdts_sim(self,para):
        seed,alpha,lr,tau,meta,delta=para

        t0 = time.time()

        np.random.seed(seed)
        random.seed(seed)
        rating, C, Reg,all_rate = self.eloobj.sampling(alpha, lr, tau, meta, delta, save_rate=1)
        rating=np.array(rating)
        C=np.array(C)
        Reg=np.array(Reg)

        para_str=str(seed)+'_'+str(alpha)+'_'+str(lr)+'_'+str(tau)+'_'+str(meta)+'_'+str(delta)#alpha,lr,tau,meta,delta
        np.save(self.save_path+para_str+'_'+'Rate.npy',rating)
        np.save(self.save_path+para_str+'_'+'C.npy',C)
        np.save(self.save_path+para_str+'_'+'Reg.npy',Reg)
        np.save(self.save_path + para_str + '_' + 'ALLRate.npy', all_rate)


    def tune_Avg_para(self,eloobj):

        self.eloobj=eloobj
        tbest = float('Inf')

        best = float('Inf')
        para = None
        reg=None
        index=0
        st=time.time()
        para_profiles=[]
        bset_top1_dis=float('Inf')
        numpros=len(self.paras['alpha'])*len(self.paras['eta'])*len(self.paras['delta'])*len(self.paras['meta'])*self.rep
        for alpha in self.paras['alpha']:
            for lr in self.paras['eta']:
                for C in self.paras['C']:
                    y = []
                    tau = int(max(self.K, math.log(self.T)) * C)
                    for meta in self.paras['meta']:
                        for delta in self.paras['delta']:
                            for seed_number in range(self.rep):
                                index+=1
                                seed=seed_number+1#(rep-1)*paralen + index
                                para_profiles.append((seed,alpha,lr,tau,meta,delta))
        num_in_parallel = min(numpros,35*self.rep)
        if self.name=='MaxInP':
            num_in_parallel=7*self.rep
        with Pool(num_in_parallel) as pool:
            ix = 0
            for exp_data in pool.map(self.sgdts_sim, [i for i in para_profiles]):
                ix += 1

