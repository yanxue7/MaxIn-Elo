import numpy as np
from mrandom import Rand
from mRGUCB import RGUCB
from mDBGD import DBGD
from mELOMLE import EloSMLE
from MaxInELO import MaxInELO
import matplotlib.pyplot as plt
from common_funcs import smooth
import pickle
import scipy.stats

import random
import os
import time
import json
import resource
_, axes = plt.subplots(1, 2, figsize=(10.0, 2.5))#2.7))
fz=14
def initial():

    _, axes = plt.subplots(1,2,figsize=(5.0,2.7))


    plt.subplot(121)
    plt.xticks(fontsize=fz)
    plt.yticks(fontsize=fz)
    plt.xlabel("Rounds $t$", fontsize=fz)
    plt.ylabel("RR", fontsize=fz)

    plt.subplot(122)

    plt.xlabel("Rounds $t$", fontsize=fz)
    plt.ylabel("Regret", fontsize=fz)
    plt.xticks(fontsize=fz)
    plt.yticks(fontsize=fz)

def picture(Reg,label):

    T = Reg.shape[-1]
    rep = np.shape(Reg)[0]
    top1dis = Reg[:, 0, 15:T]
    for i in range(rep):
        top1dis[i] = smooth(top1dis[i])

    regret=Reg[:,-3,:]
    regret=regret[:,:T]
    print(regret.shape)

    #
    plt.subplot(1, 2, 2)
    In=np.arange(np.shape(regret)[-1])

    se = scipy.stats.sem(regret, axis=0)

    plt.fill_between(In,regret.mean(0) - se, regret.mean(0) + se,alpha=0.2)
    plt.plot( In,regret.mean(0), label=label)



    plt.subplot(1, 2, 1)

    In = np.arange(np.shape(top1dis)[-1])+100

    se = scipy.stats.sem(top1dis, axis=0)

    plt.fill_between(In, top1dis.mean(0) - se, top1dis.mean(0) + se, alpha=0.2)
    plt.plot(In, top1dis.mean(0), label=label)

class Process():
    def __init__(self,melo=0):
        self.savedir='C_comp'

        self.melo=melo


        if melo==0:
            self.games=['Elo game']
            self.savedir+='_elo/'
        else:
            self.games=['Kuhn-poker']
            self.savedir+='_melo/'

        self.models=['MaxInELO']
        if not os.path.exists(self.savedir):
            os.mkdir(self.savedir)

    def read_para(self,game_name,model):
        paras = json.load(open('AAAIMELO/Avg/{}/{}/max_best_para.json'.format(game_name, model), 'r'))

        return paras
    def get_sampler(self,game_name,model,dim):
        payoff = np.load('games/{}.npy'.format(game_name))
        K = payoff.shape[0]
        T = 2000

        iter=1 #step size , per iter to save a data point
        if model == 'random':
            EloSGDobj = Rand(K, T, iter, (payoff + 1.0) / 2.0, self.melo,dim=dim)
        if model == 'RGUCB':
            EloSGDobj = RGUCB(K, T, iter, (payoff + 1.0) / 2.0, self.melo,dim=dim)
        if model == 'DBGD':
            EloSGDobj = DBGD(K, T, iter, (payoff + 1.0) / 2.0, self.melo,dim=dim)
        if model == 'MaxInELO':
            EloSGDobj = MaxInELO(K, T, iter, (payoff + 1.0) / 2.0, self.melo,dim=dim)
        if model == 'MaxInP':
            EloSGDobj = EloSMLE(K, T, iter, (payoff + 1.0) / 2.0,self.melo)
        return EloSGDobj


    def runs(self,dim):
        for game_name in self.games:
            if not os.path.exists(self.savedir+game_name):
                os.mkdir(self.savedir+game_name)

            time_dir={}
            for model in self.models:

                paras=self.read_para(game_name=game_name,model=model)
                Sampler=self.get_sampler(game_name,model,dim)
                Rate=[]
                MC=[]
                Reg=[]
                ALL_R=[]
                times=0
                for para in paras:
                    split_para = para.split('_')  # seed alpha lr tau g1 g2
                    seed=split_para[0]
                    alpha=split_para[1]
                    lr=split_para[2]
                    tau=split_para[3]
                    g1=split_para[4]
                    g2=split_para[5]
                    #seed, alpha, lr, tau, g1, g2 = para
                    seed=int(seed)
                    alpha=float(alpha)
                    tau=int(tau)
                    lr=float(lr)
                    g1=float(g1)
                    g2=float(g2)
                    np.random.seed(seed)
                    random.seed(seed)
                    st = time.time()
                    ratings,mc,reg,all_rate = Sampler.sampling(alpha, lr, tau, g1, g2,save_rate=1)
                    en = time.time()
                    times+=en-st
                    Rate.append(ratings)
                    MC.append(mc)
                    Reg.append(reg)
                    ALL_R.append(all_rate)
                times/=len(paras)
                model=model
                np.save(self.savedir+ '{}/{}rateRR_{}.npy'.format(game_name,model,str(dim)),np.array(Rate))
                np.save(self.savedir+ '{}/{}RR_{}.npy'.format(game_name,model,str(dim)),np.array(Reg))
                np.save(self.savedir + '{}/{}allrateRR_{}.npy'.format(game_name, model,str(dim)), np.array(ALL_R))

                time_dir[model]=times
                print(time_dir)
        return Reg

if __name__=="__main__":
    melo=1
    obj=Process(melo)
    initial()
    paras=obj.read_para('Kuhn-poker','MaxInELO')
    path='AAAIMELO/Avg/'

    cdatas={}
    Reg=[]
    for para in paras:
        reg = np.load(path +'Kuhn-poker/MaxInELO/'+ para + '_Reg.npy')
        Reg.append(reg)
        
    cdatas[8]=np.array(Reg)

    for dim in [0,2,4,16]:
        Reg=obj.runs(dim)
        cdatas[dim] = np.array(Reg)

    cdatas = pickle.load(open("C_comp_melo/Kuhn-poker/comparecs.pkl", "rb"))
    for dim in [0, 2, 4,8, 16]:
        picture(cdatas[dim],r'$C={}$'.format(dim))
    plt.title('Kuhn-poker', fontsize=fz)
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    plt.grid()
    plt.tight_layout()

    plt.subplot(1, 2, 2)
    plt.grid()
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    plt.savefig('finalplot/comparec.pdf')
    plt.show()


