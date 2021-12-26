
import matplotlib.pyplot as plt

import numpy as np
import pickle
import scipy.stats
import os
import random
from mrandom import Rand
from mRGUCB import RGUCB
from mDBGD import DBGD
from mELOMLE import EloSMLE
from MaxInELO import MaxInELO
import json
import sys

fz = 14
Name=''
path='finalplot/'
mode=sys.argv[1]
melo=int(sys.argv[2])

def smooth(scalar,weight=0.99):

    scalar=scalar.tolist()
    last = scalar[0]
    smoothed = []
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)
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


def logit(P):
    P=(P+1.0)/2.0
    esp=1e-2
    n=P.shape[0]
    for i in range(n):
        for j in range(n):
            if P[i,j]==0:
                P[i,j]+=esp
            if P[i,j]==1:
                P[i,j]-=esp
    A=np.log(P)-np.log(1.0-P)
    Borda=np.mean(A,axis=-1)
    return Borda

def process(X,Y):
    T=X.shape[-1]
    rep=X.shape[0]
    X.reshape((-1,T))
    Y.reshape((-1,T))

    regret=np.zeros([rep,T])
    P=np.load('games/{}.npy'.format(Name))
    Borda=logit(P)
    optimal=np.max(Borda)
    for w in range(rep):
        for i in range(T):
            regret[w,i]=regret[w,i-1]+optimal-0.5*(Borda[int(X[w,i])]+Borda[int(Y[w,i])])
    return regret


def picture(Reg,label):
    rep = np.shape(Reg)[0]

    repr=Reg.shape[0]
    T=Reg.shape[-1]

    Reg=np.reshape(Reg,(repr,-1,T))
    print(Reg.shape)

    top1dis = Reg[:, 0, 15:T]
    for i in range(rep):
        top1dis[i] = smooth(top1dis[i])

    regret=process(Reg[:,-2,:],Reg[:,-1,:])
    regret=regret[:,:T]
    print(regret.shape)

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


def get_rates(path):
    if mode=='Avg':
        paras=json.load(open(path+'best_para.json','r'))
    else:
        paras = json.load(open(path + 'max_best_para.json', 'r'))
    rates=[]
    Reg=[]
    for i in range(rep):
        para=paras[i]
        rates.append(np.load(path+para+'_Rate.npy'))
        Reg.append(np.load(path+para+'_Reg.npy'))
    rates=np.array(rates)
    Reg=np.array(Reg)
    return rates,Reg
if __name__=='__main__':
    if melo==0:

        games=['Elo game','Transitive game']
        games+=['Elo game + noise=0.1','Triangular game']#
        games+=['Elo game + noise=0.01', 'Elo game + noise=0.05']
    else:

        games = ['Kuhn-poker', 'Blotto', 'Disc game', 'tic_tac_toe', 'AlphaStar', 'hex(board_size=3)']

    if not os.path.exists(path):
        os.mkdir(path)
    rep=5
    if melo==0:
        data_path='AAAIELO/'+'Avg/'
    else:
        data_path = 'AAAIMELO/' + 'Avg/'
    for game_name in games:
        initial()
        Name=game_name
        payoff=np.load('games/{}.npy'.format(game_name))
        K=payoff.shape[0]
        T=2000
        true_rate = logit(payoff)
        names = ['random','RGUCB','DBGD','MaxInP','MaxInELO']


        for name in names:
            rates,Reg=get_rates(data_path+game_name+'/'+name+'/')

            picture(Reg,label=name)
        plt.subplot(1, 2, 1)

        plt.title(game_name, fontsize=fz)
        plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

        plt.grid()
        plt.tight_layout()

        plt.subplot(1, 2, 2)
        plt.grid()
        plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        plt.tight_layout()


        plt.savefig(path+'{}_{}.pdf'.format(game_name,mode))
        plt.show()
       
#python Elo_plot.py Max