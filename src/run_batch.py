#!/usr/bin/env python
# coding: utf-8


import numpy as np
import random
import math
import time
import os
import argparse
from sklearn.linear_model import LogisticRegression
import pickle


from grid_tune import GridSearch
from concurrent.futures import ProcessPoolExecutor as Pool
from mrandom import Rand
from mRGUCB import RGUCB
from mDBGD import DBGD
from mELOMLE import EloSMLE
from MaxInELO import MaxInELO
import sys
import os

mode='Avg'
rep=5
d=1
K=1
iter=1
game_name=''
path=''
name=sys.argv[1]
melo=int(sys.argv[2])
T=2000

if name=='random':
    parameters = {
        'alpha': [0],
        'eta': [0.01, 0.05, 0.1, 0.5, 1, 5],
        'meta': [0],
        'delta': [0],
        'C': [0],
        'explore': [1],
        'stability': 10 ** (-6)
    }
if name=='RGUCB':
    parameters = {
        'alpha': [0],
        'eta': [0.01, 0.05, 0.1, 0.5, 1, 5],
        'meta': [0],
        'delta': [0.2],
        'C': [0],
        'explore': [1],
        'stability': 10 ** (-6)
    }
if name=='DBGD':
    parameters = {
        'alpha': [0],
        'eta': [0.01, 0.05, 0.1, 0.5, 1, 5],
        'meta': [0],
        'delta': [0],
        'C': [0],
        'explore': [1],
        'stability': 10 ** (-6)
    }
if 'MaxInP' in name:
    parameters = {
        'alpha': [0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2],
        'eta': [0],
        'meta': [0],
        'delta': [0],
        'C': [0.7],
        'explore': [1],
        'stability': 10 ** (-6)
    }
if name=='MaxInELO':
    parameters = {
        'alpha':[0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2],
        'eta': [0.01, 0.05, 0.1, 0.5, 1, 5],
        'meta': [0],
        'delta': [1.0,3.0,5.0],
        'C': [0,0.1,0.3,0.5,0.7,0.9,1,2,4,8],
        'explore': [0],
        'stability': 10 ** (-6)
    }
def process(game):
    global T,d,K,iter,game_name,path,name
    game_name=game
    model = 'logistic'
    dist = 'ber'
    if dist != 'ber' and model == 'logistic':
        raise NameError('logistic regression only supports bernoulli reward')
    if melo==1:
        dir='AAAIMELO_Batch/'
    else:
        dir='AAAIELO_Batch/'
    if not os.path.exists(dir):
        os.mkdir(dir)
    dir+=mode+'/'
    if not os.path.exists(dir):
        os.mkdir(dir)
    if not os.path.exists('seconds/'):
        os.mkdir('seconds/')
    path = dir+game_name+'/'

    if not os.path.exists(path):
        os.mkdir(path)
    if not os.path.exists(path + name + '/'):
        os.mkdir(path + name + '/')


def get_payoff(game_name):
    if game_name=='transitive' or game_name=='mtransitive':
        payoff=np.load("transitive40.npy")
        K=40
        T=1000
        payoff=payoff*2.0-1.0
        d = K

    else:
        payoff=np.load('games/{}.npy'.format(game_name))
        K = payoff.shape[0]
        T=2000
    return payoff,K

def get_Avg_para(game_name):
    payoff,K=get_payoff(game_name)
    for i in range(K):
        payoff[i][i]=0
    if melo==0:
        dim=0
    else:
        dim=8
    gridsearch = GridSearch(parameters, name, payoff, melo, K, T, iter, rep=rep,dim=dim,save_path=path)
    if name=='random':
        obj=Rand(K,T,iter,(payoff+1.0)/2.0,melo,dim=dim)
    if name=='RGUCB':
        obj = RGUCB(K, T, iter,(payoff+1.0)/2.0,melo,dim=dim)
    if name == 'DBGD':
        obj = DBGD(K, T, iter,(payoff+1.0)/2.0,melo,dim=dim)
    if name=='MaxInELO':
        obj = MaxInELO(K, T, iter,(payoff+1.0)/2.0,melo,dim=dim)

    if 'MaxInP' in name:
        obj = EloSMLE(K, T, iter, (payoff + 1.0) / 2.0,melo)
    gridsearch.tune_Avg_para(eloobj=obj)

if __name__=='__main__':
    if melo==1:
        games=['Kuhn-poker']
    else:
        games = ['Elo game']
    if mode=='Avg':
        for game in games:
            process(game)
            get_Avg_para(game)
