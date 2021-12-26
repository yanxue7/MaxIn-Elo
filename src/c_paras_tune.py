import numpy as np
import sys
import os
import math
import json
from common_funcs import hit_Ratio,inverseP,RR

dir=''

def get_parameters(name):
    #name = sys.argv[1]  # 'random'
    #melo = int(sys.argv[2])

    if name == 'random':
        parameters = {
            'alpha': [0],  # [0.1,1,2,4,8,10,16],#[1],#[0.1,1,5,10],
            'eta': [0.01, 0.05, 0.1, 0.5, 1, 5],  # total 6
            'meta': [0],  # [0.01, 0.05, 0.1, 0.5, 1, 5, 10],
            'delta': [0],
            'C': [0],  # list(range(1, 11)),  # total 10
            'explore': [1],  # [0.01, 0.1, 1, 5, 10],  # total 5
            'stability': 10 ** (-6)
            # initialize matrix V_t = 10**(-6) * identity matrix to ensure the stability of inverse (UCB-GLM)
        }
    if name == 'RGUCB':
        parameters = {
            'alpha': [0],  # [0.1,1,2,4,8,10,16],#[1],#[0.1,1,5,10],
            'eta': [0.01, 0.05, 0.1, 0.5, 1, 5],  # total 7
            'meta': [0],  # [0.01, 0.05, 0.1, 0.5, 1, 5, 10],
            'delta': [0.2],  # [0.1, 0.2, 0.01],
            'C': [0],  # list(range(1, 11)),  # total 10
            'explore': [1],  # [0.01, 0.1, 1, 5, 10],  # total 5
            'stability': 10 ** (-6)
            # initialize matrix V_t = 10**(-6) * identity matrix to ensure the stability of inverse (UCB-GLM)
        }
    if name == 'DBGD':
        parameters = {
            'alpha': [0],  # [0.1,1,2,4,8,10,16],#[1],#[0.1,1,5,10],
            'eta': [0.01, 0.05, 0.1, 0.5, 1, 5],  # total 7
            'meta': [0],  # [0.01, 0.05, 0.1, 0.5, 1, 5, 10],
            'delta': [0],
            'C': [0],  # list(range(1, 11)),  # total 10
            'explore': [1],  # [0.01, 0.1, 1, 5, 10],  # total 5
            'stability': 10 ** (-6)
            # initialize matrix V_t = 10**(-6) * identity matrix to ensure the stability of inverse (UCB-GLM)
        }
    if 'MaxInP' in name:
        parameters = {
            'alpha': [0.2,0.4,0.6,0.8,1.2,1.4,1.6,1.8]+[0.1,1,2],
            'eta': [0],  # total 7
            'meta': [0],  # [0.01, 0.05, 0.1, 0.5, 1, 5, 10],
            'delta': [0],
            'C': [0.7],#[0.1,0.3,0.5,0.7,0.9,1.0],#[0.7],  # list(range(1, 11)),  # total 10
            'explore': [1],  # [0.01, 0.1, 1, 5, 10],  # total 5
            'stability': 10 ** (-6)
            # initialize matrix V_t = 10**(-6) * identity matrix to ensure the stability of inverse (UCB-GLM)
        }
    if name == 'MaxInELO':
        parameters = {
            'alpha': [0.2,0.4,0.6,0.8,1.2,1.4,1.6,1.8]+[0.1,1,2],
            'eta': [0.01, 0.05, 0.1, 0.5, 1, 5],  # total 7
            'meta': [0],  # [0.01, 0.05, 0.1, 0.5, 1, 5, 10],
            'delta': [1.0, 3.0, 5.0],  # [0.1,0.5,1.0],#fan shu /Cmax
            'C': [0.7],#[0.1,0.3,0.5,0.7,0.9,1.0],#[0.7],  # list(range(1, 11)),  # total 10
            'explore': [0],  # [0.01, 0.1, 1, 5, 10],  # total 5
            'stability': 10 ** (-6)
        }
    return parameters
def get_performance(path,game):


    reg=np.load(path+'_'+'Reg.npy')
    T=reg.shape[-1]

    st=1500
    en=T
    NDCG1=reg[0,st:en].sum()

    regret = reg[-3, en-1]
    return NDCG1,regret#
def process_avg(games):
    T=1000
    for game in games:
        K = np.load('games/{}.npy'.format(game)).shape[0]

        for model in models:
            save_path = dir + game + '/' + str(dim) + '/' + model + '/'
            #save_path=dir+game+'/'+model+'/'
            paras=get_parameters(model)
            best_per=-1000000000 #larger is better
            min_reg=float('Inf')#small is better
            best_para=[]
            for alpha in paras['alpha']:
                for lr in paras['eta']:
                    for C in paras['C']:
                        y = []
                        tau = int(max(K, math.log(T)) * C)
                        for meta in paras['meta']:
                            for delta in paras['delta']:
                                per_sum=0
                                reg_sum=0
                                para_list=[]
                                para_code=str(alpha)+'_'+str(lr)+'_'+str(tau)+'_'+str(meta)+'_'+str(delta)
                                for i in range(rep):
                                    seed=i+1
                                    para_str=str(seed)+'_'+para_code
                                    para_list.append(para_str)
                                    per,reg=get_performance(save_path+para_str,game)
                                    per_sum+=per
                                    reg_sum+=reg
                                if per_sum>best_per:
                                    best_per=per_sum
                                    min_reg=reg_sum
                                    best_para=para_list
                                else:
                                    if per_sum==best_per and reg_sum<min_reg:
                                        min_reg = reg_sum
                                        best_para=para_list
            print(game, model)
            print(best_para)
            json.dump(best_para,open(save_path+'avg_best_para_poker.json','w'))#(save_path+'best_para.json','w'))
def process_max(games):
    T=1000
    for game in games:
        K = np.load('games/{}.npy'.format(game)).shape[0]

        for model in models:
            save_path=dir+game+'/'+str(dim)+'/'+model+'/'
            paras=get_parameters(model)
            best_para_list = []

            for i in range(rep):
                seed = i + 1

                best_per = -10000  # larger is better
                min_reg = float('Inf')  # small is better
                para_list = []
                best_para = ''

                for alpha in paras['alpha']:
                    for lr in paras['eta']:
                        for C in paras['C']:
                            y = []
                            tau = int(max(K, math.log(T)) * C)
                            if tau==0:
                                tau=1
                            for meta in paras['meta']:
                                for delta in paras['delta']:
                                    per_sum = 0
                                    reg_sum = 0
                                    para_code = str(alpha) + '_' + str(lr) + '_' + str(tau) + '_' + str(
                                        meta) + '_' + str(delta)

                                    para_str = str(seed) + '_' + para_code
                                    #para_list.append(para_str)
                                    per, reg = get_performance(save_path + para_str,game)
                                    per_sum += per
                                    reg_sum += reg
                                    if per_sum > best_per:
                                        best_per = per_sum
                                        min_reg = reg_sum
                                        best_para = para_str
                                    else:
                                        if per_sum == best_per and reg_sum < min_reg:
                                            min_reg = reg_sum
                                            best_para = para_str
                best_para_list.append(best_para)
            print(game,model)
            print(best_para_list)
            json.dump(best_para_list,open(save_path+'max_best_para_poker.json','w'))
            print(save_path+'max_best_para_poker.json')
if __name__=='__main__':
    melo=int(sys.argv[1])
    mode=sys.argv[2]
    rep=int(sys.argv[3])
    dim=int(sys.argv[4])
    models=['MaxInELO']
    if melo ==0:
        games = ['Elo game']
        dir='AAAIELO_C/'
    else:
        games =['Kuhn-poker']
        dir='AAAIMELO_C/'
    dir+='Avg/'
    if mode=='Avg':
        process_avg(games)
    else:
        process_max(games)

#python c_paras_tune.py 0 Max 5 8