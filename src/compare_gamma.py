import numpy as np
import sys
import json
import matplotlib.pyplot as plt
import scipy.stats
from scipy.interpolate import make_interp_spline
from common_funcs import inverseP,get_NDCG,hit_Ratio,MRR
from find_paras import get_performance
import os
import math
mode='best'#sys.argv[1]
para_seed='avg'

topk=20
fz = 14
figpath="finalplot/"
if not os.path.exists(figpath):
    os.mkdir(figpath)
#_, axes = plt.subplots(1,2,figsize=(5.0,2.7))
parameters = {
        'alpha':[0.2,0.4,0.5,0.6,0.8,1,1.2,1.4,1.6,1.8,2],
        'eta': [0.01, 0.05, 0.1, 0.5, 1, 5],
        'meta': [0],
        'delta': [1.0,3.0,5.0],
        'C': [0.7],
        'explore': [0],
        'stability': 10 ** (-6)
    }

def initial():

    #5.0, 2.7#figsize=

    plt.subplot(121)
    plt.xticks(fontsize=fz)
    plt.yticks(fontsize=fz)
    plt.xlabel("Top $K$", fontsize=fz)
    plt.ylabel(r"Hit Ratio@$K$", fontsize=fz)
    plt.subplot(122)

    plt.xlabel("Top $K$", fontsize=fz)
    plt.ylabel(r"NDCG@$K$", fontsize=fz)
    plt.xticks(fontsize=fz)
    plt.yticks(fontsize=fz)

def smooth(Y,k):
    In = np.arange(k) + 1
    num=k*10
    x_new = np.linspace(1, k, num)
    y_smooth=np.zeros((Y.shape[0],num))
    for i in range(Y.shape[0]):
        print(In.shape,Y[i].shape)
        y_smooth[i] =  make_interp_spline(In, Y[i])(x_new)
    return y_smooth
def picture(hit,NDCG,label,linestyle='-',linewidth=3):
    k=10
    hit=hit[:,:k]
    NDCG=NDCG[:,:k]
    hit=smooth(hit,k)
    NDCG=smooth(NDCG,k)
    plt.subplot(1, 2, 1)
    In = np.arange(np.shape(hit)[-1])+1

    se = scipy.stats.sem(hit, axis=0)

    plt.fill_between(In, hit.mean(0) - se, hit.mean(0) + se, alpha=0.2)
    plt.plot(In, hit.mean(0), label=label,linestyle=linestyle,linewidth=linewidth)
    axes[0].plot(In, hit.mean(0), label=label,linestyle=linestyle)



    plt.subplot(1, 2, 2)

    In = np.arange(np.shape(NDCG)[-1]) + 1

    se = scipy.stats.sem(NDCG, axis=0)

    plt.fill_between(In, NDCG.mean(0) - se, NDCG.mean(0) + se, alpha=0.2)
    plt.plot(In, NDCG.mean(0), label=label,linestyle=linestyle,linewidth=linewidth)







def best_gamma(game,model,paras):
    print(paras)
    print('------------------------')
    for alpha in parameters['alpha']:
        new_paras=[]
        for para in paras:
            split_para=para.split('_')#seed alpha lr tau g1 g2
            split_para[1]=str(alpha)
            new_para=split_para[0]
            for j in range(5):
                new_para+='_'+split_para[j+1]
            new_paras.append(new_para)
        json.dump(new_paras,open('AAAIELO/Avg/{}/{}/{}_best_gamma.json'.format(game,model,str(alpha)),'w'))

        print(new_paras)
        print('------------------------')
def tune_gamma(game,model):
    rep=5
    K=np.load("games/{}.npy".format(game)).shape[0]
    T=1000
    print(paras)
    print('------------------------')
    save_path='AAAIELO/Avg/'+game+'/'+model+'/'
    for alpha in parameters['alpha']:
        new_paras=[]
        for i in range(rep):
            seed=i+1
            best_per=-10000
            min_reg=float('Inf')
            best_para=''
            for lr in parameters['eta']:
                for C in parameters['C']:
                    y = []
                    tau = int(max(K, math.log(T)) * C)
                    for meta in parameters['meta']:
                        for delta in parameters['delta']:
                            per_sum = 0
                            reg_sum = 0
                            para_code = str(alpha) + '_' + str(lr) + '_' + str(tau) + '_' + str(
                                meta) + '_' + str(delta)

                            para_str = str(seed) + '_' + para_code
                            # para_list.append(para_str)
                            per, reg = get_performance(save_path + para_str, game)
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
            new_paras.append(best_para)
        json.dump(new_paras,open('AAAIELO/Avg/{}/{}/{}_tune_gamma.json'.format(game,model,str(alpha)),'w'))

        print(new_paras)
        print('------------------------')
def process_data(game,model,mode):
    path='AAAIELO/Avg/{}/{}/'.format(game,model)

    for alpha in parameters['alpha']:
        paras=json.load(open('AAAIELO/Avg/{}/{}/{}_{}_gamma.json'.format(game,model,str(alpha),mode),'r'))
        NDCG=[]
        Reg=[]
        Hit=[]
        for para in paras:
            rates=np.load(path+para+'_Rate.npy')
            reg = np.load(path + para + '_Reg.npy')
            Reg.append(reg[-3,:])
            ndcg=[]
            hit=[]
            for k in range(topk):
                ndcg.append(get_NDCG(rates,true_rating,k+1))
                hit.append((hit_Ratio(rates,true_rating,k+1)))
            NDCG.append(ndcg)
            Hit.append(hit)
        NDCG=np.array(NDCG)
        Reg=np.array(Reg)
        Hit=np.array(Hit)

        width=1.5
        if alpha==0.6 or alpha==1.0:
            width=2.0
        style='-'
        if alpha==0.6:
            style='--'
        if game=='Transitive game':
            width = 1.5
            style='-'
            if alpha==0.4:
                width=2.0
                style='--'
        picture(Hit,NDCG,label=r"$\gamma=$"+str(alpha),linestyle=style,linewidth=width)

    plt.subplot(1, 2, 1)

    plt.title(game, fontsize=fz)
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.xticks([0, 50, 100], ['1', '5', '10'], fontsize=fz)
    # plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.subplot(1, 2, 2)
    plt.grid()
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.xticks([0, 50, 100], ['1', '5', '10'], fontsize=fz)

    plt.tight_layout()

    plt.savefig(figpath + '{}_{}_{}.pdf'.format(game, mode,para_seed))
    plt.show()
if __name__=='__main__':

    alpha_set={}
    alpha_set['Elo game']=[0.4,0.5,0.8,1,1.2]
    alpha_set['Transitive game'] = [0.4,0.6,0.8,1,1.2]
    alpha_set['Triangular game'] = [0.4,0.6,0.8,1,1.2]
    alpha_set['Elo game + noise=0.1'] =[0.4,0.6,0.8,1,1.2]
    alpha_set['Elo game + noise=0.05'] = [0.4,0.6,0.8,1,1.2]
    alpha_set['Elo game + noise=0.01'] = [0.4,0.6,0.8,1,1.2]
    games = ['Elo game', 'Transitive game', 'Triangular game']
    games += ['Elo game + noise=0.1', 'Elo game + noise=0.05', 'Elo game + noise=0.01']
    model='MaxInELO'
    for game in games:
        _, axes = plt.subplots(1, 2, figsize=(5.0, 2.7))

        parameters['alpha'] =alpha_set[game]
        payoff = np.load("games/{}.npy".format(game))
        payoff = (payoff + 1.0) / 2.0

        true_rating = inverseP(payoff)
        initial()
        if para_seed=='max':
            path = 'AAAIELO/Avg/{}/{}/max_best_para.json'.format(game, model)
        else:
            path = 'AAAIELO/Avg/{}/{}/best_para.json'.format(game, model)
        paras = json.load(open(path, 'r'))
        print(paras)


        if mode == 'best':
            best_gamma(game, model, paras)

            process_data(game, model, mode)
        if mode=='tune':
            tune_gamma(game, model)

            process_data(game, model, mode)
