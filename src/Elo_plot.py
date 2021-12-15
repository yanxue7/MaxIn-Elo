
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
#random.seed(1)
#np.random.seed(1)

fz = 14
Name=''
path='finalplot/'
mode=sys.argv[1]
melo=int(sys.argv[2])

def smooth(scalar,weight=0.99):
    # data = pd.read_csv(filepath_or_buffer=csv_path,header=0,names=['Step','Value'],dtype={'Step':np.int,'Value':np.float})
    # scalar = data['Value'].values
    scalar=scalar.tolist()
    last = scalar[0]
    smoothed = []
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)
def initial():

    _, axes = plt.subplots(1,2,figsize=(5.0,2.7))#(5.0,2.7))#5.0, 2.7#figsize=

    # plt.subplot(121)
    #
    # plt.xlabel("rounds $t$", fontsize=fz)
    # plt.ylabel("top rank correct", fontsize=fz)
    # plt.xticks(fontsize=fz)
    # plt.yticks(fontsize=fz)
    # plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))

    plt.subplot(121)
    plt.xticks(fontsize=fz)
    plt.yticks(fontsize=fz)
    plt.xlabel("Rounds $t$", fontsize=fz)
    plt.ylabel("RR", fontsize=fz)
    # plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    # plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

    # plt.subplot(223)
    #
    # plt.xlabel("rounds $t$", fontsize=fz)
    # plt.ylabel("Weighted tau rank", fontsize=fz)
    # plt.xticks(fontsize=fz)
    # plt.yticks(fontsize=fz)

    # plt.subplot(132)
    #
    # plt.xlabel("Top $K$", fontsize=fz)
    # plt.ylabel(r"$\hat{P}$ error", fontsize=fz)
    # plt.xticks(fontsize=fz)
    # plt.yticks(fontsize=fz)
    plt.subplot(122)

    plt.xlabel("Rounds $t$", fontsize=fz)
    plt.ylabel("Regret", fontsize=fz)
    plt.xticks(fontsize=fz)
    plt.yticks(fontsize=fz)
    # plt.subplot(223)
    #
    # plt.xlabel("k", fontsize=fz)
    # plt.ylabel("NDCG@$k$", fontsize=fz)
    # plt.xticks(fontsize=fz)
    # plt.yticks(fontsize=fz)
    # plt.subplot(224)
    #
    # plt.xlabel("t", fontsize=fz)
    # plt.ylabel("NDCG@1", fontsize=fz)
    # plt.xticks(fontsize=fz)
    # plt.yticks(fontsize=fz)

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
def get_NDCG(pre_rating,true_rating,k):
    pre_rank=np.argsort(-pre_rating)
    DCGP=0
    for i in range(k):
        rel=true_rating[pre_rank[i]] # true rating of player with predicted rank i
        DCGP+=(np.power(2.0,rel)-1.0)/np.log2(i+1+1) # from 2
    true_rank=np.argsort(-true_rating)
    IDCGP = 0
    for i in range(k):
        rel = true_rating[true_rank[i]]  # true rating of player with predicted rank i
        IDCGP += (np.power(2.0, rel) - 1.0) / np.log2(i + 1 + 1)  # from 2
    return DCGP/IDCGP
def get_ins(pre_rating,true_rating,k):
    pre_rank=np.argsort(-pre_rating)
    true_rank=np.argsort(-true_rating)[:k]
    ins=0.0
    for i in range(k):
        if pre_rank[i] in true_rank:
            ins+=1
    return ins/k
# def top_distance(pre_rating,true_rating):
#     premax=np.
def picture(X,Reg,label):
    rep = np.shape(X)[0]

    k=np.shape(X)[-1]
    top1cor=X[:,0,:].reshape(rep,-1)
    NDCG=X[:,1,:].reshape(rep,-1)


    repr=Reg.shape[0]
    T=Reg.shape[-1]
    #T=1000

    Reg=np.reshape(Reg,(repr,-1,T))
    print(Reg.shape)

    top1dis = Reg[:, 0, 15:T]
    for i in range(rep):
        # top1cor[i]= smooth(top1cor[i])
        top1dis[i] = smooth(top1dis[i])

    regret=process(Reg[:,-2,:],Reg[:,-1,:])#Reg[:,-3,:]#
    regret=regret[:,:T]
    print(regret.shape)
    #top1dis=regret
    # plt.subplot(2, 2, 1)
    #
    # #plt.xlim(1, 51)
    #
    #
    # ST=0
    #
    # In=np.arange(np.shape(top1cor)[-1])+1
    #
    # se = scipy.stats.sem(top1cor, axis=0)
    #
    # plt.fill_between(In,top1cor.mean(0) - se, top1cor.mean(0) + se, alpha=0.2)
    # plt.plot(In,top1cor.mean(0), label=label)
    # plt.subplot(2, 2, 3)
    # se = scipy.stats.sem(fullrank, axis=0)
    #
    # plt.fill_between(In,fullrank.mean(0) - se, fullrank.mean(0) + se,
    #                  alpha=0.2)
    # plt.plot(In,fullrank.mean(0), label=label)
    # In=np.arange(np.shape(top1dis)[-1])
    #
    plt.subplot(1, 2, 2)
    #plt.xlim(1, 51)
    In=np.arange(np.shape(regret)[-1])

    se = scipy.stats.sem(regret, axis=0)

    plt.fill_between(In,regret.mean(0) - se, regret.mean(0) + se,alpha=0.2)
    plt.plot( In,regret.mean(0), label=label)

    # plt.subplot(2, 2, 3)
    #
    # In = np.arange(np.shape(NDCG)[-1])+1
    #
    # se = scipy.stats.sem(NDCG, axis=0)
    #
    # plt.fill_between(In, NDCG.mean(0) - se, NDCG.mean(0) + se, alpha=0.2)
    # plt.plot(In, NDCG.mean(0), label=label)

    plt.subplot(1, 2, 1)

    In = np.arange(np.shape(top1dis)[-1])+100

    se = scipy.stats.sem(top1dis, axis=0)

    plt.fill_between(In, top1dis.mean(0) - se, top1dis.mean(0) + se, alpha=0.2)
    plt.plot(In, top1dis.mean(0), label=label)
    # plt.subplot(1, 2, 2)
    # In = np.arange(np.shape(regret)[-1])
    # se = scipy.stats.sem(regret, axis=0)
    #
    # plt.fill_between(In,regret.mean(0) - se, regret.mean(0) + se,alpha=0.2)
    # plt.plot(In,regret.mean(0), label=label)

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

        games=['Elo game','Transitive game']#'Transitive game', 'Triangular game']
        games+=['Elo game + noise=0.1','Triangular game']#
        games+=['Elo game + noise=0.01', 'Elo game + noise=0.05']
    else:
        games=['Kuhn-poker']#['5,3-Blotto']#,'Random game of skill']
        #games = ['Kuhn-poker', 'Blotto', 'Disc game', 'tic_tac_toe', 'AlphaStar', 'hex(board_size=3)']
    #games=['Transitive game']
    #games = ['Kuhn-poker']#, 'AlphaStar', 'hex(board_size=3)', 'Blotto', 'Disc game', 'tic_tac_toe']
    #melo=0
    if not os.path.exists(path):
        os.mkdir(path)
    rep=5
    #mode='Avg'
    if melo==0:
        data_path='AAAIELO/'+'Avg/'
    else:
        data_path = 'AAAIMELO1/AAAIMELO/' + 'Avg/'
    #games = ['Elo game + noise=0.1','Elo game + noise=0.5','Elo game + noise=1.0']
    # games = ['Kuhn-poker', '10,3-Blotto', '5,3-Blotto', '3-move parity game 2']
    # games += ['10,4-Blotto', '10,5-Blotto', '5,4-Blotto', '5,5-Blotto', 'AlphaStar', 'Blotto', 'Disc game',
    #            'Elo game + noise=0.1', 'Elo game + noise=0.5', 'Elo game + noise=1.0', 'Elo game',
    #            'Normal Bernoulli game']
    # games += ['Random game of skill', 'Transitive game', 'Triangular game', 'connect_four',
    #            'go(board_size=3,komi=6.5)', 'go(board_size=4,komi=6.5)', 'hex(board_size=3)',
    #            'misere(game=tic_tac_toe())', 'quoridor(board_size=3)', 'quoridor(board_size=4)',
    #            'tic_tac_toe']  # ['Kuhn-poker']#['10,3-Blotto','5,3-Blotto','3-move parity game 2']#['Kuhn-poker']#['Elo game']['transitive']#
    # ['Kuhn-poker','10,3-Blotto','5,3-Blotto','3-move parity game 2']#['Kuhn-poker']#['Triangular game']#['Elo game']
    #games=['Kuhn-poker']#['10,4-Blotto','10,5-Blotto','5,4-Blotto','5,5-Blotto','AlphaStar','Blotto','Disc game','Elo game + noise=0.1','Elo game + noise=0.5','Elo game + noise=1.0','Elo game','Normal Bernoulli game','Kuhn-poker','10,3-Blotto','5,3-Blotto','3-move parity game 2']
    #gmaes=['RPS','Random game of skill','Transitive game','Triangular game','connect_four','go(board_size=3,komi=6.5)','go(board_size=4,komi=6.5)','hex(board_size=3)','misere(game=tic_tac_toe())','quoridor(board_size=3)','quoridor(board_size=4)','tic_tac_toe']#['Kuhn-poker']#['10,3-Blotto','5,3-Blotto','3-move parity game 2']#['Kuhn-poker']#['Elo game']['transitive']#
    for game_name in games:
        initial()
        Name=game_name
        payoff=np.load('games/{}.npy'.format(game_name))
        K=payoff.shape[0]
        T=1000
        # EloSGDobj = Rand(K, T, 1, (payoff + 1.0) / 2.0, melo)
        true_rate = logit(payoff) #get true melo ratings #np.mean((payoff+1.0)/2,axis=-1)
        names = ['random','RGUCB','DBGD','MaxInP','MaxInELO']  # 'netuneFIPUCBElo_d40' + '_k' + str(K)
        # save_rate_path='NDCG_test_elo/'
        # if game_name=='Elo game + noise=0.01' or game_name=='Elo game + noise=0.05':
        #     save_rate_path = 'myresultshat/'

        for name in names:
            rates,Reg=get_rates(data_path+game_name+'/'+name+'/')
            # if game_name=='Elo game' and name=='MaxInELO':
            #     paras = json.load(open('AAAIELO/Avg/test_para.json', 'r'))
            #     rates = []
            #     Reg = []
            #     for i in range(rep):
            #         para = paras[i]
            #         rates.append(np.load(data_path+game_name+'/'+name+'/' + para + '_Rate.npy'))
            #         Reg.append(np.load(data_path+game_name+'/'+name+'/' + para + '_Reg.npy'))
            #     rates = np.array(rates)
            #     Reg = np.array(Reg)
            #np.load(save_rate_path+'{}/{}rate.npy'.format(game_name,name))
            # repath = 'myresultselo3tranred'
            # Regname='Regret0.5.npy'
            # if name=='MaxInELO' or name=='MaxInP':
            #     Regname = 'Regret.npy'
            # Reg = np.load('{}/{}/{}/{}'.format(repath,game_name, name,Regname))

            #Reg=get_regs(save_rate_path+'{}/{}ndcgmax.npy'.format(game_name,name))
            # if name=='MaxInELO':
            #     rates = np.load('myresultshat/{}/{}rate2.npy'.format(game_name, name))
            #     Reg = np.load('myresultshat/{}/{}2.npy'.format(game_name, name))
            #     print(1111,Reg[:,1,-1])
            #CV=np.load('myresultshat/{}/{}CV.npy'.format(game_name,name))

            rep=rates.shape[0]
            Ins=[]
            Phat=[]
            NDCGS=[]
            for i in range(rep):
                # if name != 'MaxInP':
                #     EloSGDobj.C = CV[i]
                #     print(2222222, EloSGDobj.C.shape)
                # else:
                #     EloSGDobj.C=np.zeros([K,8])
                #     print(11111111,EloSGDobj.C.shape)

                    # print(np.argmax(true_rate))
                    # print(np.argmax(rates[i]))
                ins=[]
                phat=[]
                NDCG=[]
                for topk in range(1,20):
                    ins.append(get_ins(rates[i],true_rate,topk))
                    # phat.append(EloSGDobj.tran_best_index(rates[i],true_rate))#
                    NDCG.append(get_NDCG(rates[i],true_rate,topk))

                Ins.append(ins)
                # Phat.append(phat)
                NDCGS.append(NDCG)

            Ins=np.array(Ins).reshape((rep,1,-1)) #rep 1 10
            # Phat=np.array(Phat).reshape((rep,1,-1))
            NDCGS=np.array(NDCGS).reshape((rep,1,-1))
            res = np.concatenate((Ins, NDCGS), axis=1)  # rep 2 10
            #res=np.concatenate((Ins,NDCGS),axis=1) #rep 3 10
            picture(res,Reg,label=name)
        plt.subplot(1, 2, 1)

        plt.title(game_name, fontsize=fz)
        plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        # plt.yticks([0, 0.5, 1.0])
        # plt.xticks([0, 50, 100], ['1', '5', '10'], fontsize=fz)
        #plt.legend()
        plt.grid()
        plt.tight_layout()

        plt.subplot(1, 2, 2)
        plt.grid()
        plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        plt.tight_layout()
        # plt.subplot(1, 2, 3)
        # plt.grid()
        # plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        # plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        # plt.tight_layout()

        # plt.subplot(2, 2, 3)
        # plt.grid()
        # plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        # plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        # plt.legend()
        #
        # plt.tight_layout()
        # plt.subplot(2, 2, 4)
        # plt.grid()
        # plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        # plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        # plt.tight_layout()

        plt.savefig(path+'{}_{}.pdf'.format(game_name,mode))
        plt.show()
        #
        #
        # drawevery(game_name,name, rep=5)
    # for game_name in games:
    #
    #     drawevery(game_name, 'MaxInElo', rep=3)

#python Elo_plot.py Max