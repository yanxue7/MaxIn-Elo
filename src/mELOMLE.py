import numpy as np
import random
from sklearn.linear_model import LogisticRegression
import math
import time
import random
from scipy import stats
from common_funcs import get_NDCG, inverseP,RR
import os

class EloSMLE():
    def __init__(self,K,T,iter,payoff,melo):
        self.K=K
        self.T=T
        self.iter=iter
        self.tau=100
        self.alpha_ts=0.1
        self.r=np.zeros([self.K])
        self.melo=melo
        self.eta=0.1
        self.adver=False
        self.theta=np.zeros([self.T//self.tau+1,self.K])
        self.var=0.001

        self.W=np.zeros([self.K])
        self.pairwiseh = np.zeros([self.K, self.K])
        self.wtime=np.zeros([self.K,self.K])
        self.ltime=np.zeros([self.K,self.K])
        self.gamma=np.zeros([self.K])+1
        self.MMiter=100

        self.GD_iter=1
        self.etagd=0.01

        self.X = np.eye(self.K)
        self.d=self.K

        self.P = payoff
        self.true_theta = inverseP(self.P)
        self.ranking=self.get_ranking(self.true_theta)

        self.optimal=np.max(self.true_theta)
        np.set_printoptions(precision=3, suppress=True)

    def f(self, x):
        if x >= 0:
            return 1.0 / (1 + np.exp(-x))
        else:
            return np.exp(x) / (1 + np.exp(x))

    def grad(self, x, y, theta, lamda=0):
        return self.alpha*x * (-y + 1 / (1 + np.exp(-x.dot(theta)))) + 2 * lamda * theta

    def sampling(self,alpha,eta,tau,g1,g2,save_rate=1,pr=0):


        Armx=[]
        Army=[]
        NDCG3=[]
        NDCG5=[]
        All_ratings=[]
        regret=np.zeros([self.T])
        self.alpha=1

        self.tau=tau
        self.g1=g1
        self.g2=g2
        self.CT=1

        eps=10**(-6)
        self.Vnorm= np.identity(self.d) * eps

        top1cor=[]
        top1dis=[]
        ktau=[]
        Reg=[]
        start=time.time()



        pairs = []
        for i in range(self.K):
            for j in range(i):
                pairs.append((i, j))
        history=[]
        y = np.array([])

        y = y.astype('int')
        x = np.empty([0, self.d])
        t=0

        for t in range(self.tau):
            self.reward=self.true_theta
            xt, yt = random.choice(pairs)
            ot = np.random.binomial(1, p=self.P[xt][yt])
            history.append((xt, yt, ot))
            x=np.concatenate((x,[self.X[xt]-self.X[yt]]),axis=0)
            y=np.concatenate((y,[ot]),axis=0)
            regret[t]=regret[t-1]+self.calelo_reg(xt,yt)
            top1cor.append(0)
            top1dis.append(self.K)
            ktau.append(0)
            Reg.append(regret[t])
            NDCG3.append(0)
            NDCG5.append(5)
            All_ratings.append(np.zeros(self.K))
            self.Vnorm+=np.outer(self.X[xt]-self.X[yt], self.X[xt]-self.X[yt])
            if pr==1:
                print(xt,yt)
            Armx.append(xt)
            Army.append(yt)
        if y[0] == y[1]:
            y[1] = 1-y[0]
        x=x*self.alpha

        B_inv=np.linalg.inv(self.Vnorm)
        for t in range(self.tau, self.T):
            self.reward = self.true_theta
            vr=alpha

            clf = LogisticRegression(penalty='none', fit_intercept=False, solver='lbfgs').fit(x, y)
            theta_hat = clf.coef_[0]
            theta_bar = theta_hat / (3 * np.max(np.abs(theta_hat)))#normlize
            UCB= np.zeros((self.K,self.K))
            for i in range(self.K):
                for j in range(self.K):
                    UCB[i,j]=vr*np.sqrt(B_inv[i,i]-B_inv[i,j]-B_inv[j,i]+B_inv[j,j])
            C=[]

            for i in range(self.K):
                flag=1
                for j in range(self.K):
                    if i!=j:
                        h=theta_bar[i]-theta_bar[j]+UCB[i,j]
                        if h<0:
                            flag=0
                            break
                if flag==1:
                    C.append(i)
            pairs=None
            maxun=-10000

            if pr==1 and t%100==0:
                print(C)
            lenc=len(C)
            for i in range(lenc):
                for j in range(i):
                    if UCB[C[i],C[j]]>maxun:
                        maxun=UCB[C[i],C[j]]
                        pairs=(C[i],C[j])
            if lenc==1:
                pairs=(C[0],C[0])

            xt=pairs[0]
            yt=pairs[1]
            Armx.append(xt)
            Army.append(yt)
            if pr==1:
                print(xt,yt)
            ot = np.random.binomial(1, p=self.P[xt,yt])

            tmp = np.zeros([self.K])
            for i in range(self.K):
                tmp[i] = B_inv[i][xt] - B_inv[i][yt]
            fdt = tmp[xt] - tmp[yt]
            B_inv -= np.outer(tmp, tmp) / (1 + fdt)

            x = np.concatenate((x, [self.X[xt] - self.X[yt]]), axis=0)
            y = np.concatenate((y, [ot]), axis=0)
            if t == 0:
                regret[t] = self.calelo_reg(xt, yt)
            else:
                regret[t] = regret[t - 1] + self.calelo_reg(xt, yt)


            if   ((t+1)%self.iter==0 or t==self.T):
                j = t // self.tau
                All_ratings.append(theta_bar)
                top1cor.append(RR(theta_bar, self.P,self.melo, 1))

                Reg.append(regret[t])



        endtime=time.time()
        if save_rate==1:
            return theta_bar,0,[top1cor,Reg,Armx,Army],np.array(All_ratings)
        else:
            return [top1cor,NDCG3,NDCG5,Reg,Armx,Army]

    def gupdate(self,x):

        w=self.W[x]
        if w==0:
            return 0
        CE=0
        for i in range(self.K):
            CE+=1.0*self.pairwiseh[x][i]/(self.gamma[x]+self.gamma[i]+1e-8)
        return 1.0*w/CE


    def predict_p(self,ri, rj):
        return 1.0 / (1.0 + np.exp(- self.alpha*(ri - rj)))

    def calelo_reg(self,x,y):
        tru=self.optimal
        rx=self.reward[x]
        ry=self.reward[y]


        return 0.5*((tru-rx)
                    +(tru-ry))





    def get_ranking(self,Borda):
        Bordar=np.sort(-Borda)
        Ranking=[]
        for i in range(self.K):
            p = -Borda[i]
            for j in range(self.K):
                if Bordar[j] == p:
                    Ranking.append(j)
                    break
        Ranking = np.array(Ranking)
        return Ranking

