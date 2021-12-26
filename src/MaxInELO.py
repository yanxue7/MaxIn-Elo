import numpy as np
import random
from sklearn.linear_model import LogisticRegression
import math
import time
import random

from common_funcs import get_NDCG, inverseP,RR
from scipy import stats
import os

class MaxInELO():
    def __init__(self,K,T,iter,payoff,melo,dim=8):
        self.K=K
        self.T=T
        self.iter=iter
        self.tau=100
        self.alpha_ts=0.1
        self.r=np.zeros([self.K])

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
        self.dim=dim

        self.X = np.eye(self.K)
        self.d=self.K

        self.melo=melo
        self.P=payoff #winning probability
        self.true_theta=inverseP(self.P)
        self.ranking=self.get_ranking(self.true_theta)
        self.optimal=np.max(self.true_theta)
        if melo==1 and self.dim>0:
            self.MC = np.ones([self.K, dim], dtype=np.float64)/self.dim
        else:
            self.dim=0
            self.MC=np.zeros([self.K,self.dim],dtype=np.float64)

        np.set_printoptions(precision=3, suppress=True)

    def f(self,x):
        if x >= 0:
            return 1.0 / (1 + np.exp(-x))
        else:
            return np.exp(x) / (1 + np.exp(x))


    def grad(self, x, y, theta, lamda=0):
        return self.alpha*x * (-y + 1 / (1 + np.exp(-x.dot(theta)))) + 2 * lamda * theta
    def calP(self,xt,yt,t):

        return self.P[xt][yt]
    def cal_Csum(self,i,j):
        sum=0.0
        for k in range(self.dim//2):
            sum+=self.MC[i][k*2]*self.MC[j][k*2+1]-self.MC[j][k*2]*self.MC[i][k*2+1]
        return sum
    def initial(self):
        if self.melo == 1:
            if self.dim>0:
                self.MC = np.random.normal(loc=0.0,scale=1.0/8,size=(self.K,self.dim))
        else:
            self.MC = np.zeros([self.K, self.dim], dtype=np.float64)

    def sampling(self,alpha,eta,tau,g1,g2,save_rate=1,pr=0):
        self.initial()
        Armx=[]
        Army=[]
        regret=np.zeros([self.T])
        self.V=np.ones([self.d])
        self.alpha=1
        self.meta=g1
        vr=alpha #balance parameter
        self.eta=eta # learning rate of elo
        self.tau=tau # initial batch size
        self.g2=g2 # normlize MLE of first batch size

        self.CT=1
        # initialize historical matrix V
        eps=10**(-6)
        self.Vnorm= np.identity(self.d) * eps
        #  define list to save result
        top1cor=[]
        top1dis=[]
        ktau=[]
        Reg=[]
        NDCG3=[]
        NDCG5=[]
        All_ratings=[]
        start=time.time()
        # save all pairs
        self.pairs = []
        for i in range(self.K):
            for j in range(i):
                self.pairs.append((i, j))

        history=[]

        # calculate MLE result of the first batch
        y = np.array([])
        y = y.astype('int') #Y label save pairwise comparison result
        x = np.empty([0, self.d]) # X label

        for t in range(self.tau):
            All_ratings.append(np.zeros(self.K))
            self.reward=self.true_theta
            xt, yt = random.choice(self.pairs)
            ot = np.random.binomial(1, p=self.calP(xt,yt,t))
            history.append((xt, yt, ot))
            x=np.concatenate((x,[self.X[xt]-self.X[yt]]),axis=0)
            y=np.concatenate((y,[ot]),axis=0)
            self.V[xt]+=1
            self.V[yt]+=1
            regret[t]=regret[t-1]+self.calelo_reg(xt,yt)
            top1cor.append(0)
            top1dis.append(self.K)
            ktau.append(0)
            NDCG3.append(0)
            NDCG5.append(0)
            Reg.append(regret[t])
            self.Vnorm+=np.outer(self.X[xt]-self.X[yt], self.X[xt]-self.X[yt])
            Armx.append(xt)
            Army.append(yt)
            if pr==1:
                print(xt,yt)

        if y[0] == y[1]:
            y[1] = 1-y[0]
        x=x*self.alpha
        if self.tau>1:
            clf = LogisticRegression(penalty='none', fit_intercept=False, solver='lbfgs').fit(x, y)
            theta_hat = clf.coef_[0]
            theta_hat= theta_hat/(self.g2*np.max(np.abs(theta_hat)))         #get MLE result
        else:
            theta_hat = np.zeros(self.d)
        # clf = LogisticRegression(penalty='none', fit_intercept=False, solver='lbfgs').fit(x, y)
        # theta_hat = clf.coef_[0]
        # theta_hat= theta_hat/(self.g2*np.max(np.abs(theta_hat)))         #get MLE result

        x2=np.linalg.norm(x=theta_hat, ord=2)

        # SGD stage
        grad = np.zeros(self.d)
        gradC=np.zeros((self.d,self.dim))
        theta_tilde = np.zeros(self.d)
        theta_tilde[:] = theta_hat[:]
        theta_bar = np.zeros(self.d)

        C_hat=self.MC
        if pr==1:
            print(np.sort(theta_hat))

            print(np.argsort(theta_hat))

        B_inv=np.linalg.inv(self.Vnorm)# inverse
        for t in range(self.tau, self.T):
            self.reward = self.true_theta
            if t % self.tau == 0:
                j = t//self.tau
                vr=alpha#/j
                eta = self.eta/ j
                meta=self.meta
                theta_tilde += eta * grad
                distance = np.linalg.norm(theta_tilde - theta_hat)
                if distance > 2:
                    theta_tilde = theta_hat + 2 * (theta_tilde - theta_hat) / distance
                self.MC+=eta*gradC



                grad = np.zeros(self.d)
                gradC = np.zeros((self.d, self.dim))
                theta_bar = (theta_bar * (j - 1) + theta_tilde) / j



            UCB= np.zeros((self.K,self.K))
            for i in range(self.K):
                for j in range(self.K):
                    UCB[i,j]=vr*np.sqrt(B_inv[i,i]-B_inv[i,j]-B_inv[j,i]+B_inv[j,j])
            C=[]
            for i in range(self.K):
                flag=1
                for j in range(self.K):
                    if i!=j:
                        MCsum=self.cal_Csum(i,j)
                        h=theta_bar[i]-theta_bar[j]+UCB[i,j]+MCsum
                        if h<0:
                            flag=0
                            break
                if flag==1:
                    C.append(i)
            pairs=None
            maxun=-1000



            lenc=len(C)

            for i in range(lenc):
                for j in range(i):
                    if UCB[C[i],C[j]]>maxun:
                        maxun=UCB[C[i],C[j]]
                        pairs=(C[i],C[j])
            if lenc==1:
                pairs=(C[0],C[0])
            if pairs==None:
                pairs=random.choice(self.pairs)
            xt=pairs[0]


            yt=pairs[1]
            Armx.append(xt)
            Army.append(yt)

            ot = np.random.binomial(1, p=self.calP(xt,yt,t))

            tmp=np.zeros([self.K])
            for i in range(self.K):
                tmp[i]=B_inv[i][xt]-B_inv[i][yt]
            fdt=tmp[xt]-tmp[yt]#feature.dot(tmp)
            B_inv -= np.outer(tmp, tmp) / (1 + fdt)

            MCsum = self.cal_Csum(xt, yt)
            delta = ot - self.f(self.r[xt] - self.r[yt] +MCsum)
            grad[xt]+=delta
            grad[yt]+=-delta
            if self.dim>0:
                for i in range(self.dim//2):
                    i0 = delta * self.MC[yt][2*i+1]
                    i1 = - delta * self.MC[yt][2*i+0]
                    j0 = - delta * self.MC[xt][2*i+1]
                    j1 = delta * self.MC[xt][2*i+0]
                    gradC[xt][2*i+0] += i0
                    gradC[xt][2*i+1] += i1
                    gradC[yt][2*i+0] += j0
                    gradC[yt][2*i+1] += j1

            if t == 0:
                regret[t] = self.calelo_reg(xt, yt)
            else:

                regret[t] = regret[t - 1] + self.calelo_reg(xt, yt)


            if   ((t+1)%self.iter==0 or t==self.T):
                j = t // self.tau
                All_ratings.append(theta_bar)

                top1cor.append(RR(theta_bar,self.P,self.melo,1))

                Reg.append(regret[t])



        endtime=time.time()
        print("end random sampling",endtime-start)
        if save_rate==1:
            return theta_bar,self.MC,[top1cor,Reg,Armx,Army],np.array(All_ratings)
        else:
            return [top1cor,NDCG3,NDCG5,Reg,Armx,Army]




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



