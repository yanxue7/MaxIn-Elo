import numpy as np
import time
import random
from scipy import stats
from common_funcs import get_NDCG, inverseP,RR

class Rand():
    def __init__(self,K,T,iter,payoff,melo,dim=8):
        self.K=K
        self.T=T
        self.iter=iter


        self.alpha = 1
        self.r=np.zeros([self.K])
        self.s=np.zeros([self.K])
        self.num=np.zeros([self.K])
        self.eta=1

        self.P=payoff
        self.true_theta = inverseP(self.P)
        self.adver=False
        self.ranking=self.get_ranking(self.true_theta)
        self.gamma=np.zeros([self.K])+1
        self.W=np.zeros([self.K])
        self.pairwiseh=np.zeros([self.K,self.K])
        self.optimal=np.max(self.true_theta)
        self.MMiter=10
        self.melo=melo
        self.dim=dim
        if melo==0:
            self.C=np.zeros((self.K,8),dtype=np.float32)
        else:
            self.C = np.ones((self.K, 8), dtype=np.float64)
    def initial(self):
        if self.melo==0:
            self.C=np.zeros((self.K,8),dtype=np.float32)
        else:
            self.C =  np.random.normal(loc=0.0,scale=1.0/8,size=(self.K,8))
        self.r=np.zeros([self.K])

    def f(self, x):
        if x >= 0:
            return 1.0 / (1 + np.exp(-x))
        else:
            return np.exp(x) / (1 + np.exp(x))

    def cal_Csum(self,i,j):
        sum=0.0
        for k in range(self.dim//2):
            sum+=self.C[i][k*2]*self.C[j][k*2+1]-self.C[j][k*2]*self.C[i][k*2+1]
        return sum
    def sampling(self,alpha,eta,tau,meta,delta,save_rate=1):
        self.initial()
        self.eta=eta
        self.meta=eta

        Armx=[]
        Army=[]
        NDCG3=[]
        NDCG5=[]
        All_ratings=[]

        top1cor=[]
        top1dis=[]
        ktau=[]

        reg=[]
        regret=np.zeros([self.T])
        K_list=np.arange(self.K)
        pairs=[]
        for i in range(self.K):
            for j in range(i):
               pairs.append((i,j))
        start=time.clock()
        for t in range(self.T):
            xt,yt=random.choice(pairs)
            Armx.append(xt)
            Army.append(yt)
            regret[t]=regret[t-1]+(self.optimal-0.5*(self.true_theta[xt]+self.true_theta[yt]))
            reg.append(regret[t])
            ot=np.random.binomial(1,p=self.P[xt][yt])
            Csum=self.cal_Csum(xt,yt)
            delta=ot-self.f(self.r[xt]-self.r[yt]+Csum)

            xt_new=self.r[xt]+self.eta*delta
            yt_new=self.r[yt]-self.eta*delta
            self.r[xt]=xt_new
            self.r[yt]=yt_new
            if self.dim>0:
                for i in range(self.dim//2):
                    i0 = self.meta * delta * self.C[yt][2*i+1]
                    i1 = -self.meta * delta * self.C[yt][2*i+0]
                    j0 = -self.meta * delta * self.C[xt][2*i+1]
                    j1 = self.meta * delta * self.C[xt][2*i+0]
                    self.C[xt][2*i+0] += i0
                    self.C[xt][2*i+1] += i1
                    self.C[yt][2*i+0] += j0
                    self.C[yt][2*i+1] += j1




            if (t+1)%self.iter==0:
                All_ratings.append(self.r)
                top1cor.append(RR(self.r, self.P,self.melo, 1))


        endtime=time.clock()

        if save_rate==1:
            return self.r,self.C,[top1cor,reg,Armx,Army],np.array(All_ratings)
        else:
            return [top1cor,reg,Armx,Army]

    def gupdate(self,x):

        w=self.W[x]
        if w==0:
            return 0
        CE=0
        for i in range(self.K):
            CE+=1.0*self.pairwiseh[x][i]/(self.gamma[x]+self.gamma[i]+1e-8)
        return 1.0*w/CE
    def query(self,x,y,t):

        if self.adver==False:
            return np.random.binomial(1,p=self.P[x][y])





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

