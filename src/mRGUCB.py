import numpy as np
import time
import random
#random.seed(1)
#np.random.seed(1)
from scipy import stats
from common_funcs import get_NDCG, inverseP,RR

class RGUCB():
    def __init__(self,K,T,iter,payoff,melo,dim=8):
        self.K=K
        self.T=T
        self.iter=iter


        self.alpha = 1#0.01#np.log(10) / 400
        self.r=np.zeros([self.K])
        self.s=np.zeros([self.K])
        self.num=np.zeros([self.K])
        self.eta=1

        self.P=payoff#self.get_winning()
        self.true_theta = inverseP(self.P)#[i for i in range(0, 10 * self.K, 10)]
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
            self.C=np.zeros([self.K,8],dtype=np.float32)
        else:
            self.C = np.ones([self.K, 8], dtype=np.float64)
        self.count = np.zeros([self.K, self.K])
        self.avg = np.zeros([self.K, self.K])
    def initial(self):
        if self.melo==0:
            self.C=np.zeros((self.K,8),dtype=np.float32)#np.random.random((self.K,2))
        else:
            self.C =  np.random.normal(loc=0.0,scale=1.0/8,size=(self.K,8))# np.random.standard_normal(size=(self.K,8))#np.ones((self.K, 8), dtype=np.float64)
        self.r=np.zeros([self.K])
        self.count = np.zeros([self.K, self.K])
        self.avg = np.zeros([self.K, self.K])

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
    def removepair(self,x,y):
        bm = self.avg[x,y]
        bc = self.count[x,y]
        nm = self.avg[y,x]
        nc = self.count[x,y]
        if bc == 0 or nc == 0:
            return
            # We have no observed evaluations, CI overlaps
        bi = np.sqrt((np.log(2 / self.delta) * (1 ** 2)) / (2 * bc))
        base_lower = bm - bi
        base_upper = bm + bi
        ni = np.sqrt((np.log(2 / self.delta) * (1 ** 2)) / (2 * nc))
        new_lower = nm - ni
        new_upper = nm + ni

        if bm >= nm and new_upper > base_lower:
            pass
        elif bm < nm and base_upper > new_lower:
            pass
        else:
            # Remove the pair
            self.pairs.remove((x,y))
            self.pairs.remove((y, x))

    def sampling(self,alpha,eta,tau,meta,delta,save_rate=1):#(self,alpha,eta,tau,meta,delta)
        self.initial()
        self.eta=eta
        self.meta=eta
        self.delta=delta
        #self.C*=meta

        # print(self.delta)
        Armx=[]
        Army=[]
        NDCG3=[]
        NDCG5=[]
        # print("random:",np.random.random(1))
        print("start random sampling")
        top1cor=[]
        top1dis=[]
        ktau=[]
        All_ratings=[]
        reg=[]
        regret=np.zeros([self.T])
        K_list=np.arange(self.K)
        self.pairs = []
        for i in range(self.K):
            for j in range(i):
                if i != j:
                    self.pairs.append((i, j))
                    self.pairs.append((j, i))

        start=time.clock()
        for t in range(self.T):
            if len(self.pairs)==0:
                regret[t] = regret[t - 1]
                reg.append(regret[t])
                All_ratings.append(self.r)
                top1cor.append(RR(self.r, self.P, self.melo, 1))
                Armx.append(-1)
                Army.append(-1)
                continue
            xt,yt=random.choice(self.pairs)
            Armx.append(xt)
            Army.append(yt)

            regret[t]=regret[t-1]+(self.optimal-0.5*(self.true_theta[xt]+self.true_theta[yt]))
            reg.append(regret[t])
            ot=np.random.binomial(1,p=self.P[xt][yt])
            Csum=self.cal_Csum(xt,yt)
            delta=ot-self.f(self.r[xt]-self.r[yt]+Csum)

            xt_new=self.r[xt]+self.eta*delta
            yt_new=self.r[yt]-self.eta*delta#*self.alpha*(1-ot-self.f(self.r[yt]-self.r[xt]))
            self.r[xt]=xt_new
            self.r[yt]=yt_new
            if self.dim > 0:
                for i in range(self.dim // 2):
                    i0 = self.meta * delta * self.C[yt][2 * i + 1]
                    i1 = -self.meta * delta * self.C[yt][2 * i + 0]
                    j0 = -self.meta * delta * self.C[xt][2 * i + 1]
                    j1 = self.meta * delta * self.C[xt][2 * i + 0]
                    self.C[xt][2 * i + 0] += i0
                    self.C[xt][2 * i + 1] += i1
                    self.C[yt][2 * i + 0] += j0
                    self.C[yt][2 * i + 1] += j1

            self.count[xt, yt] += 1
            self.count[yt, xt] += 1
            N = self.count[xt, yt]
            self.avg[xt, yt] = 1.0 * ((N - 1) * self.avg[xt, yt] + ot) / N
            self.avg[yt, xt] = 1.0 * ((N - 1) * self.avg[yt, xt] + 1 - ot) / N

            self.removepair(xt, yt)

            if (t+1)%self.iter==0:
                All_ratings.append(self.r)
                top1cor.append(RR(self.r, self.P,self.melo, 1))

        endtime=time.clock()
        print("end random sampling",endtime-start)
        print(regret[-1])
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
        #random.seed(self.seed)

        if self.adver==False:
            return np.random.binomial(1,p=self.P[x][y])

    def tran_best_correct(self,pre,true,K):
        return self.Perror(pre,K)


    def tran_best_index(self,pre,true):

        pre_best = np.max(pre)
        pre_best_indexs = np.argwhere(pre == pre_best).reshape(-1).tolist()
        x=[]
        for index in pre_best_indexs:
            x.append(self.ranking[index])
        x=np.array(x)
        return x.mean()

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

    def Perror(self, pre, K):
        F_error = 0
        pre_P = np.zeros([self.K, self.K])
        true_rank = np.argsort(-self.true_theta)[0:K]
        for l in range(K):
            i = true_rank[l]
            for g in range(K):
                j = true_rank[g]
                if self.melo == 0:
                    pre_P[i, j] = self.f(pre[i] - pre[j])
                else:
                    pre_P[i, j] = self.f(
                        pre[i] - pre[j] + self.C[i][0] * self.C[j][1] - self.C[j][0] * self.C[i][1])
                F_error += np.square(pre_P[i, j] - self.P[i, j])
        F_error = np.sqrt(F_error)
        return F_error

    def total_rank(self, pre, true, K):
        true_top = np.argsort(-true)[0:K]
        pre_top = np.argsort(-pre)[0:K]

        interlist = list(set(true_top) & set(pre_top))

        # tau, p_value = stats.weightedtau(true, pre)
        return len(interlist)
