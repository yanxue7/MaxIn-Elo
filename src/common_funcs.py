import numpy as np
from random import shuffle
def get_NDCG(pre_rating,true_rating,k):
    true_rank = np.argsort(-true_rating)
    labeln=k#k#10

    K=pre_rating.shape[0]
    label=np.zeros(K)

    for i in range(labeln):
        label[true_rank[i]]=1
    for i in range(labeln,K):
        if true_rating[true_rank[i]]==true_rating[true_rank[i-1]]:
            label[true_rank[i]]=1
        else:
            break
    pre_rank = np.argsort(-pre_rating)
    DCGP = 0
    for i in range(k):
        rel=label[pre_rank[i]] # true rating of player with predicted rank i
        DCGP+=(np.power(2.0,rel)-1.0)/np.log2(i+1+1) # from 2
    IDCGP = 0
    for i in range(k):
        rel = label[true_rank[i]]  # true rating of player with predicted rank i
        IDCGP += (np.power(2.0, rel) - 1.0) / np.log2(i + 1 + 1)  # from 2
    return DCGP / IDCGP

def hit_Ratio(pre_rating,true_rating,k):
    n=pre_rating.shape[0]
    pre_rank = np.argsort(-pre_rating)
    true_rank=np.argsort(-true_rating)
    pre_set=pre_rank[:k].tolist()
    true_set=true_rank[:k].tolist()
    for i in range(k,n,1):
        now=true_rank[i]
        last=true_rank[i-1]
        if true_rating[now]==true_rating[last]:
            true_set.append(now)
        else:
            break
    for i in range(k,n,1):
        now=pre_rank[i]
        last=pre_rank[i-1]
        if pre_rating[now]==pre_rating[last]:
            pre_set.append(now)
        else:
            break
    return len(list(set(true_set)&set(pre_set)))/k#len(true_set)

def RR(pre_rating,P,melo,k=1):
    n=pre_rating.shape[0]
    if melo==0:
        borda=np.mean(P,axis=-1)
    else:
        borda=inverseP(P)

    largest_score = np.max(borda)
    best_indexs = np.argwhere(borda == largest_score).reshape(-1).tolist()
    sort_pre=np.sort(-pre_rating)

    sum=-1
    num=0
    for index in best_indexs:
        for i in range(n):
            if sort_pre[i]==-pre_rating[index]:
                sum=max(1.0/(i+1),sum)
                num+=1
    return sum
def MRR(pre_rating,true_rating,k):
    n=pre_rating.shape[0]
    true_rank = np.argsort(-true_rating)
    sort_pre=np.sort(-pre_rating)
    sum=0
    for l in range(k):
        opt_index=true_rank[l]
        for i in range(n):
            if sort_pre[i]==-pre_rating[opt_index]:
                sum+= 1.0/(i+1)
                break
    return sum/k
def inverseP(P):
    esp = 1e-2
    n = P.shape[0]
    for i in range(n):
        for j in range(n):
            if P[i, j] == 0:
                P[i, j] += esp
            if P[i, j] == 1:
                P[i, j] -= esp
    A = np.log(P) - np.log(1.0 - P)
    Borda = np.mean(A, axis=-1)
    return Borda
def change_label(label):
    if label=='random':
        label='Random'
    if label=='RGUCB':
        label='RG-UCB'
    if label=='MaxInELO':
        label='MaxIn-Elo'
    return label
def smooth(scalar,weight=0.99):

    scalar=scalar.tolist()
    last = scalar[0]
    smoothed = []
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)
