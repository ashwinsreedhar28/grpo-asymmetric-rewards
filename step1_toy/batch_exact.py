import numpy as np
from scipy.special import gammaln
from scipy.optimize import brentq
from grpo_exact import thresh

def multi_pmf(n,q):
    """pmf over (K,M) for n trials, probs q=(qc,qw,q0); returns arrays K,M,P"""
    K,M=np.meshgrid(np.arange(n+1),np.arange(n+1),indexing="ij")
    Z=n-K-M; ok=Z>=0
    K,M,Z=K[ok],M[ok],Z[ok]
    with np.errstate(divide="ignore"):
        lq=np.log(np.array(q))
    lp=gammaln(n+1)-gammaln(K+1)-gammaln(M+1)-gammaln(Z+1)
    lp=lp+np.where(K>0,K*lq[0],0)+np.where(M>0,M*lq[1],0)+np.where(Z>0,Z*lq[2],0)
    return K,M,np.exp(lp)

def drift_batch(p,pi,G,B,c):
    q=(pi*p,pi*(1-p),1-pi); N=B*G
    k1,m1,P1=multi_pmf(G,q); n01=G-k1-m1
    if B>1: K2,M2,P2=multi_pmf((B-1)*G,q)
    else: K2,M2,P2=np.array([0]),np.array([0]),np.array([1.0])
    K=k1[:,None]+K2[None,:]; M=m1[:,None]+M2[None,:]
    S2=(K+c*c*M)/N-((K-c*M)/N)**2
    S=np.sqrt(np.clip(S2,0,None))
    num=((k1-c*m1)*n01)[:,None]
    term=np.where(S>1e-12,num/np.where(S>1e-12,S,1),0.0)
    return (P1[:,None]*P2[None,:]*term).sum()/G**2

def thresh_batch(pi,G,B,c):
    return brentq(lambda p: drift_batch(p,pi,G,B,c),1e-6,1-1e-6,xtol=1e-10)

if __name__=="__main__":
    G,c=8,5
    print("B=1 check:",thresh_batch(0.5,G,1,c),thresh(0.5,G,c))
    for pi in [0.1,0.5,0.9]:
        for B in [1,2,4,8,16,32,64]:
            t=thresh_batch(pi,G,B,c); print(f"pi={pi} B={B}: p*={t:.4f} c_eff={t/(1-t):.3f}")
