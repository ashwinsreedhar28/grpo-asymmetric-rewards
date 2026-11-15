import numpy as np
from scipy.special import gammaln
def comps_arrays(G):
    K,M=np.meshgrid(np.arange(G+1),np.arange(G+1),indexing="ij"); ok=K+M<=G
    K,M=K[ok].astype(float),M[ok].astype(float); Z=G-K-M
    lc=gammaln(G+1)-gammaln(K+1)-gammaln(M+1)-gammaln(Z+1)
    return K,M,Z,lc
def grpo_drift(p,pi,G,c):
    """exact E[g] for group-std GRPO, vectorized over pi array"""
    K,M,Z,lc=comps_arrays(G)
    s2=(K+c*c*M)/G-((K-c*M)/G)**2; ok=s2>1e-12
    w=np.where(ok,(K-c*M)*Z/np.sqrt(np.where(ok,s2,1)),0)
    pi=np.atleast_1d(pi)[:,None]
    with np.errstate(divide="ignore",invalid="ignore"):
        lp=lc+K*np.log(pi*p)+M*np.log(pi*(1-p))+Z*np.log(1-pi)
    lp=np.where(np.isnan(lp),-np.inf,lp)
    return (np.exp(lp)*w).sum(1)/G**2
def equilibrium(p,c,G,npi=400):
    """stable answer rate predicted from the sign of the exact drift; None for rules with constant threshold"""
    pis=np.linspace(1e-3,1-1e-3,npi)
    d=grpo_drift(p,pis,G,c)
    if np.all(d>0): return 1.0
    if np.all(d<0): return 0.0
    idx=np.where((d[:-1]>0)&(d[1:]<=0))[0]
    if len(idx)==0:
        return 1.0 if d[-1]>0 else 0.0
    i=idx[0]; return pis[i]+(pis[i+1]-pis[i])*d[i]/(d[i]-d[i+1])
def predict(rule,p,c,B=4,G=8):
    if rule=="grpo": return equilibrium(p,c,G)
    if rule=="batch": return equilibrium(p,c,B*G)
    return 1.0 if p>c/(1+c) else 0.0
if __name__=="__main__":
    for p in [0.6,0.7,0.8]:
        print(p, predict("grpo",p,5), predict("batch",p,5))
    # check for multiple sign changes anywhere on the grid
    multi=0
    for G in [8,32]:
        for c in range(1,11):
            for p in np.arange(0.40,0.99,0.02):
                d=grpo_drift(p,np.linspace(1e-3,1-1e-3,400),G,c)
                multi+= (np.sum(np.diff(np.sign(d))!=0)>1)
    print("cells with >1 sign change:",multi)
