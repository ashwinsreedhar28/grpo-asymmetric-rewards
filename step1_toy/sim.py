import numpy as np

RULES=["grpo","dr","rloo","batch","fixed"]

def advantages(r,rule,c):
    # r: (R,B,G) rewards for R parallel runs
    G=r.shape[2]
    mu=r.mean(2,keepdims=True)
    if rule=="grpo":
        s=r.std(2,keepdims=True,ddof=1)            # TRL: Bessel std per group
        return np.where(s>0,(r-mu)/(s+1e-4),0.0)
    if rule=="dr":
        return r-mu
    if rule=="rloo":
        return r-(r.sum(2,keepdims=True)-r)/(G-1)
    if rule=="batch":
        R=r.shape[0]
        S=r.reshape(R,-1).std(1,ddof=1)[:,None,None]   # TRL: one Bessel std over the batch
        return np.where(S>0,(r-mu)/(S+1e-4),0.0)
    if rule=="fixed":
        return (r-mu)/(c+1.0)
    raise ValueError(rule)

def run(rule,p,c,seeds=20,B=4,G=8,T=3000,eta=0.5,theta0=0.0,rec_every=10,seed0=0):
    """p, c: 1-D arrays of equal length (grid cells). Returns pi trajectory (T//rec_every, cells, seeds)."""
    rng=np.random.default_rng(seed0)
    cells=len(p)
    P=np.repeat(np.asarray(p,float),seeds)[:,None,None]
    C=np.repeat(np.asarray(c,float),seeds)[:,None,None]
    R=cells*seeds
    th=np.full(R,theta0)
    traj=[]
    for t in range(T):
        pi=1/(1+np.exp(-th))
        ans=rng.random((R,B,G))<pi[:,None,None]
        cor=rng.random((R,B,G))<P
        r=np.where(ans,np.where(cor,1.0,-C),0.0)
        A=advantages(r,rule,C)
        score=np.where(ans,(1-pi)[:,None,None],-pi[:,None,None])
        g=(A*score).mean((1,2))
        th=np.clip(th+eta*g,-12,12)
        if t%rec_every==rec_every-1:
            traj.append((1/(1+np.exp(-th))).reshape(cells,seeds))
    return np.array(traj)
