import numpy as np
from math import comb
from scipy.optimize import brentq

def comps(G):
    for k in range(G+1):
        for m in range(G+1-k):
            yield k, m, G-k-m

def prob(k,m,n0,G,pi,p):
    qc,qw,q0=pi*p,pi*(1-p),1-pi
    return comb(G,k)*comb(G-k,m)*qc**k*qw**m*q0**n0

def drift(p,pi,G,c,rule="grpo",ddof=0,kappa=6.0):
    tot=0.0
    for k,m,n0 in comps(G):
        num=(k-c*m)*n0/G   # sum over answering samples of (r_i - rbar)
        if rule=="grpo":
            s2=(k+c*c*m)/G-((k-c*m)/G)**2
            if ddof==1: s2*=G/(G-1)
            if s2<=1e-15: continue
            num/=np.sqrt(s2)
        elif rule=="rloo":
            num*=G/(G-1)
        elif rule=="fixed":
            num/=kappa
        tot+=prob(k,m,n0,G,pi,p)*num
    return tot/G

def thresh(pi,G,c,rule="grpo",ddof=0):
    return brentq(lambda p: drift(p,pi,G,c,rule,ddof),1e-9,1-1e-9)

def drift_closed(p,pi,G,c,rule="dr",kappa=6.0):
    base=(G-1)/G*pi*(1-pi)*(p*(1+c)-c)
    return {"dr":base,"rloo":base*G/(G-1),"fixed":base/kappa}[rule]

if __name__=="__main__":
    c=5;G=8
    print("dr", thresh(0.5,G,c,"dr"), c/(1+c))
    for pi in [0.05,0.1,0.25,0.5,0.75,0.9,0.95]:
        t=thresh(pi,G,c); t1=thresh(pi,G,c,ddof=1)
        print(f"pi={pi}: p*={t:.4f} (ddof1 {t1:.4f})  c_eff={t/(1-t):.3f}")
    for G in [2,4,8,16,32,64]:
        t=thresh(0.5,G,5); print("G",G,round(t,4),round(t/(1-t),3))
