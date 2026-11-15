import numpy as np,time
from sim import run, RULES
from predict import predict
ps=np.round(np.arange(0.41,0.991,0.02),2); cs=np.arange(1,11).astype(float)
PP,CC=np.meshgrid(ps,cs,indexing="xy")   # rows c, cols p
pf,cf=PP.ravel(),CC.ravel()
out={"ps":ps,"cs":cs}
for k,rule in enumerate(RULES):
    t=time.time()
    tr=run(rule,pf,cf,seeds=20,T=8000,rec_every=100,seed0=100+k)
    out[rule+"_traj"]=tr.reshape(tr.shape[0],len(cs),len(ps),20)
    out[rule+"_pred"]=np.array([predict(rule,p,c) for p,c in zip(pf,cf)]).reshape(len(cs),len(ps))
    print(rule,f"{time.time()-t:.0f}s",flush=True)
np.savez_compressed("grid.npz",**out)
