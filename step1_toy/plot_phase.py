import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from grpo_exact import thresh
d=np.load("grid.npz"); ps,cs=d["ps"],d["cs"]
INK="#0b0b0b"; INK2="#52514e"; ORANGE="#eb6834"; SURF="#fcfcfb"
cmap=LinearSegmentedColormap.from_list("blue",["#f3f7fc","#cde2fb","#9ec5f4","#6da7ec","#3987e5","#256abf","#184f95","#0d366b"])
plt.rcParams.update({"font.size":9,"axes.edgecolor":INK2,"axes.labelcolor":INK,"xtick.color":INK2,"ytick.color":INK2,
                     "axes.titlesize":10,"axes.titleweight":"bold","figure.facecolor":"white"})
rules=[("grpo","Standard GRPO (per-group std)"),("dr","Dr. GRPO (no std)"),("rloo","RLOO"),
       ("batch","Batch std (4 groups of 8)"),("fixed",r"Fixed scale ($\kappa=c+1$)")]
cfine=np.linspace(1,10,46)
truth=cfine/(1+cfine)
pred_line={"grpo":np.array([thresh(0.5,8,c) for c in cfine]),
           "batch":np.array([thresh(0.5,32,c) for c in cfine])}
fig,axes=plt.subplots(2,3,figsize=(10.5,6.6),constrained_layout=True)
dp=ps[1]-ps[0]
for ax,(r,title) in zip(axes.ravel(),rules):
    fin=d[r+"_traj"][-10:].mean(0).mean(-1)
    m=ax.pcolormesh(np.r_[ps-dp/2,ps[-1]+dp/2],np.r_[cs-0.5,cs[-1]+0.5],fin,cmap=cmap,vmin=0,vmax=1,
                    edgecolors=SURF,linewidth=0.6)
    ax.plot(truth,cfine,color=INK,lw=2,label=r"correct boundary $p=c/(1+c)$")
    if r in pred_line:
        ax.plot(pred_line[r],cfine,color=ORANGE,lw=2,ls=(0,(4,2)),label=r"Task 1: $p^*$ at $\pi=0.5$")
    ax.set_title(title,loc="left",color=INK)
    ax.set_xlim(0.40,1.00); ax.set_ylim(0.5,10.5); ax.set_yticks(range(1,11))
    ax.set_xlabel(r"accuracy when answering, $p$"); ax.set_ylabel(r"penalty $c$")
    for s in ["top","right"]: ax.spines[s].set_visible(False)
cb=fig.colorbar(m,ax=axes[0,:],shrink=0.9,pad=0.01); cb.set_label(r"final answer rate $\pi$ (mean of 20 seeds)")
cb.outline.set_visible(False)
# panel 6: simulated vs predicted
ax=axes[1,2]
for r,_ in rules:
    fin=d[r+"_traj"][-10:].mean(0).mean(-1).ravel(); pr=d[r+"_pred"].ravel()
    ax.scatter(pr,fin,s=14,color="#256abf",alpha=0.45,edgecolors="none")
ax.plot([0,1],[0,1],color=INK2,lw=1,ls=(0,(2,2)))
ax.set_title("Simulation vs Task 1 prediction",loc="left",color=INK)
ax.set_xlabel(r"predicted final $\pi$"); ax.set_ylabel(r"simulated final $\pi$")
ax.set_xlim(-0.03,1.03); ax.set_ylim(-0.03,1.03); ax.set_aspect("equal")
for s in ["top","right"]: ax.spines[s].set_visible(False)
ax.text(0.04,0.93,"all 5 rules, 1,500 cells",color=INK2,fontsize=8,transform=ax.transAxes)
h,l=axes[0,0].get_legend_handles_labels()
fig.legend(h,l,loc="outside lower center",ncol=2,frameon=False)
fig.savefig("phase_diagram.png",dpi=200); fig.savefig("phase_diagram.pdf")
