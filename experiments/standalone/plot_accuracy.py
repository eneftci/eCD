import meta_parameters
meta_parameters.parameters_script = 'accuracy_parameters_short_unbounded'
from common import *
from plot_options import *
matplotlib.rcParams['font.size']=22.0



et.globaldata.directory = 'Results/235a__27-11-2013/'
out_cd = et.load()
et.globaldata.directory = 'Results/238a__27-11-2013/'
out_ecd8 = et.load()

et.globaldata.directory = 'Results/240a__27-11-2013/'
out_ecd5 = et.load()

et.globaldata.directory = 'Results/306__27-11-2013/'
out_ecd = et.load()

out_ecd.free_en_perf = .908




pool_out_ecd  = out_ecd.pool_out
pool_out_cd   = out_cd.pool_out
pool_out_ecd8 = out_ecd8.pool_out
pool_out_ecd5 = out_ecd5.pool_out
test_labels = out_ecd8.test_labels

matplotlib.rcParams['figure.subplot.top'] = .85
matplotlib.rcParams['figure.subplot.right'] = .75
matplotlib.rcParams['figure.subplot.wspace'] = .4
nt = pool_out_cd[0].shape[1]

N=1000

def res_out(pool_out):
    res_out = [[None for j in range(N)] for i in range(nt)]
    N_c = len(pool_out[0])
    for i in range(1,nt):
        for j,r in enumerate(pool_out):
            rate_up_to_ti = r[:,:i].mean(axis=1)
            res_out[i][j] = rate_up_to_ti.reshape(n_classes,N_c/n_classes).mean(axis=1).argmax() == test_labels[j]
    res_out = np.array(res_out[1:])
    return res_out

xaxis = np.arange(tbin,t_sim*1000,tbin)/1000

res_out_ecd = res_out(pool_out_ecd)
res_out_cd = res_out(pool_out_cd)
res_out_ecd8 = res_out(pool_out_ecd8)
res_out_ecd5 = res_out(pool_out_ecd5)

subplot(121)
plot(xaxis,res_out_cd.mean(axis=1), linewidth=2, color='k', label='CD')
plot(xaxis,res_out_ecd.mean(axis=1), linewidth=2, color='g', label='eCD')
plot(xaxis,res_out_ecd8.mean(axis=1), linewidth=2, color='y', label='eCD8')
plot(xaxis,res_out_ecd5.mean(axis=1), linewidth=2, color='r', label='eCD5')

t_stop = 99

axhline(.1,color='k',alpha=0.2,linewidth=3,linestyle='-')

pt50 = np.nonzero(xaxis ==.05)[0]
annotate(
        '$50\\mathrm{{ms}}$,${0}\%$'.format(round(res_out_ecd.mean(1)[pt50],4)*100),
        (0.05,res_out_ecd.mean(1)[pt50]),
        (.15,.6),
        arrowprops=dict(arrowstyle="-",
        connectionstyle="arc3")
        )
plot(xaxis[pt50],res_out_ecd.mean(axis=1)[pt50], marker='o',markersize=5,color='b')

xlim(0,1)
xticks([0,1])
ylim(0,1)
#xlabel('Sampling Duration [s]')
yticks([0.1,1],[10,100])
ylabel('Recognition Accuracy%')

#annotate(
#        '',
#        (1.0,.95),
#        (1.25,1.0),
#        arrowprops=dict(arrowstyle="-"),
#        )
#
#annotate(
#        '',
#        (1.0,.85),
#        (1.25,.05),
#        arrowprops=dict(arrowstyle="-")
#        )

gca().add_patch(Rectangle((.05,.81),.95,.14, facecolor=None, edgecolor='k',linewidth=1,fill=False))

axhline(.936,color='k',alpha=0.5,linewidth=2,linestyle='--')

subplot(122)
plot(xaxis,res_out_ecd.mean(axis=1), linewidth=2, color='g', label='eCD')
plot(xaxis,res_out_ecd8.mean(axis=1), linewidth=2, color='y', label='eCD8')
plot(xaxis,res_out_ecd5.mean(axis=1), linewidth=2, color='r', label='eCD5')
plot(xaxis,res_out_cd.mean(axis=1), linewidth=2, color='k', label='CD')

#axhline(res_out_ecd.mean(axis=1)[t_stop],color='k',alpha=0.5,linewidth=2,linestyle='-')
#axhline(res_out_ecd8.mean(axis=1)[t_stop],color='k',alpha=0.5,linewidth=2,linestyle='-')
#axhline(res_out_ecd5.mean(axis=1)[t_stop],color='k',alpha=0.5,linewidth=2,linestyle='-')
axhline(out_ecd.free_en_perf,color='g',alpha=0.7,linewidth=2,linestyle='--')

pt999 = np.nonzero(xaxis==0.99)[0]
annotate(
        'CD ${0}\%$'.format(round(100*res_out_cd.mean(1)[pt999],4)),
        (1.0,res_out_cd.mean(1)[pt999]),
        (1.05,.95),
        arrowprops=dict(arrowstyle="-",
        connectionstyle="arc,angleA=-90,angleB=0,armA=30,armB=30,rad=0"),
        fontsize=17
        )

pt999 = np.nonzero(xaxis==0.99)[0]
annotate(
        'eCD ${0}\%$'.format(round(100*res_out_ecd.mean(1)[pt999],4)),
        (1.0,res_out_ecd.mean(1)[pt999]),
        (1.33,.017+res_out_ecd.mean(1)[pt999]),
        arrowprops=dict(arrowstyle="-",
        connectionstyle="arc,angleA=-90,angleB=0,armA=30,armB=30,rad=0"),
        fontsize=17
        )

pt999 = np.nonzero(xaxis==0.99)[0]
annotate(
        'eCD(8 bit)$\,{0}\%$'.format(round(100*res_out_ecd8.mean(1)[pt999],4)),
        (1.0,res_out_ecd8.mean(1)[pt999]),
        (1.05,.898),
        arrowprops=dict(arrowstyle="-",
        connectionstyle="arc,angleA=90,angleB=0,armA=30,armB=30,rad=0"),
        fontsize=17
        )

pt999 = np.nonzero(xaxis==0.99)[0]
annotate(
        'eCD(5 bit)$\,{0}\%$'.format(round(100*res_out_ecd5.mean(1)[pt999],4)),
        (1.0,res_out_ecd5.mean(1)[pt999]),
        (1.05,.85),
        arrowprops=dict(arrowstyle="-",
        connectionstyle="arc,angleA=90,angleB=0,armA=30,armB=30,rad=0"),
        fontsize=17
        )

text(.32,93.6/100 +.001, 'Free-Energy\nCD (93.6%)', color='k', fontsize=14,alpha=0.7) 
text(.45,out_ecd.free_en_perf-.007, 'Free-Energy\neCD ({0}%)'.format(out_ecd.free_en_perf*100), color='g', fontsize=14,alpha=0.8) 

xlabel('')

axhline(.936,color='k',alpha=0.5,linewidth=2,linestyle='--')
ylabel('')
xlim(0.1, 1.0)
ylim(0.81,.955)
xticks([.05,1])
yticks([.85,0.9,.95],[85,90,95])

legend(bbox_to_anchor = (1., 1.17),ncol=4,columnspacing=0.3, frameon = False, handletextpad = .2, prop={'size':17})

et.savefig('accuracy_progress.png',format='png',dpi=500)



