from kldivergence import *
import numpy as np
import sys
import matplotlib #For changing rcParams
import getopt

#Parse command line options
optlist,args=getopt.getopt(sys.argv[1:],'lprd:')

for o,a in optlist:        
    if o == '-l': #Use very cool Latex fonts
        print 'Using latex for figures'
        matplotlib.rcParams['text.usetex']=True
        matplotlib.rcParams['font.size']=22.0
    elif o == '-p': #Make nice figures for publication
        matplotlib.rcParams['savefig.dpi']=180.
        matplotlib.rcParams['font.size']=17.0
        matplotlib.rcParams['figure.figsize']=(8.0,6.0)
        matplotlib.rcParams['axes.formatter.limits']=[-10,10]
    elif o=='-f': #Make flatter figures (good for raster plots)
        matplotlib.rcParams['figure.figsize']=(8.0,4.8)
    elif o=='-d':
            et.globaldata.directory=a #Use this directory containing results instead of current

from plot_options import *

if len(sys.argv)==1:
    d=et.globaldata.directory
elif len(sys.argv)==2:
    d=et.globaldata.directory=sys.argv[1]

et.globaldata = et.load()
out = et.globaldata.out

Nruns=1

nsteps = int(t_sim/t_ref)
def r(params):
    return run_GS(params[0],params[1],params[2], n = nsteps)

def c(p):
    return compute_distr_ns(p[0], p[1], p[2], p[3])

def c_gs(p):
    return compute_distr_ns(p[0], p[1], p[2], p[3], T = int(t_sim/t_ref))

def pmplot(ax, x,y,s, **kwparams):
    plot(x,y, **kwparams)
    kwparams['linewidth']=0
    kwparams['alpha']=0.5
    kwparams.pop('marker')
    fill_between(x, y-s, y+s, **kwparams)
    return ax

import multiprocessing
pool = multiprocessing.Pool(24)

states_ns, W, b_v, b_h = zip(*out)
states_ns_ = zip(*states_ns)
states_gs_ = pool.map(r, [(W[i], b_v[i], b_h[i]) for i in range(Nruns)])

res_ns0_params = [(states_ns_[0][i], W[i], b_v[i], b_h[i]) for i in range(Nruns) ]
res_ns1_params = [(states_ns_[1][i], W[i], b_v[i], b_h[i]) for i in range(Nruns) ]
res_0 = pool.map(c, res_ns0_params)
res_1 = pool.map(c, res_ns1_params)
#res_0 = [compute_distr_ns(states_ns_[0][i], W[i], b_v[i], b_h[i]) for i in xrange(Nruns)]
#res_1 = [compute_distr_ns(states_ns_[1][i], W[i], b_v[i], b_h[i]) for i in xrange(Nruns)]
#
#
##res_2 = [compute_distr_ns(states_ns_[2][i], W[i], b_v[i], b_h[i]) for i in xrange(Nruns)]
res_gs_params = [(states_gs_[i], W[i], b_v[i], b_h[i]) for i in xrange(Nruns)]
res_exact = [run_exact(W[i], b_v[i], b_h[i]) for i in xrange(Nruns ) ]
res_gs = pool.map(c_gs, res_gs_params)
#
avg = lambda x: np.mean([x[i][2] for i in range(len(x))],axis=0)
std = lambda x: np.std([x[i][2] for i in range(len(x))],axis=0)
#
matplotlib.rcParams['figure.subplot.bottom'] = .19
matplotlib.rcParams['figure.subplot.top'] = .94
matplotlib.rcParams['figure.subplot.right'] = .94
matplotlib.rcParams['figure.subplot.left'] = .18
figure(figsize=[4.8,4.8])
ax = axes()
pmplot(ax, res_0[0][1] ,avg(res_0) ,std(res_0)  ,color=color2[0], linestyle='-', marker='x', label = '$P_{{ {0}, {1} }}$'.format("NS",runs[0]))
pmplot(ax, res_1[0][1] ,avg(res_1) ,std(res_1)  ,color=colors[1], linestyle='-', marker='x', label = '$P_{{ {0}, {1} }}$'.format("NS",runs[1]))
pmplot(ax, res_1[0][1] ,avg(res_gs),std(res_gs) ,color='k', linestyle='-', marker='x', label = '$P_{{ {0} }}$'.format("Gibbs"))
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(True)
ylim([.5e-3,1])
xlabel('Time [s]')
ylabel('Relative Inaccuracy')
legend(loc=3, prop={'size':20}, frameon=True, labelspacing=0.)
draw()
et.savefig('sampling_progress.png')


#print kl_divergence_pdf(np.ones(2**(N_v+N_h))/2**(N_v+N_h), W, b_v, b_h)/entropy(np.ones(2**(N_v+N_h))/2**(N_v+N_h))
#    
#
#
##Get parameters of first run
#distr_ns, d_gs, d_l, d_ex, kl1, kl2, params = out[2]
distr_ns = [res_0[0][0], res_1[0][0]]
#
##--------------------Plot------------------------------#
f= lambda x: '$[\\mathtt{'+np.binary_repr(x, width=5)+'}]$'
matplotlib.rcParams['figure.subplot.bottom'] = .35
figure(figsize=[10.0,4.8])

Nbars = len(distr_ns)+1
for i,d_ns in enumerate(distr_ns):
    bar(np.arange(32)+0.9*(float(i)+1)/Nbars, -np.log(d_ns[:32]), width=.9/Nbars, linewidth=0, color = colors[i], label = '$P_{{ {0}, {1} }}$'.format("NS",runs[i]))
    xticks(np.array(range(0,32,4)+[31])-1, map(f, range(0,32,4)+[31]), rotation=45)
    xlabel('$\\mathbf{z} = (z_1, z_2, z_3, z_4, z_5)$')
    ylabel('$-log(p(\\mathbf{z}))$')

bar(np.arange(32), -np.log(res_exact[0][:32]), width=.9/Nbars, linewidth=0, color = 'k', label = '$P_{exact}$')
legend(prop={'size':20},labelspacing=0.,frameon=False)

xlim([0,32])
ylim([0,15])
yticks([0,5,10,15])
et.savefig('kldivergence.pdf', format='pdf')

