#!/bin/python
#-----------------------------------------------------------------------------
# File Name : mnist_feedback.py
# Purpose:
#
# Author: Emre Neftci
#
# Creation Date : 25-04-2013
# Last Modified : Tue 28 Jan 2014 11:48:42 AM PST
#
# Copyright : (c) 
# Licence : GPLv2
#----------------------------------------------------------------------------- 

import meta_parameters
meta_parameters.parameters_script = 'convergence_parameters_UB'
from common import *
from MNIST_IF_STDP_WMON_UB import main

#dataset_dir = 'Results/073a__06-10-2013/'
dataset_dir = '../data/'
ioff()

N = 1

data =  mnist_data = load_MNIST(1,
                            min_p = 1e-5,
                            max_p = .98,
                            binary = True,
                            seed = 0)

if __name__ == '__main__':

    Ids = create_Id(data, c_min_p = 1e-5, c_max_p = .98,seed=0)


    W, b_v, b_c, b_h = create_rbm_parameters()
    out_before = main(W, b_v, b_c, b_h, Id = Ids, display=True, mnist_data = data)

    #Wh, Wc, b_init = load_matlab_v2(N_v, N_h, N_c, model = 'model5', dataset = '../data/model_5units.mat')
    Wh,Wc,b_init = load_NS_v2(N_v, N_h, N_c, dataset = dataset_dir + 'WSCD.pkl')
    #Wh,Wc,b_init = load_matlab_v2(N_v, N_h, N_c, model='model', dataset = '../data/rbm_gibbs_bias_momentum_decay_multiclassunits.mat')
    W = np.zeros([N_v+N_c,N_h])
    W[:(N_v),:] = Wh
    W[N_v:(N_v+N_c),:] = Wc.T
    b_h = b_init[(N_v+N_c):]
    b_v = b_init[:N_v]
    b_c = b_init[N_v:(N_v+N_c)]
    out_after = main(W, b_v, b_c, b_h, Id = Ids, display=True, mnist_data = data)

    from plot_options import *
    d = et.mksavedir()

    def raster_plot_extra(Mv,Mc,Mh,SynMg, SynMn, SynMa, SynMd, w_hist_v, w_hist_c, t_start = 0, t_stop = 1.0, fn_pre = None, legend_d=True):
        matplotlib.rcParams['font.size']=26.0
        matplotlib.rcParams['figure.subplot.top'] = .9
        matplotlib.rcParams['figure.subplot.bottom'] = .15
        matplotlib.rcParams['figure.subplot.left'] =.14
        matplotlib.rcParams['figure.subplot.right'] = .96
        figure(figsize=(8.0, 4.8))
        raster_plot(Mv,Mh,Mc,markersize=1,marker='|', color='k',mew=1)
        
        xticks([t_start*1000, float(t_stop-t_start)*1000/2, t_stop*1000], [t_start, float(t_stop-t_start)/2, t_stop])
        yticks([.5, 1.5, 2.5], [ '$v_d$', '$h$', '$v_c$'])
        axhline(1.0, color='k', linewidth=2)
        axhline(2.0, color='k', linewidth=2)
        xlim([t_start*1000, t_stop*1000])
        ylabel('')
        xlabel('Time[s]')
        ylim([0,3.0])
        #x=[]
        t = np.arange(t_start,t_stop,10*defaultclock.dt)
        #for tt in t:
        #    x.append((mean(input_v(tt)))>0)
        if fn_pre!=None:
            et.savefig(fn_pre+'_training_raster.png',format='png')

        matplotlib.rcParams['figure.subplot.top'] = .8
        matplotlib.rcParams['figure.subplot.bottom'] = .2
        figure(figsize=(8.0,1.0))
        zv=np.array(w_hist_v)[t_start*1000:1000*t_stop]
        zc=np.array(w_hist_c)[t_start*1000:1000*t_stop]
        plot(t,(zv+zc)/2*beta, linewidth=2, alpha=0.6)
        xticks([])
        yticks(np.array(yticks()[0])[[0,-1]], np.array(yticks()[0])[[0,-1]]*1e3)
        if legend_d: ylabel('$\\langle W \\rangle [10^{{-3}}]$', fontsize=17)
        if legend_d: legend(prop={'size':15}, ncol=2, bbox_to_anchor=(1.0,1.15))
        if fn_pre!=None:
            et.savefig(fn_pre+'_training_W.png',format='png')

        figure(figsize=(8.0,1.0))
        g=SynMg.values.T[t_start*1000:1000*t_stop,:]
        plot(t,g, label='g(t)', linewidth=2, alpha=0.6)
        ylim([-1.2,1.5])
        yticks([-1,1])
        xticks([])
        #plot(t,x, label='clamping', linewidth=2, alpha=0.6);
        if legend_d: legend(loc=1,prop={'size':16}, bbox_to_anchor=(1.0, 1.39), ncol=2)
        if fn_pre!=None:
            et.savefig(fn_pre+'_training_clamp.png',format='png')

        #ilk = w_input*beta_fi*noise_input_rate
        figure(figsize=(8.0,2.0))
        #plot(t, 1e9*(SynMn.values.T[t_start*1000:1000*t_stop,:]-ilk), 'k', label='$I_n$ [nA]', linewidth=2, alpha=0.4)
        plot(t, 1e9*SynMa.values.T[t_start*1000:1000*t_stop,:], label='$I_h$ [nA]', linewidth=2, alpha=1.0)
        plot(t, 1e9*SynMd.values.T[t_start*1000:1000*t_stop,:], label='$I_d$ [nA]', linewidth=2, alpha=1.0)
        ylim([-1.5,1.5])
        yticks([-10,2])
        ylabel('I[nA]')
        xticks([])
        if legend_d: legend(prop={'size':16}, ncol=3, bbox_to_anchor=(1.0,0.18))
        if fn_pre!=None:
            et.savefig(fn_pre+'_training_I.png',format='png')



    raster_plot_extra(
            out_before['Mv'],
            out_before['Mc'],
            out_before['Mh'],
            out_before['SynMg'],
            out_before['SynMn'],
            out_before['SynMa'],
            out_before['SynMd'],
            out_before['w_hist_v'],
            out_before['w_hist_c'],
            dcmt*t_ref, 3*dcmt*t_ref,'before')
    raster_plot_extra(
            out_after['Mv'],
            out_after['Mc'],
            out_after['Mh'],
            out_after['SynMg'],
            out_after['SynMn'],
            out_after['SynMa'],
            out_after['SynMd'],
            out_after['w_hist_v'],
            out_after['w_hist_c'],
            dcmt*t_ref,3*dcmt*t_ref, 'after')

#   out = cPickle.load(file(dataset_dir+'/globaldata.pickle','r'))

#   matplotlib.rcParams['figure.subplot.top'] = .9
#   matplotlib.rcParams['figure.subplot.bottom'] = .2
#   matplotlib.rcParams['figure.subplot.left'] = .23
#   matplotlib.rcParams['figure.subplot.right'] = .85
#   plot(np.arange(0,10000,10), out.res_hist_test[:1000], color='k', linewidth=3, label = '0.1s', alpha=1.0)
#   ylabel('Accuracy')
#   xlabel('# Samples')
#   yticks([0.,.1,.5,.90,1])
#   axhline(0.90,color='k')
#   axhline(.1,color='k')
#   xlim([0, 6000])
#   xticks([0,5000,10000])
#   et.savefig('convergence.png', format='png')

