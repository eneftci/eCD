#!/bin/python
#-----------------------------------------------------------------------------
# File Name : mnist_feedback.py
# Purpose:
#
# Author: Emre Neftci
#
# Creation Date : 25-04-2013
# Last Modified : Fri 27 Jun 2014 03:42:28 PM PDT
#
# Copyright : (c) UCSD, Emre Neftci, Srinjoy Das, Bruno Pedroni, Kenneth Kreutz-Delgado, Gert Cauwenberghs
# Licence : GPLv2
#----------------------------------------------------------------------------- 
import numpy
import meta_parameters
meta_parameters.parameters_script = 'parameters_demo'
from common import *
from MNIST_IF_STDP_SEQ import main
from common import *
ioff()

matplotlib.rcParams['font.size']=22.0

#n_c_unit = N_c / n_classes


Wh,Wc,b_init = load_NS_v2(N_v, N_h, N_c, dataset = 'data/WSCD.pkl')
W = np.zeros([N_v+N_c,N_h])
W[:(N_v),:] = Wh
W[N_v:(N_v+N_c),:] = Wc.T
b_h = b_init[(N_v+N_c):]
b_v = b_init[:N_v]
b_c = b_init[N_v:(N_v+N_c)]

data =  mnist_data = load_MNIST(1,
                            min_p = 1e-4,
                            max_p = .9999,
                            binary = True,
                            seed = None)

def create_single_Id(idx, data, min_p = 1e-16, max_p = .9999, seed = None, mult_class=0.0, mult_data=1.0):
    
    iv_seq, iv_l_seq, train_iv, train_iv_l, test_iv, test_iv_l = data
    Idp = np.ones([N_v+N_c])*min_p
    i = np.nonzero(iv_l_seq==idx)[0][0]
    cl = np.zeros(N_c)
    cl[(iv_l_seq[i]*n_c_unit):((iv_l_seq[i]+1)*n_c_unit)] = max_p
    Idp[N_v:] = clamped_input_transform(cl, min_p = min_p, max_p = max_p)*mult_class
    Idp[:N_v] = clamped_input_transform(iv_seq[i,:], min_p = min_p, max_p = max_p)*mult_data
    Id = (Idp /beta)
    return Id



if __name__ == '__main__':
    #Mh, Mv, Mc, Mhelp = run_NS(5)
    hacked_digit = create_single_Id(8,data,mult_class=0.0,mult_data=1.0)
    hacked_digit[:N_v].reshape(28,28)[:,:14] = 0.
    cl = np.zeros(N_c)
    cl[(3*n_c_unit):(4*n_c_unit)] = .98
    cl[(6*n_c_unit):(7*n_c_unit)] = .98
    hacked_digit[N_v:]= clamped_input_transform(cl, min_p = 1e-16, max_p = .500+.2e-9)

    Ids_demo = np.load('data/ids.npy')
    Ids = np.column_stack([
        create_single_Id(3,data,mult_class=0.0,mult_data=1.0)*0,
        create_single_Id(3,data,mult_class=0.0,mult_data=1.0),
        create_single_Id(5,data,mult_class=1.0,mult_data=0.0),
        hacked_digit,
        ]).T

    Ids[-1,:N_v] = Ids_demo[-1,:N_v]
    Ids[1,:N_v] = Ids_demo[1,:N_v]

    out = main(W, b_v, b_c, b_h, Id = Ids)
    Mh, Mv, Mc= out['Mh'], out['Mv'], out['Mc']



    d = et.mksavedir()
    et.globaldata.Mc = Mc.spikes
    et.globaldata.Mv = Mv.spikes
    et.globaldata.Mh = Mh.spikes
    et.save()
    
    from plot_options import *
    pylab.ioff()


    bone()
    matplotlib.rcParams['figure.subplot.wspace']=.0
    matplotlib.rcParams['figure.subplot.hspace']=.0
    matplotlib.rcParams['figure.subplot.bottom']=.0
    matplotlib.rcParams['figure.subplot.left']=.0
    matplotlib.rcParams['figure.subplot.right']=1.0
    matplotlib.rcParams['figure.subplot.top']=1.0
    f1=np.array(spike_histogram(Mv,T1_s+10*t_ref,T1_e)).T[1].reshape(28,28)
    figure(); imshow(f1, interpolation = 'bicubic'); xticks([]), yticks([])
    et.savefig('pre_trained_prediction.png', format = 'png')

    f2=np.array(spike_histogram(Mv,T2_s+60*t_ref,T2_e)).T[1].reshape(28,28)
    figure(); imshow(f2, interpolation = 'bicubic'); xticks([]), yticks([])
    et.savefig('pre_trained_construction.png', format = 'png')

    f3=np.array(spike_histogram(Mv,T3_s+15*t_ref,T3_e)).T[1].reshape(28,28)
    figure(); imshow(f3, interpolation = 'bicubic'); xticks([]), yticks([])
    axvline(14,color = 'w', linewidth=3, alpha=0.8)
    et.savefig('pre_trained_inference.png', format = 'png')

    figure(figsize=(12.0, 4.5))
    matplotlib.rcParams['figure.subplot.bottom']=.17
    matplotlib.rcParams['figure.subplot.left']=.04
    matplotlib.rcParams['figure.subplot.right']=.90
    matplotlib.rcParams['figure.subplot.top']=.95
    raster_plot(Mv, Mc,newfigure=False,markersize=2,marker='|', color='k',mew=1)
    xt = xticks()[0]
    axhline(1, color='k', linewidth=2, alpha=0.8)
    axhline(2, color='k', linewidth=2, alpha=0.8)
    for i in range(n_classes):
        axhline(1+float(i)/10, color='k')

    axhline(2, color='k', linewidth=2, alpha=0.8)
    axvline(T1_s*1000, color='k')
    axvline(T1_e*1000, color='k')
    axvline(T2_s*1000, color='k')
    axvline(T2_e*1000, color='k')
    axvline(T3_s*1000, color='k')
    axvline(T3_e*1000, color='k')
    yticks([.5, 1.5],['$v_d$','$v_c$'])
    xlabel('Time[s]')
    ylabel('')
    ax = gca().twinx()
    xt = np.array([0,round(T1_s),round(T2_s),round(T3_s,1)])
    xticks((xt+init_delay)*1000, xt) 
    xlim([.1*1000,t_sim*1000-40])
    ylim([0,3])
    yticks(np.arange(1.05,2.0,.2), ['${0}$'.format(i) for  i in range(0,10,2)], fontsize=22)
    et.savefig('pretrained_raster_all.png', format='png')

    matplotlib.rcParams['figure.subplot.left']=.25
    matplotlib.rcParams['figure.subplot.right']=.94
    figure(figsize=(6.0,4.0))
    Sh=monitor_to_spikelist(Mh).time_slice(0,t_sim*1000)
    Sv=monitor_to_spikelist(Mv).time_slice(0,t_sim*1000)
    Sc=monitor_to_spikelist(Mc).time_slice(0,t_sim*1000)
    Sh.time_offset(-init_delay*1000)
    Sv.time_offset(-init_delay*1000)
    Sc.time_offset(-init_delay*1000)
    tbin = 10
    labello = ['$v_d$', '$h$', '$v_c$']
    for i, S in enumerate([Sv, Sh, Sc]):
        plot(S.time_axis(tbin)[:-1], S.spike_histogram(time_bin=tbin, normalized=True).mean(axis=0), '.-' , linewidth=2,label = labello[i], markersize=7)
    axvline(T1_s*1000-init_delay*1000, color='k')
    axvline(T1_e*1000-init_delay*1000, color='k')
    axvline(T2_s*1000-init_delay*1000, color='k')
    axvline(T2_e*1000-init_delay*1000, color='k')
    axvline(T3_s*1000-init_delay*1000, color='k')
    axvline(T3_e*1000-init_delay*1000, color='k')
    xlim([-.2*1000,t_sim*1000-init_delay*1000])
    xticks((xt)*1000, xt) 
    ylabel('Firing rate [Hz]')
    ylim([0,100])
    legend(labelspacing=0, ncol=3, frameon=1, borderpad=0, borderaxespad=0, columnspacing=.1, handletextpad=0)
    gca().add_patch(Rectangle((0,1), 10*t_ref*1000, 2.5, fill=True, color='k'))
    et.savefig('pretrained_rates.png', format='png')

#    xt = np.array([0,0.1,0.2,0.3,.85])
#    figure(figsize = (8,6))
#    colorlist = ['b','b','b','r','b','b','b','b','b','b']
#    for i in range(10): plot(out['Mvmem'].times[0:T1_e*10000], i+ 0.75*out['Mvmem'].values[2+i*4,0:T1_e*10000], color=colorlist[i])
#    xticks((xt+init_delay), xt)
#    axvline(0.0)
#    xlim([0,T1_e])
#
#    figure(figsize = (8,6))
#    raster_plot(Mv, Mc,newfigure=False,markersize=4,marker='.')
#    axhline(1, color='k', linewidth=2, alpha=0.8)
#    axhline(2, color='k', linewidth=2, alpha=0.8)
#    for i in range(n_classes):
#        axhline(1+float(i)/10, color='k')
#
#    axhline(2, color='k', linewidth=2, alpha=0.8)
#    axvline(T1_s*1000, color='k')
#    axvline(T1_e*1000, color='k')
#    axvline(T2_s*1000, color='k')
#    axvline(T2_e*1000, color='k')
#    axvline(T3_s*1000, color='k')
#    axvline(T3_e*1000, color='k')
#    yticks([.5, 1.5],['$v_{d}$','$v_{c}$'])
#    xlabel('Time[s]')
#    ylabel('')
#    ax = gca().twinx()
#    xlim([0,T1_e*1000])
#    xlabel('')
#    xticks((xt+init_delay)*1000, xt) 
#    ylim([0,3])
#    yticks([])
    
    show()
