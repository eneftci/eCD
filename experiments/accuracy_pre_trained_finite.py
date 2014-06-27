#!/bin/python
#-----------------------------------------------------------------------------
# File Name : accuracy_pre_trained_finite_UB.py
# Purpose: Tests the accuracy of a pre-trained RBM for finite precision parameters
#
# Author: Emre Neftci
#
# Creation Date : 25-04-2013
# Last Modified : Fri 27 Jun 2014 01:56:12 PM PDT
#
# Copyright : (c) UCSD, Emre Neftci, Srinjoy Das, Bruno Pedroni, Kenneth Kreutz-Delgado, Gert Cauwenberghs
# Licence : GPLv2
#----------------------------------------------------------------------------- 

import meta_parameters
meta_parameters.parameters_script = 'accuracy_parameters_short_unbounded'
from common import *
from MNIST_IF_RATE_UB import main

Wh,Wc,b_init = load_NS_v2(N_v, N_h, N_c, dataset = '../data/WSCD.pkl')
W = np.zeros([N_v+N_c,N_h])
W[:(N_v),:] = Wh
W[N_v:(N_v+N_c),:] = Wc.T
b_h = b_init[(N_v+N_c):]
b_v = b_init[:N_v]
b_c = b_init[N_v:(N_v+N_c)]


data =  mnist_data = load_MNIST(1,
                            min_p = 1e-5,
                            max_p = .98,
                            binary = True,
                            seed = None)

#Discretize the Boltzmann machine parameters 
#The parameter range is [mean-c*std, mean+c*std]
nbits = 8 #Number of bits
c=8.5 #Cut-off
b_v = round_to_bits(b_v, nbits, clip = [b_v.mean()-c*b_v.std(),b_v.mean()+c*b_v.std()])
b_h = round_to_bits(b_h, nbits, clip = [b_h.mean()-c*b_h.std(),b_h.mean()+c*b_h.std()])
b_c = round_to_bits(b_c, nbits, clip = [b_c.mean()-c*b_c.std(),b_c.mean()+c*b_c.std()])
W=np.vstack([Wh,Wc.T])
W_finite = round_to_bits(W, nbits, clip = [W.mean()-c*W.std(),W.mean()+c*W.std()])
Wh = W_finite[:N_v]
Wc = W_finite[N_v:].T

def create_Id_no_class(data, min_p = 1e-4, max_p = .98, seed = None):
    iv_seq, iv_l_seq, train_iv, train_iv_l, test_iv, test_iv_l = data
    Idp = np.ones([N, N_v+N_c])*min_p
    for i in range(N):
        Idp[i,N_v:] = 0.
        Idp[i,:N_v] = clamped_input_transform(test_iv[i,:], min_p = min_p, max_p = max_p)
    Id = (Idp /beta)
    return Id, test_iv, test_iv_l


def wrap_run(Id):
    out = main(W_finite, b_v, b_c, b_h, Id = np.array([Id]))
    Mh, Mv= out['Mh'], out['Mv']
    return monitor_to_spikelist(Mv).firing_rate(tbin)[N_v:,:]
if __name__ == '__main__':

    Ids, test_data, test_labels = create_Id_no_class(data)
    
    free_en_perf = classification_free_energy(Wh, Wc.T, b_h, b_c, test_data[:N,:N_v], test_labels[:N], n_c_unit)[0]
    print 'free energy all test samples {0}'.format(classification_free_energy(Wh, Wc.T, b_h, b_c, data[4], data[5], n_c_unit)[0])
    print 'free energy {1} test samples {0}'.format(free_en_perf, N)

    import multiprocessing
    pool = multiprocessing.Pool(8)
    pool_out = pool.map(wrap_run, Ids)

    et.globaldata.pool_out = pool_out
    et.mksavedir()
    et.save()

    from plot_options import *
    matplotlib.rcParams['figure.subplot.right'] = .8
    nt = pool_out[0].shape[1]
    res_out = [[None for j in range(N)] for i in range(nt)]
    for i in range(1,nt):
        for j,r in enumerate(pool_out):
            rate_up_to_ti = r[:,:i].mean(axis=1)
            res_out[i][j] = rate_up_to_ti.reshape(n_classes,N_c/n_classes).mean(axis=1).argmax() == test_labels[j]
    res_out = np.array(res_out[1:])
    xaxis = np.arange(tbin,t_sim*1000,tbin)/1000
    plot(xaxis,res_out.mean(axis=1), alpha=0.5, linewidth=2, color='b')
    axhline(.1,color='k',alpha=0.5,linewidth=3,linestyle='--')
    axvline(.2,color='k',alpha=0.5,linewidth=3)
    ylim(0,1)
    xlim(0,t_sim)
    axhline(free_en_perf,color='k',alpha=0.5,linewidth=2,linestyle='-')
    yticks([0,.1,1])
    ylabel('Accuracy')


#    res_out = [p[:,tl[i]].mean(axis=1).argmax() for i,p in enumerate(pool_out)]
#    print np.mean(np.array(res_out) == test_labels[:N])
##    print os.path.dirname(os.path.abspath(__file__))
#    et.mksavedir()
#    et.annotate('PERFORMANCE','{0}'.format(np.mean(np.array(res_out) == test_labels[:N])))
#    et.globaldata.pool_out = pool_out
#    et.save_file(__file__)
#    et.save()
#    import matplotlib, pylab
#    matplotlib.rcParams['savefig.dpi']=180.
#    matplotlib.rcParams['font.size']=26.0
#    matplotlib.rcParams['figure.figsize']=(6.0,6.0)
#    matplotlib.rcParams['axes.formatter.limits']=[-10,10]
#    pylab.rc('legend', borderaxespad=0., borderpad=.4,
#    handlelength=1.4, labelspacing=0.4)
#
#    figure()
#    ion()
#    raster_plot(Mv, Mh, Mc)
#    axhline(1, color='k', linewidth=2, alpha=0.8)
#    axhline(2, color='k', linewidth=2, alpha=0.8)
#    yticks([.5, 1.5, 2.5],['v$','$h$','$c$'])
#    ylabel('')
#    xlim([0,500])
#    pylab.savefig('paper/raster_reconstruction.png', format='png')
#
#    figure()
#    imshow(np.array(spike_histogram(Mv,.1,1)).T[1].reshape(28,28))
#    xticks([])
#    yticks([])
#    pylab.savefig('paper/reconstruction.png', format='png')
#
#    figure()
#    N = MV.values.shape[0]
#    for i in range(N):
#    if i==9:
#        c='r'
#    else:
#        c='k'
#    plot(np.concatenate([np.array([-0.1]),MV.times]),np.concatenate([np.array([i]),0.7*MV.values[i,:]+i]), c)
#    xlim([-0.1,0.5])
#    ylim([-1,10])
#    yticks(range(10))
#    xticks([0,0.5])
#    xlabel('Time[s]')
#    ylabel('Class Label Neuron #')
#    #gca().add_patch(Rectangle((-0.05,0),0.02,.7, color='k'))
#    #text(-0.07,-0.6, '1.0V')
#    pylab.savefig('paper/vmem.png', format='png')
