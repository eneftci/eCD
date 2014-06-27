#!/bin/python
#-----------------------------------------------------------------------------
# File Name : mnist_feedback.py
# Purpose:
#
# Author: Emre Neftci
#
# Creation Date : 25-04-2013
# Last Modified : Fri 27 Jun 2014 02:51:15 PM PDT
#
# Copyright : (c) UCSD, Emre Neftci, Srinjoy Das, Bruno Pedroni, Kenneth Kreutz-Delgado, Gert Cauwenberghs
# Licence : GPLv2
#----------------------------------------------------------------------------- 
import meta_parameters
meta_parameters.parameters_script = 'parameters_accuracy'
from common import *
from MNIST_IF_RATE import main

#Data ran on Results//207__25-11-2013/WSCD.pkl
Wh,Wc,b_init = load_NS_v2(N_v, N_h, N_c, dataset = '../data/WSCD.pkl')
W = np.zeros([N_v+N_c,N_h])
W[:(N_v),:] = Wh
W[N_v:(N_v+N_c),:] = Wc.T
b_h = b_init[(N_v+N_c):]
b_v = b_init[:N_v]
b_c = b_init[N_v:(N_v+N_c)]


data =  mnist_data = load_MNIST(1,
                            min_p = 1e-32,
                            max_p = 0.99,
                            binary = True,
                            seed = None)

def create_Id_no_class(data, min_p = 1e-32, max_p = 0.99, seed = None):
    iv_seq, iv_l_seq, train_iv, train_iv_l, test_iv, test_iv_l = data
    Idp = np.ones([N, N_v+N_c])*min_p
    for i in range(N):
        Idp[i,N_v:] = 0.
        Idp[i,:N_v] = clamped_input_transform(test_iv[i,:], min_p = min_p, max_p = max_p)
    Id = (Idp /beta)
    return Id, test_iv, test_iv_l


def wrap_run(Id):
    out = main(W, b_v, b_c, b_h, Id = np.array([Id]))
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

    et.globaldata.free_en_perf = free_en_perf
    et.globaldata.test_labels = test_labels
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
