#!/bin/python
#-----------------------------------------------------------------------------
# File Name : MNIST_SRM_RATE_script200.py
# Purpose:
#
# Author: Emre Neftci
#
# Creation Date : 01-10-2013
# Last Modified : Fri 27 Jun 2014 02:36:33 PM PDT
#
# Copyright : (c) 
# Licence : GPLv2
#----------------------------------------------------------------------------- 
import meta_parameters
meta_parameters.parameters_script = 'parameters_mnist'

from MNIST_IF_STDP_UB import *

if __name__ == '__main__':
    mnist_data = load_mnist_data(min_p = .00001, max_p = .98, binary = True, seed=0)
    Id = create_Id(mnist_data, c_min_p = .00001, c_max_p = .98)
    W, b_v, b_c, b_h = create_rbm_parameters()
#    Wh,Wc,b_init = load_NS_v2(N_v,N_h,N_c,dataset = 'Results//073a__06-10-2013/WSCD.pkl')
#    W = np.zeros([N_v+N_c,N_h])
#    W[:(N_v),:] = Wh
#    W[N_v:(N_v+N_c),:] = Wc.T
#    b_h = b_init[(N_v+N_c):]
#    b_v = b_init[:N_v]
#    b_c = b_init[N_v:(N_v+N_c)]
    loc = main(W, b_v, b_c, b_h, Id = Id, dorun = True, monitors = False, display=True, mnist_data=mnist_data)
    locals().update(loc)
    W=Srs.w.data.copy().reshape(N_v+N_c, N_h)*beta
    Wvh=W[:N_v,:]
    Wch=W[N_v:,:]
    mBv = Sbv.w.data*beta*tau_rec*bias_input_rate
    mBh = Sbh.w.data*beta*tau_rec*bias_input_rate
    b_c = mBv[N_v:(N_v+N_c)]
    b_v = mBv[:N_v]
    b_h = mBh
    mB = np.concatenate([mBv,mBh])

    d = et.mksavedir()
    et.save_file(__file__)
    et.globaldata.W = W
    et.globaldata.mB = mB
    
    try:
        et.globaldata.Mv = monitor_to_spikelist(Mv)
        et.globaldata.Mh = monitor_to_spikelist(Mh)
    except NameError:
        print "SpikeMonitors are not defined"
        
    et.globaldata.res_hist_train = res_hist_train 
    et.globaldata.res_hist_test = res_hist_test 
    et.globaldata.w_hist_v   = w_hist_v
    et.globaldata.w_hist_c   = w_hist_c
    et.globaldata.b_hist_vcn = res_hist_train
    et.globaldata.b_hist_h   = res_hist_test 
    et.save({'Wh':Wvh, 'Wc':Wch, 'b_vch': mB}, 'WSCD.pkl')
    et.save()
    et.savefig('progress.png', format='png')
#
