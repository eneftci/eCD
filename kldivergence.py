#!/bin/python
#-----------------------------------------------------------------------------
# File Name : neural_sampler_clif.py
# Purpose:
#
# Author: Emre Neftci
#
# Creation Date : 24-04-2013
# Last Modified : Fri 27 Jun 2014 04:01:20 PM PDT
#
# Copyright : (c) 
# Licence : GPLv2
#----------------------------------------------------------------------------- 
import meta_parameters
meta_parameters.parameters_script = 'parameters_kldivergence'
from common import *
from neuro_kl import kl_tools

import MNIST_IF_STDP
import MNIST_IF_RATE

runs = ['STDP','RATE']
        ]
colors = ['g','b','r']

def run_NS(params=None):
    if params == None:
        np.random.seed()
        b_h = np.random.normal(-1.5, 0.5, size=N_h)
        b_v = np.random.normal(-1.5, 0.5, size=N_v)
        b_c = np.random.normal(-1.5, 0.5, size=N_c)
        W = np.random.normal(-.75, 1.5, size=(N_v,N_h))  
    else:
        W = params['W']
        b_v = params['b_v']
        b_h = params['b_h']
        b_c = np.random.normal(-.75, 1.5, size=N_c)
    
    Id = create_Id(False)
    r1 = MNIST_IF_STDP.main(W, b_v, b_c, b_h, Id, monitors = True)
    r2 = MNIST_IF_RATE.main(W, b_v, b_c, b_h, Id, monitors = True, vmem_monitors=False)
    return [r1,r2], W, b_v, b_h


def run_GS(W, b1, b2, n):
    steps = n
    from neusa.RBMSampler import RBMGibbsSampler 
    g = RBMGibbsSampler(W, b1, b2)
    i = np.random.randint(0,2, size=N_h)
    lv, lh = g.manysteps(i,steps)
    a=np.hstack([lv,lh])
    states = kl_tools.spikes2states(a)
    return states

def states_NS(Mv, Mh, t_conv=1.0):
    Sv = monitor_to_spikelist(Mv).time_slice(10*t_ref, t_sim*1000)
    Sh = monitor_to_spikelist(Mh).time_slice(10*t_ref, t_sim*1000)
    Sv.time_offset(-10*t_ref)
    Sh.time_offset(-10*t_ref)
    Sv.t_start = 0
    Sv.t_stop = t_sim*1000 - 10*t_ref
    Sh.t_start = 0
    Sh.t_stop = t_sim*1000 - 10*t_ref
    Sv.complete(np.arange(N_v))
    Sh.complete(np.arange(N_h))
    sv = Sv.spike_histogram(time_bin=1.).astype('bool').astype('int')
    sh = Sh.spike_histogram(time_bin=1.).astype('bool').astype('int')
    s = np.row_stack([sv, sh])
    s_conv = []
    for row in s:
        s_conv.append(np.convolve(row, np.ones([t_conv*t_ref/ms])))
    s_conv = np.array(s_conv).T.astype('int')
    s_conv[s_conv>1]=1
    states = kl_tools.spikes2states(s_conv)
    return states

def entropy(P):
    negh = 0
    for p_i in P:
        negh += p_i*np.log(p_i)
    return -negh

def run_exact(W, b1, b2):
    distr2 = np.empty([2**(N_v+N_c+N_h)])
    f,w,b = build_pdf(W, b1, b2)
    for i,z in enumerate(iter_bin_vec(N_v+N_c+N_h)):
        distr2[i] = f(z)
    distr2 /= distr2.sum()
    return distr2

def wrap_run(runID = 0):
    if isinstance(runID, int):
        r, W, b_v, b_h = run_NS()
    elif isinstance(runID, dict):
        r, W, b_v, b_h = run_NS(runID)

    states_ns = []
    for rr in r: 
        states = states_NS(rr['Mv'], rr['Mh'], t_conv = 1.0)
        distr = kl_tools.states2distr(states,N_v+N_c+N_h)+1
        states_ns.append(states)


    return states_ns, W, b_v, b_h

def compute_distr_ns(states, W, b_v, b_h, steps=10, T=t_sim*1000):
    nT = int(T)
    distr = kl_tools.states2distr(states[:nT],N_v+N_c+N_h)+1
    distr/= distr.sum()
    res_distr_cut = np.zeros(steps)
    nTi = np.logspace(2,np.log10(nT),10,base=10)
    for i in range(steps):
        d  = kl_tools.states2distr(states[ :nTi[i] ], N_v+N_c+N_h)+1
        d /= d.sum()
        res_distr_cut[i] = kl_divergence_pdf(d, W, b_v, b_h)/entropy(d)

    return distr, nTi, res_distr_cut


if __name__ == '__main__':
    import multiprocessing
    pool = multiprocessing.Pool(4)

    out = pool.map(wrap_run, range(Nruns))

    #Save Data
    d = et.mksavedir()
    et.globaldata.out = out
    et.save()

