#!/bin/python
#-----------------------------------------------------------------------------
# File Name : calibrate.py
# Purpose: Calibrates the parameters of the sigmoid function by fitting the parameters gamma and beta to the transfer function of the IF neuron
#
# Author: Emre Neftci
#
# Creation Date : 24-04-2013
# Last Modified : Thu 22 Jan 2015 12:37:59 PM PST
#
# Copyright : (c) UCSD, Emre Neftci, Srinjoy Das, Bruno Pedroni, Kenneth Kreutz-Delgado, Gert Cauwenberghs
# Licence : GPLv2
#----------------------------------------------------------------------------- 
import meta_parameters 
meta_parameters.parameters_script = 'parameters_kldivergence'
from common import *

N = 20
t_sim = 5000*ms



def wrap_run_srm(bias):
    defaultclock.reinit()
    #----------------------------------------- RBM parameters
    #------------------------------------------ Neuron Groups
    eqs = Equations(eqs_str_lif_rd, 
            Cm = cm,
            I_inj = i_inj + bias,
            g = g_leak,
            tau_rec = tau_rec)

    #beta arbitrary choice
    neuron_group =  NeuronGroup((N),
          model=eqs,
          threshold = SimpleFunThreshold( exp_prob_beta_gamma(defaultclock.dt, beta, g_leak, gamma), state = 'v'),
          refractory = t_ref)


    #set injection currents
    #---------------------- Connections and Synapses
    #Inject Noise for neural sampling

    #--------------------------- Monitors
    M  = SpikeMonitor(neuron_group)
    net = Network([neuron_group, M])
    print "running"
    net.run(t_sim)
    mmon_slice = time_slice(M, t_start=.5*second, t_stop=t_sim)
    return float(len(mmon_slice.spikes)) / N / (t_sim-.5*second)

def wrap_run_if(bias):
    defaultclock.reinit()
    #----------------------------------------- RBM parameters
    #------------------------------------------ Neuron Groups
    eqs_if = Equations(eqs_str_lif_wnrd, #neuron equations are described in neusa.experimentLib
            Cm = cm,
            I_inj = i_inj + bias,
            g = g_leak,
            sigma = wnsigma,
            tau_rec = tau_rec)
                                               
    #beta arbitrary choice
    neuron_group = NeuronGroup(\
            N,
            model = eqs_if,
            threshold = 'v>theta*volt',
            refractory = t_ref,
            reset = 0*volt
            )


    #set injection currents
    #---------------------- Connections and Synapses
    #Inject Noise for neural sampling

    #--------------------------- Monitors
    M  = SpikeMonitor(neuron_group)
    net = Network([neuron_group, M])
    print "running"
    net.run(t_sim)
    mmon_slice = time_slice(M, t_start=.5*second, t_stop=t_sim)
    return float(len(mmon_slice.spikes)) / N / (t_sim-.5*second)

def wrap_run_clif(bias):
    defaultclock.reinit()
    #----------------------------------------- RBM parameters
    #------------------------------------------ Neuron Groups
    eqs = Equations(eqs_str_clif_wnrd, #neuron equations are described in neusa.experimentLib
            Cm = cm,
            I_inj = ci_inj + bias*beta/cbeta,
            g = cg_leak,
            sigma = wnsigma,
            tau_rec = tau_rec)
                                               
    neuron_group = NeuronGroup(\
            N,
            model = eqs,
            threshold = 'v>ctheta*volt',
            refractory = t_ref,
            reset = 0*volt
            )

    #set injection currents
    #---------------------- Connections and Synapses
    #Inject Noise for neural sampling

    #--------------------------- Monitors
    M  = SpikeMonitor(neuron_group)
    @network_operation
    def rigid_boundary():
        neuron_group.v[neuron_group.v<0] = 0
    net = Network([neuron_group, M, rigid_boundary])
    print "running"
    net.run(t_sim)
    mmon_slice = time_slice(M, t_start=.5*second, t_stop=t_sim)
    return float(len(mmon_slice.spikes)) / N / (t_sim-.5*second)

if __name__ == '__main__':
    #Number of points of the transfer curve
    N_run = 36
    import multiprocessing
    pool = multiprocessing.Pool(8)

    input_rates = np.linspace(-4000e-12,4000e-12, N_run)
    rates_out_clif = np.array(pool.map(wrap_run_clif, input_rates))
    rates_out_srm = np.array(pool.map(wrap_run_srm, input_rates))
    rates_out_if = np.array(pool.map(wrap_run_if, input_rates))

    plot(input_rates,rates_out_srm, label = 'srm')
    plot(input_rates,rates_out_if, label = 'if')
    plot(input_rates,rates_out_clif, label = 'clif')
    plot(input_rates, (1./t_ref)/(1+np.exp(-beta*input_rates)), linewidth=2, color='k',  label='$\\left(\\tau+\\gamma^{-1}\\exp(-\\beta)\\right)^{-1}$')
    legend()


    
    #reconstruct pdf
    
    

