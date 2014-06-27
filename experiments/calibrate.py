#!/bin/python
#-----------------------------------------------------------------------------
# File Name : calibrate.py
# Purpose: Calibrates the parameters of the sigmoid function by fitting the parameters gamma and beta to the transfer function of the IF neuron
#
# Author: Emre Neftci
#
# Creation Date : 24-04-2013
# Last Modified : Fri 27 Jun 2014 02:08:32 PM PDT
#
# Copyright : (c) UCSD, Emre Neftci, Srinjoy Das, Bruno Pedroni, Kenneth Kreutz-Delgado, Gert Cauwenberghs
# Licence : GPLv2
#----------------------------------------------------------------------------- 
import meta_parameters 
meta_parameters.parameters_script = 'parameters_calibrate'
from common import *

N_h = N_hidden = 20
t_sim = 5000*ms



def wrap_run(bias, t_ref):
    defaultclock.reinit()
    #----------------------------------------- RBM parameters
    bias_h = b_h = bias

    #------------------------------------------ Neuron Groups
    i_inj = 0*amp #We are trying to calibrate gamma and beta, so i-inj cannot be calculated
                  
    eqs = Equations(eqs_str_lif_wnrd, #neuron equations are described in neusa.experimentLib
            Cm = 1e-12*farad,
            I_inj = i_inj + bias_h,
            g = g_leak,
            sigma = wnsigma,
            tau_rec = tau_rec)
                                               
    neuron_group_hidden = NeuronGroup(\
            N_h,
            model = eqs,
            threshold = 'v>theta*volt',
            refractory = t_ref,
            reset = 0*volt
            )


    #set injection currents
    #---------------------- Connections and Synapses
    #Inject Noise for neural sampling

    #--------------------------- Monitors
    Mh  = SpikeMonitor(neuron_group_hidden)
    net = Network([neuron_group_hidden, Mh])
    print "running"
    net.run(t_sim)
    mmon_slice = time_slice(Mh, t_start=.5*second, t_stop=t_sim)
    return float(len(mmon_slice.spikes)) / N_h / (t_sim-.5*second)

def wrap_run_notref(bias):
    return wrap_run(bias, t_ref =0.)

def wrap_run_tref(bias):
    return wrap_run(bias, t_ref = t_ref)


if __name__ == '__main__':
    #Number of points of the transfer curve
    N_run = 48
    import multiprocessing
    pool = multiprocessing.Pool(8)
    d = et.mksavedir()
    rates = np.linspace(-6000e-12,00e-12, N_run)
    pool_out = pool.map(wrap_run_notref, rates)
    rates_out = np.array(pool_out)
 
    rates_sigm = np.linspace(-6000e-12,2000e-12, N_run)
    pool_out_sigm = pool.map(wrap_run_tref, rates_sigm)
    rates_out_sigm = np.array(pool_out_sigm)

    idx = (rates_out>30) * (rates_out<.7/t_ref)
    P = -polyfit(rates[idx], log(rates_out[idx]), 1)

    idx = (rates_out_sigm>40) * (rates_out_sigm<1./t_ref-40)
    P = polyfit(rates_sigm[idx], log(rates_out_sigm[idx]**-1-t_ref), 1)
    
    #Generate plot
    from plot_options import * 
    matplotlib.rcParams['font.size']=15.0
    matplotlib.rcParams['figure.subplot.left'] = .2
    matplotlib.rcParams['figure.subplot.bottom'] = .27
    matplotlib.rcParams['figure.subplot.right'] = .92
    figure(figsize = (5.0,3.0))
    title('$(\\tau_r = 0\\mathrm{ms})$')
    dx_plot = range(np.nonzero(rates_out>.5)[0][0],np.nonzero(rates_out>75)[0][0])
    plot(rates, np.exp(-P[1]-P[0]*rates), linewidth=2, color='k', label='$\\gamma\,\\exp(\\beta)$')
    plot(rates, rates_out, 'o', markersize=3, color='b',markeredgecolor='b', label='$\\rho(I)$')
    xlabel('$I_{inj}\mathrm{[nA]}$',fontsize=17)
    ylabel('$\\nu\mathrm{[Hz]}$',fontsize=17)
    xticks( xticks()[0][::2],
            xticks()[0][::2]*1e9)
    yticks([0,500])
    ylim([-15,500])
    legend(frameon=False, loc=2,prop={'size':15},numpoints=1,borderaxespad=0.1,handletextpad=0.2)
    et.savefig('exp.png', format='png', dpi=300)

    figure(figsize = (5.0,3.0))
    title('$(\\tau_r = 4\\mathrm{ms})$')
    plot(rates_sigm, (1./t_ref)/(1+np.exp(P[0]*rates_sigm+P[1]-np.log(t_ref))), linewidth=2, color='k',  label='$\\left(\\tau+\\gamma^{-1}\\exp(-\\beta)\\right)^{-1}$')
    plot(rates_sigm, rates_out_sigm, 'o', markersize=3, mfc='b',markeredgecolor='b',label='$\\rho(I)$')
    xlabel('$I_{inj}\mathrm{[nA]}$',fontsize=17)
    ylabel('$\\nu\mathrm{[Hz]}$',fontsize=17)
    ylim([-10,60])
    yticks([0,int(1./t_ref)/2,int(1./t_ref)])
    xticks( np.concatenate([rates_sigm[[0,-1]],xticks()[0]])[::2],
            np.concatenate([rates_sigm[[0,-1]],xticks()[0]])[::2]*1e9)
    xlim([rates_sigm[0],rates_sigm[-1]])
    legend(frameon=False, loc=2,prop={'size':15},numpoints=1,borderaxespad=0.1,handletextpad=0.2)
    et.savefig('sigmoid.png', format='png', dpi=300)

    et.globaldata.pool_out = pool_out
    et.globaldata.pool_out_sigm = pool_out_sigm
    et.globaldata.P = P
    et.save()

    show()


    
    #reconstruct pdf
    
    

