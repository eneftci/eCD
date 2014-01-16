#!/bin/python
#-----------------------------------------------------------------------------
# File Name : kldivergence_parameters.py
# Purpose:
#
# Author: Emre Neftci
#
# Creation Date : 24-04-2013
# Last Modified : Tue 26 Nov 2013 10:00:40 AM PST
#
# Copyright : (c) 
# Licence : GPLv2
#----------------------------------------------------------------------------- 
import numpy as np
import brian_no_units
from brian.globalprefs import *
set_global_preferences(useweave=True)
    
from brian import *

n_classes = 10
N_v = N_inputs =5
N_c = N_class = 0
N_h = N_hidden =5

n_c_unit =  N_c/n_classes

t_sim = 1000.  #seconds
dcmt = 5000000 #duty cyle in multiples of t_ref

#----------------------------------------- Neuron parameters
t_ref = 0.004 # second
bias_input_rate = 1000 #Hz
beta = 2.04371561e+09
gamma = np.exp(9.08343441e+00)
tau_noise = .001
tau_rec = t_ref
theta = .1 # volt
cm = 1e-12 #farad
beta_fi = 1./cm/theta
sigma = 1.e-9 #amp
cal_i_lk = 0.0e-10
g_leak = 1e-9
dt = 0.00005
n_samples = t_sim/(dcmt*t_ref)+1
wnsigma = 4.24e-11

t_burn_percent = 10.
tau_learn = 0.004

deltaT = ((0.49-t_burn_percent/100)*dcmt*t_ref)

eta = 0e-3
epsilon = eta/beta*t_ref**2*(dcmt*t_ref)/deltaT
epsilon_bias = eta/beta*t_ref*(1./bias_input_rate)*(dcmt*t_ref)/deltaT

deltaA  = eta/beta/tau_learn*(dcmt*t_ref)/deltaT*t_ref**2/2
deltaAbias = eta/beta/tau_learn*(dcmt*t_ref)/deltaT*t_ref*(1./bias_input_rate)/2

i_inj = (- np.log(float(gamma))
         - np.log(float(t_ref))
         )/beta #amp
         
         
def exp_prob_beta_gamma(dt, beta, g_leak, gamma, t_ref):
    def func(V):
        return np.random.rand( len(V) ) < (1-np.exp(-np.exp(V*beta*g_leak+np.log(gamma))*float(dt)))
    return func


defaultclock.dt = dt


Nruns = 48

         
