#!/bin/python
#-----------------------------------------------------------------------------
# File Name : funcs.py
# Purpose:
#
# Author: Emre Neftci
#
# Creation Date : 24-04-2013
# Last Modified : Thu 22 Jan 2015 12:00:19 PM PST
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
N_v = N_inputs =784
N_c = N_class = 200
N_h = N_hidden =500

n_c_unit =  N_c/n_classes

t_sim = 2000.0  #seconds
dcmt = 25 #duty cyle in multiples of t_ref

#----------------------------------------- Neuron parameters
t_ref = 0.004 # second
bias_input_rate = 300 #Hz
tau_rec = t_ref
tau_noise = .001
theta = .1 # volt
ctheta = 1. # volt
cm = 1e-12 #farad
sigma = 1.e-9 #amp
cal_i_lk = 0.0e-10
g_leak = 1e-9
cg_leak = 1e-9
dt = 0.00005
n_samples = t_sim/(dcmt*t_ref)+1
wnsigma = 4.24e-11

t_burn_percent = 10.
tau_learn = 0.01

deltaT = ((0.49-t_burn_percent/100)*dcmt*t_ref)

eta = 0.
epsilon = eta
epsilon_bias = eta

deltaA  = eta
deltaAbias = eta

def exp_prob_beta_gamma(dt, beta, g_leak, gamma):
    def func(V):
        return np.random.rand( len(V) ) < (1-np.exp(-np.exp(V*beta*g_leak+np.log(gamma))*float(dt)))
    return func

def test_exp(V, dt, beta, g_leak, gamma, t_ref):
    return (1-np.exp(-np.exp(V*beta*g_leak+np.log(gamma))*float(dt)))

defaultclock.dt = dt

         
