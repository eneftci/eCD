#!/bin/python
import numpy as np
import brian_no_units
from brian.globalprefs import *
set_global_preferences(useweave=True)
    
from brian import *

n_classes = 10
N_v = N_inputs =784
N_c = N_class = 40
N_h = N_hidden =500

n_c_unit =  N_c/n_classes

dcmt = 35 #duty cyle in multiples of t_ref

#----------------------------------------- Neuron parameters
t_ref = 0.004 # second
t_sim = dcmt*t_ref*10000  #seconds
bias_input_rate = 1000. #Hz
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
tau_learn = t_ref

deltaT = ((0.49-t_burn_percent/100)*dcmt*t_ref)

eta = 32e-3
epsilon = eta/beta*t_ref**2
epsilon_bias = eta/beta*t_ref*(1./bias_input_rate)

deltaA  = eta/beta/tau_learn/deltaT*t_ref**2/2
deltaAbias = eta/beta/tau_learn/deltaT*t_ref*(1./bias_input_rate)/2

i_inj = (- np.log(float(gamma))
         - np.log(float(t_ref))
         )/beta #amp
         
def exp_prob_beta_gamma(dt, beta, g_leak, gamma, t_ref):
    def func(V):
        return np.random.rand( len(V) ) < (1-np.exp(-np.exp(V*beta*g_leak+np.log(gamma))*float(dt)))
    return func

defaultclock.dt = dt
