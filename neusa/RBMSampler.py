#!/bin/python
#-----------------------------------------------------------------------------
# File Name : base.py
# Purpose: A class for sampling Restricted Boltzmann Machine using spiking neurons
#
# Author: Emre Neftci
#
# Creation Date : 14-01-2013
# Last Modified : Fri 03 May 2013 06:00:18 PM PDT
#
# Copyright : (c) Emre Neftci
# Licence : GPLv2
#----------------------------------------------------------------------------- 

import brian_no_units
import sys, time
from brian import *
from nnf.global_funcs import *
from funcs import ns_hist, plot_ns_hist
from nnf.nm.models import Instantiable
from base import WithNeuralParameters

class NeuralNetworkInterface(WithNeuralParameters):
    '''
    Neural Network virtual class
    '''

    def set_random_input(self, seed = 0):
        '''
        Set a random input
        '''
        np.random.seed(seed)
        input_v = np.zeros(self.N_i)
        nia = self.p['n_input_active']
        input_v[:nia] = self.p['avg_input_rate']
        np.random.shuffle(input_v)
        self.input_v = input_v

    def _reset_groups(self):
        self.neuron_groups = []
        self.input_groups = []
        self.update_funcs = []
        self.synapses = []
        self.mons = []

    def output_n(self, t_burne=0.1):
        '''
        Returns a vector of firing rates, starting from a given burn-in time 
        Arguments:
        *t_burne*: Burn-in time in seconds
        '''
        s_count = np.array(spike_histogram(self.basemon, t_burne, float(self.get_trun()))).T[1]
        return s_count

    def output_p(self, t_burne=0.1):
        '''
        Returns a vector of firing rates normaled to the sum of all activities, starting from a given burn-in time 
        Arguments:
        *t_burne*: Burn-in time in seconds
        '''
        return ns_hist(self)

    def create_input_vector(self):
        raise NotImplementedError

    def create_input_group(self):
        raise NotImplementedError

    def create_input_matrix(self):
        raise NotImplementedError

    def create_neuron_group(self):
        raise NotImplementedError

    def create_synapses(self):
        raise NotImplementedError

    def create_update_funcs(self):
        raise NotImplementedError

    def create_base_monitor(self):
        raise NotImplementedError

    def create_monitors(self, input_group, neuron_group):
        raise NotImplementedError

    def get_netlist(self):
        l_net = self.input_groups + self.neuron_groups + self.synapses + self.mons + self.update_funcs
        return l_net

    def instantiate(self, **parameters):
        self._reset_groups()
        self.update_parameters(parameters)
        self.create_input_vector()
        self.create_input_group()
        self.create_neuron_group()
        self.create_input_matrix()
        self.create_synapses()
        self.create_monitors(self.input_groups[0], self.neuron_groups[0])
        self.create_base_monitor()
        self.create_update_funcs()
        print 'instantiated'

    def run(self, **parameters):
        '''
        Run the neural sampler. Returns the vector of firing rates (using self.output_n() )

        Arguments:
        *parameters*: keyword parameters to override default parameters
        '''
        self.instantiate(**parameters)
        net = Network(self.get_netlist())
        np.random.seed(0)
        defaultclock.reinit()
        net.run(self.p['t_sim'])
        return self.output_n()

class ChainableNeuralSampler(WithNeuralParameters):

    def __init__(self, neural_sampler, N_inputs, N_units, *params, **kw_params):
        super(ChainableNeuralSampler, self).__init__(*params, **kw_params)
        self.N_i = N_inputs
        self.N_u = N_units
        self.ns = neural_sampler(N_inputs, N_units)
        self.chains = []

    def _link_groups(self):
        self.neuron_groups = self.ns.neuron_groups
        self.input_groups = self.ns.input_groups
        self.update_funcs = self.ns.update_funcs
        self.synapses = self.ns.synapses
        self.mons = self.ns.mons

    def propagate_parameters(self):
        self.ns.p.update(self.p)
        self.ns.set_weight(self.get_weight())
        self.ns.set_bias  (self.get_bias())
        self.ns.set_input (self.get_input())
        self.ns.N_u     = self.N_u
        self.ns.N_i     = self.N_i

    def chain(self, pre, synapse = 'I_AMPA'):
        psc_duration = self.p['t_ref']
        tau_noise = self.p['tau_noise']
        tau_input = self.p['tau_input']
        noise_input_rate = self.p['noise_input_rate']
        beta_fi = self.p['beta_fi']
        beta = self.p['beta']
        beta_fi = self.p['beta_fi']
        gamma = self.p['gamma']
        sigma = self.p['sigma']

        q_input = sigma * (2*tau_noise/noise_input_rate)**.5
        i_input = q_input/tau_noise
        w_input = i_input/beta_fi*tau_noise
        N_INPUT, N = self.N_i, self.N_u
        M_rec = (self.Wb/beta*coulomb)

        P2 = self.input_groups[1]
        neuron_group = self.neuron_groups[0]
        input_group = self.input_groups[0]

        if self.instantiated:
            print 'poo'
            S = Connection(pre.neuron_groups[0], self.neuron_groups[0], synapse)
            S.connect(pre.neuron_groups[0], self.neuron_groups[0], W = M_rec)
            self.synapses.append(S)
            self.chains.append(pre)
            print S
            print 'dob'
            for s in self.synapses:
                print s
        else:
            raise RuntimeError('Neural sampler must be instantiated first')

    def instantiate(self):        
        if not self.instantiated:
            self.propagate_parameters()
            self.ns.instantiate()
            self._link_groups()
            self.instantiated = True

    def get_netlist(self):
        net = []
        for link in self.chains:
            net += link.get_netlist()
        net += self.ns.get_netlist()
        return net

    def run(self):
        self.instantiate()
        net = Network(self.get_netlist())
        defaultclock.reinit()
        net.run(self.get_trun())

    def set_trun(self, value):
        self.ns.set_trun(value)

    def get_trun(self):
        return self.ns.get_trun()

    def update_default_parameters(self):
        super(ChainableNeuralSampler, self).update_default_parameters()
        self.p['noise_input_rate'] = 500*Hz
        self.p['beta'] = 5.15695881e+10
        self.p['gamma'] = 0.0685
        self.p['tau_noise'] = 1*ms
        self.p['tau_input'] = self.p['t_ref']
        self.p['theta'] = 1.0 * volt
        self.p['cm'] = 1e-12*farad
        self.p['beta_fi'] = 1./self.p['cm']/self.p['theta']
        self.p['sigma'] = .1e-9*amp





    
    
class NeuralSamplerBase(NeuralNetworkInterface):
    def __init__(self, N_inputs=32, N_units=32, *params, **kw_params):
        '''
        This class builds a neural sampler based on Buesing et al. 2011, without recurrent connections (W=0). 
        The neurons sample from the distribution:

        p(z) = exp(M.x + b)/Z , 
        
        where z is a binary random variable describing the activity of the neuron (or equivalently, the state of the PSP), x is the input vector (e.g. sensory data or data from previous layer), and b is a bias.

        Arguments:
        *N_inputs*: number of data units
        *N_units*: number of visible units
        *t_ref*: refractory period of underlying neuron model (in [s])      

        *params and **kw_params are used to override the default parameters.
        *kw_params* 
            *avg_input_rate*: input rate used to generate random input [Hz]
            *n_input_active*: number of active units in randomly generated inputs
            *dt*: simulation time step
        '''
        super(NeuralSamplerBase, self).__init__(*params, **kw_params)
        self.N_i = N_inputs
        self.N_u = N_units
        self._reset_groups()



    def create_input_vector(self):
        '''
        Create a random input if set_input was never called. Otherwise does nothing
        '''
        if not hasattr(self, 'input_v'):
            self.set_random_input()
        elif self.input_v is None:
            self.set_random_input()


    def create_input_group(self):
        '''
        Creates the population of neurons that stimulate the sampling neurons
        '''
        P = PoissonGroup(self.N_i, rates = self.input_v)
        self.input_groups.append(P)
        

    def create_input_matrix(self):
        '''
        Creates a random input matrix M, if set_weight was never called
        '''
        if not hasattr(self, 'Wb'):
            np.random.seed(214678)
            N_INPUT, N = self.N_i, self.N_u
            DUT = self.p['t_ref']/ms
            Mw = np.random.normal(0., 250.0, size=(N_INPUT,N))/N_INPUT/DUT
            self.Wb = Mw

    def create_neuron_group(self):
        '''
        Create the population of sampling neurons
        '''
        Q = NeuronGroup(self.N_u,
              model='''
                      v = Itot + b : 1
                      Itot = I1: 1
                      b : 1
                      I1 :1
                      ''',
              threshold = SimpleFunThreshold( exponential_prob(self.p['dt']/self.p['t_ref']), state = 'v'),
              refractory = self.p['t_ref'])

        Q.b = self.p['dc_input']
        self.neuron_groups.append(Q)

    def create_synapses(self):
        '''
        Create connections and synapses
        '''
        ig = self.input_groups[0]
        ng = self.neuron_groups[0]
        PSC_DURATION = self.p['t_ref']
        S1=Synapses(ig, ng, model='''
                        w : 1
                        I = w*heaviside(q,0) :1
                        dq/dt = -1/second : 1''',                        
                        pre='q=PSC_DURATION')

        N_INPUT, N = self.N_i, self.N_u
#        S1[:,:] = True
#        S1.w[:,:] = self.Wb[:,:]
#        S1.delay[:,:] = 0.*ms

        for i in range(N_INPUT):
            print i
            S1[i, :] = True
            S1.w[i, :] = self.Wb[i, :]
            S1.delay[0,0] = .0*ms

        self.synapses.append(S1)

        ng.I1 = S1.I

    def create_update_funcs(self):
        S1 = self.synapses[0]
        @network_operation
        def update_S1(when='after'):
            S1.q.data[S1.q.data<0.]=0.
        self.update_funcs.append(update_S1)

    def create_base_monitor(self):
        self.basemon = SpikeMonitor(self.neuron_groups[0])
        self.mons += [self.basemon]

    def create_monitors(self, input_group, neuron_group):
        MP  = PopulationRateMonitor(neuron_group, bin=1*second)
        Mv  = StateMonitor(neuron_group, 'v', record=True)
        MvI = StateMonitor(neuron_group, 'I1', record=True)
        Mg  = StateMonitor(self.synapses[0], 'I', record=range(self.N_u*self.N_i))
        Ms  = SpikeMonitor(neuron_group)
        Mo  = SpikeMonitor(input_group)
        self.mons += [MP, Mv, MvI, Mg, Ms, Mo]
        return self.mons

    def update_parameters(self, parameters):
        self.p.update(parameters)

    def update_default_parameters(self):
        super(NeuralSamplerBase, self).update_default_parameters()
        self.p['avg_input_rate'] = 20*Hz
        self.p['n_input_active'] = self.p['N_inputs']/4

class NeuralSamplerAdditivePSP(NeuralSamplerBase):
    '''
    NON FUNCTIONAL
    '''
    def create_synapses(self):
        ig = self.input_groups[0]
        ng = self.neuron_groups[0]
        PSC_DURATION = self.p['t_ref']
        S1=Synapses(ig, ng, model='''
                        w : 1
                        I = w*heaviside(q,0) :1
                        dq/dt = -1/second : 1''',                        
                        pre='q+=PSC_DURATION')

        N_INPUT, N = self.N_i, self.N_u
        for i in range(N_INPUT):
            for j in range(N):
                S1[j,i] = True
                S1.w[j,i] = self.Wb[j,i]
                S1.delay[0,0] = .0*ms

        self.synapses = self.synapses+[S1]

        ng.I1 = S1.I


class NeuralSamplerAlphaPSP(NeuralSamplerBase):
    '''
    NON FUNCTIONAL
    '''
    def create_synapses(self):
        ig = self.input_groups[0]
        ng = self.neuron_groups[0]
        PSC_DURATION = self.p['t_ref']
        TAU_INPUT = PSC_DURATION
        S1=Synapses(ig, ng, model='''
                        w : 1
                        dI/dt = -I/TAU_INPUT :1
                        ''',                        
                        pre='I += w')

        N_INPUT, N = self.N_i, self.N_u
        for i in range(N_INPUT):
            for j in range(N):
                S1[i,j] = True
                S1.w[j,i] = self.Wb[i,j]
                S1.delay[0,0] = .0*ms

        self.synapses =self.synapses + [S1]

        ng.I1 = S1.I

    def create_update_funcs(self):
        pass


class NeuralSamplerCLIF(NeuralSamplerBase):
    def __init__(self, N_inputs=32, N_units=32, *params, **kw_params):
        '''
        This class builds a neural sampler based on Buesing et al. 2011, without recurrent connections (W=0). 
        The neurons sample from the distribution:

        p(z) = exp(M.x + b)/Z , 
        
        where z is a binary random variable describing the activity of the neuron (or equivalently, the state of the PSP), x is the input vector (e.g. sensory data or data from previous layer), and b is a bias.

        Arguments:
        *N_inputs*: number of data units
        *N_units*: number of visible units
        *t_ref*: refractory period of underlying neuron model (in [s])      

        *params and **kw_params are used to override the default parameters.
        *kw_params* 
            *avg_input_rate*: input rate used to generate random input [Hz]
            *n_input_active*: number of active units in randomly generated inputs
            *dt*: simulation time step
            *noise_input_rate*:  (default 500*Hz)
            *beta*: (default 5.15695881e+10
            *gamma*:  (default 0.0685
            *tau_noise*:  (default 1*ms
            *tau_input*:  (default self.p['t_ref']
            *theta*:  (default 1.0 * volt
            *cm*:  (default 1e-12*farad
            *beta_fi*:  (default 1./Cm/theta)
            *sigma*:  (default .1e-9*amp)
        '''
        super(NeuralSamplerBase, self).__init__(
                N_inputs=N_inputs,
                N_units=N_units,
                *params,
                **kw_params)

    def update_default_parameters(self):
        super(NeuralSamplerCLIF, self).update_default_parameters()
        self.p['noise_input_rate'] = 500*Hz
        self.p['beta'] = 5.15695881e+10
        self.p['gamma'] = 0.0685
        self.p['tau_noise'] = 1*ms
        self.p['tau_input'] = self.p['t_ref']
        self.p['theta'] = 1.0 * volt
        self.p['cm'] = 1e-12*farad
        self.p['beta_fi'] = 1./self.p['cm']/self.p['theta']
        self.p['sigma'] = .1e-9*amp

    def create_neuron_group(self):
        import nnf.nm.spiking_model as sm
        import scipy.io

        #------------------------------------------------------------- Neuron parameters
        psc_duration = self.p['t_ref']
        tau_noise = self.p['tau_noise']
        tau_input = self.p['tau_input']
        noise_input_rate = self.p['noise_input_rate']
        beta = self.p['beta']
        beta_fi = self.p['beta_fi']
        gamma = self.p['gamma']
        sigma = self.p['sigma']

        i_inj = ( self.p['dc_input']
                - log(float(self.p['gamma']))
                - log(float(self.p['t_ref']))
                )/self.p['beta']*amp
         #sigma is std
        q_input = sigma * (2*tau_noise/noise_input_rate)**.5
        i_input = q_input/tau_noise
        w_input = i_input/beta_fi*tau_noise
        N_INPUT, N = self.N_i, self.N_u

        eqs_str = '''
        dI_NMDA/dt = -I_NMDA/tau_NMDA : amp
        dI_AMPA/dt = -I_AMPA/tau_AMPA :amp
        dv/dt = (I_tot)/Cm :volt
        I_tot = -I_lk + I_inj + I_AMPA + I_NMDA :amp
        I_inj :amp
        '''
        eqs = Equations(eqs_str, 
                Cm = 1e-12*farad,
                I_lk = w_input*beta_fi*noise_input_rate + 1e-10*amp,
                tau_NMDA = tau_noise,
                tau_AMPA = tau_input)
                                                   
        P2=PoissonGroup(N, rates = noise_input_rate)

        neuron_group = NeuronGroup(\
                self.N_u,
                model = eqs,
                threshold = 'v>1*volt',
                refractory = self.p['t_ref'],
                reset=0*volt
                )
        #set bias
        neuron_group.I_inj =  i_inj

        self.neuron_groups += [neuron_group]
        self.input_groups += [P2]

    def create_synapses(self):
        psc_duration = self.p['t_ref']
        tau_noise = self.p['tau_noise']
        tau_input = self.p['tau_input']
        noise_input_rate = self.p['noise_input_rate']
        beta_fi = self.p['beta_fi']
        beta = self.p['beta']
        beta_fi = self.p['beta_fi']
        gamma = self.p['gamma']
        sigma = self.p['sigma']

        q_input = sigma * (2*tau_noise/noise_input_rate)**.5
        i_input = q_input/tau_noise
        w_input = i_input/beta_fi*tau_noise
        N_INPUT, N = self.N_i, self.N_u
        M_rec = (self.Wb/beta*coulomb)

        P2 = self.input_groups[1]
        neuron_group = self.neuron_groups[0]
        input_group = self.input_groups[0]

        S1=Connection(P2, neuron_group, 'I_NMDA')
        S1.connect_one_to_one(P2, neuron_group, weight = i_input)
        S2=Connection(input_group, neuron_group, 'I_AMPA')
        S2.connect(input_group, neuron_group, W = M_rec)
        self.synapses = self.synapses + [S1, S2]

    def create_update_funcs(self):
        ng = self.neuron_groups[0]
        @network_operation(clock=defaultclock)
        def update_mpot(when='after'):
            ng.v[ng.v<=0.0*volt]=0.0*volt
        self.update_funcs += [update_mpot]

    def create_monitors(self, input_group, neuron_group):
        MP  = PopulationRateMonitor(neuron_group, bin = 1*second)
        Mv  = StateMonitor(neuron_group, 'v', record = True)
        MvI = StateMonitor(neuron_group, 'I_AMPA', record = True)
        Mg  = StateMonitor(neuron_group, 'I_tot', record = True)
        Ms  = SpikeMonitor(neuron_group)
        Mo  = SpikeMonitor(input_group)
        self.mons += [MP, Mv, MvI, Mg, Ms, Mo]

#class NeuralSamplerCLIFLearning(NeuralSamplerCLIF):
#
#    def create_neuron_group(self):
#        import nnf.nm.spiking_model as sm
#        import scipy.io
#
#        #------------------------------------------------------------- Neuron parameters
#        psc_duration = self.p['t_ref']
#        tau_noise = self.p['tau_noise']
#        tau_input = self.p['tau_input']
#        noise_input_rate = self.p['noise_input_rate']
#        beta = self.p['beta']
#        beta_fi = self.p['beta_fi']
#        gamma = self.p['gamma']
#        sigma = self.p['sigma']
#
#        i_inj = ( self.p['dc_input']
#                - log(float(self.p['gamma']))
#                - log(float(self.p['t_ref']))
#                )/self.p['beta']*amp
#         #sigma is std
#        q_input = sigma * (2*tau_noise/noise_input_rate)**.5
#        i_input = q_input/tau_noise
#        w_input = i_input/beta_fi*tau_noise
#        N_INPUT, N = self.N_i, self.N_u
#
#        eqs_str = '''
#        dI_NMDA/dt = -I_NMDA/tau_NMDA : amp
#        dI_AMPA/dt = -I_AMPA/tau_AMPA : amp
#        dv/dt = (I_tot)/Cm :volt
#        I_tot = -I_lk + I_inj + I_AMPA + I_NMDA :amp
#        I_inj :amp
#        dI_recon/dt = -I_recon/tau_AMPA : amp
#        '''
#        eqs = Equations(eqs_str, 
#                Cm = 1e-12*farad,
#                I_lk = w_input*beta_fi*noise_input_rate + 1e-10*amp,
#                tau_NMDA = tau_noise,
#                tau_AMPA = tau_input)
#                                                   
#        P2=PoissonGroup(N, rates = noise_input_rate)
#        P3=PoissonGroup(N, rates = noise_input_rate)
#        bias_v=PoissonGroup(1, rates = noise_input_rate)
#        bias_h=PoissonGroup(1, rates = noise_input_rate)
#
#        neuron_group = NeuronGroup(\
#                self.N_u,
#                model = eqs,
#                threshold = 'v>1*volt',
#                refractory = self.p['t_ref'],
#                reset=0*volt
#                )
#
#        neuron_group_pre = NeuronGroup(\
#                self.N_i,
#                model = eqs,
#                threshold = 'v>1*volt',
#                refractory = self.p['t_ref'],
#                reset=0*volt
#                )
#        #set bias
#        neuron_group.I_inj =  i_inj
#        neuron_group_pre.I_inj =  i_inj
#
#        self.neuron_groups += [neuron_group, neuron_group_pre]
#        self.input_groups += [P2,P3, bias_v, bias_h]
#
#    def create_synapses(self):
#        psc_duration = self.p['t_ref']
#        tau_noise = self.p['tau_noise']
#        tau_input = self.p['tau_input']
#        noise_input_rate = self.p['noise_input_rate']
#        beta_fi = self.p['beta_fi']
#        beta = self.p['beta']
#        beta_fi = self.p['beta_fi']
#        gamma = self.p['gamma']
#        sigma = self.p['sigma']
#
#        q_input = sigma * (2*tau_noise/noise_input_rate)**.5
#        i_input = q_input/tau_noise
#        w_input = i_input/beta_fi*tau_noise
#        N_INPUT, N = self.N_i, self.N_u
#        M_rec = (self.Wb/beta*coulomb)
#
#        P2 = self.input_groups[1]
#        P3 = self.input_groups[2]
#        bias_v = self.input_groups[3]
#        bias_h = self.input_groups[4]
#
#        neuron_group = self.neuron_groups[0]
#        neuron_group_pre = self.neuron_groups[1]
#        input_group = self.input_groups[0]
#
#        S0=Connection(P3, neuron_group_pre, 'I_NMDA')
#        S0.connect_one_to_one(P3, neuron_group_pre, weight = i_input)
#        S1=Connection(P2, neuron_group, 'I_NMDA')
#        S1.connect_one_to_one(P2, neuron_group, weight = i_input)
#        S2=Connection(input_group, neuron_group_pre, 'I_AMPA')
#        S2.connect_one_to_one(input_group, neuron_group_pre, weight = 5./beta*coulomb)
#
#        S3 = Synapses(neuron_group_pre, 
#                     neuron_group, 
#                     model = 'w : amp',
#                     pre = 'I_AMPA_post += w',
#                     post = '''I_recon_pre += w
#                               w += 0.002*(I_AMPA_pre - I_recon_pre)'''
#                     )
#
#        S4=Connection(bias_h, neuron_group, 'I_AMPA')
#        S4.connect(bias_h, neuron_group, W = self.get_bias().reshape(-1,1)/beta*coulomb)
#
#        for i in range(len(neuron_group_pre)):
#            S3[i, :] = True
#            S3.w[i, :] = np.random.normal(0, 1e-16, size = M_rec[i,:].shape)
#            S3.delay[0,0] = .0*ms
#
#        self.synapses = self.synapses + [S0,S1, S2, S3, S4]
#
#    def create_update_funcs(self):
#        for ng in self.neuron_groups:
#            @network_operation(clock=defaultclock)
#            def update_mpot(when='after'):
#                ng.v[ng.v<=0.0*volt]=0.0*volt
#            self.update_funcs += [update_mpot]
#
#    def create_monitors(self, input_group, neuron_group):
#        neuron_group_pre = self.neuron_groups[1]
#        MP  = PopulationRateMonitor(neuron_group, bin = 1*second)
#        Mv  = StateMonitor(neuron_group_pre, 'v', record = True)
#        MvI = StateMonitor(neuron_group_pre, 'I_AMPA', record = True)
#        Mg  = StateMonitor(neuron_group_pre, 'I_recon', record = True)
#        M = StateMonitor(self.synapses[3],'w',record=True, timestep=100)
#        Ms  = SpikeMonitor(neuron_group_pre)
#        Mo  = SpikeMonitor(neuron_group)
#        self.mons += [MP, Mv, MvI, Mg, M, Ms, Mo]
#

class NeuralSamplerCLIFSpikingBias(NeuralSamplerCLIF):
    def create_neuron_group(self):
        import nnf.nm.spiking_model as sm
        import scipy.io

        #------------------------------------------------------------- Neuron parameters
        psc_duration = self.p['t_ref']
        tau_noise = self.p['tau_noise']
        tau_input = self.p['tau_input']
        noise_input_rate = self.p['noise_input_rate']
        beta = self.p['beta']
        beta_fi = self.p['beta_fi']
        gamma = self.p['gamma']
        sigma = self.p['sigma']
        #Missing Gamma?!?
        i_inj = (- log(float(self.p['t_ref']))
                )/self.p['beta']*amp

        q_input = sigma * (2*tau_noise/noise_input_rate)**.5
        i_input = q_input/tau_noise
        w_input = i_input/beta_fi*tau_noise
        N_INPUT, N = self.N_i, self.N_u

        eqs_str = '''
        dI_NMDA/dt = -I_NMDA/tau_NMDA : amp
        dI_AMPA/dt = -I_AMPA/tau_AMPA :amp
        dv/dt = (I_tot)/Cm :volt
        I_tot = -I_lk + I_inj + I_AMPA + I_NMDA :amp
        I_inj :amp
        '''
        eqs = Equations(eqs_str, 
                Cm = 1e-12*farad,
                I_lk = w_input*beta_fi*noise_input_rate + 1e-10*amp,
                tau_NMDA = tau_noise,
                tau_AMPA = tau_input)
                                                   
        P2=PoissonGroup(N, rates = noise_input_rate)

        neuron_group = NeuronGroup(\
                self.N_u,
                model = eqs,
                threshold = 'v>1*volt',
                refractory = self.p['t_ref'],
                reset=0*volt
                )
        #set bias
        neuron_group.I_inj = i_inj 
        I_bias = (self.p['dc_input'] - log(float(self.p['gamma'])))/self.p['beta']*amp
        idxpos = I_bias>=0
        idxneg = I_bias<0
        self.wpos = wpos = np.mean(I_bias[idxpos])/500
        self.wneg = wneg = np.mean(I_bias[idxneg])/500

        self.ratepos = ratepos = np.zeros(N)
        ratepos[idxpos] = I_bias[idxpos]/wpos
        self.rateneg = rateneg = np.zeros(N)
        rateneg[idxneg] = I_bias[idxneg]/wneg
        P3 = PoissonGroup(N, rates = ratepos)
        P4 = PoissonGroup(N, rates = rateneg)
        self.neuron_groups += [neuron_group]
        self.input_groups += [P2, P3, P4]

    def create_synapses(self):
        psc_duration = self.p['t_ref']
        tau_noise = self.p['tau_noise']
        tau_input = self.p['tau_input']
        noise_input_rate = self.p['noise_input_rate']
        beta_fi = self.p['beta_fi']
        beta = self.p['beta']
        beta_fi = self.p['beta_fi']
        gamma = self.p['gamma']
        sigma = self.p['sigma']

        q_input = sigma * (2*tau_noise/noise_input_rate)**.5
        i_input = q_input/tau_noise
        w_input = i_input/beta_fi*tau_noise
        N_INPUT, N = self.N_i, self.N_u
        M_rec = (self.Wb/beta*coulomb)

        P2 = self.input_groups[1]
        P3 = self.input_groups[2]
        P4 = self.input_groups[3]
        neuron_group = self.neuron_groups[0]
        input_group = self.input_groups[0]

        S1=Connection(P2, neuron_group, 'I_NMDA')
        S1.connect_one_to_one(P2, neuron_group, weight = i_input)
        S2=Connection(input_group, neuron_group, 'I_AMPA')
        S2.connect(input_group, neuron_group, W = M_rec)
        S3=Connection(P3, neuron_group, 'I_AMPA')
        S3.connect_one_to_one(P3, neuron_group, weight = self.wpos/tau_input)
        S4=Connection(P4, neuron_group, 'I_AMPA')
        S4.connect_one_to_one(P4, neuron_group, weight = self.wneg/tau_input)

        self.synapses = self.synapses + [S1, S2, S3, S4]


class NeuralSamplerCLIFNoiseless(NeuralSamplerCLIF):
    #
    #
    #NON-FUNCTIONAL: Transfer function is not corrected to noiseless case!
    #
    def create_neuron_group(self):
        import nnf.nm.spiking_model as sm
        import scipy.io

        #------------------------------------------------------------- Neuron parameters
        psc_duration = self.p['t_ref']
        tau_noise = self.p['tau_noise']
        tau_input = self.p['tau_input']
        beta = self.p['beta']
        gamma = self.p['gamma']
        beta_fi = self.p['beta_fi']


        i_inj = ( self.p['dc_input']
                - log(float(self.p['gamma']))
                - log(float(self.p['t_ref']))
                )/self.p['beta']*amp

        eqs_str = '''
        dI_NMDA/dt = -I_NMDA/tau_NMDA : amp
        dI_AMPA/dt = -I_AMPA/tau_AMPA :amp
        dv/dt = (I_tot)/Cm :volt
        I_tot = -I_lk + I_inj + I_AMPA + I_NMDA :amp
        I_inj :amp
        '''
        eqs = Equations(eqs_str, 
                Cm = 1e-12*farad,
                I_lk = 0*amp,
                tau_NMDA = tau_noise,
                tau_AMPA = tau_input)
                                                   
        neuron_group = NeuronGroup(\
                self.N_u,
                model = eqs,
                threshold = 'v>1*volt',
                refractory = self.p['t_ref'],
                reset=0*volt
                )
        #set bias
        neuron_group.I_inj =  i_inj

        self.neuron_groups += [neuron_group]

    def create_synapses(self):
        psc_duration = self.p['t_ref']
        tau_noise = self.p['tau_noise']
        tau_input = self.p['tau_input']
        noise_input_rate = self.p['noise_input_rate']
        beta_fi = self.p['beta_fi']
        beta = self.p['beta']
        beta_fi = self.p['beta_fi']
        gamma = self.p['gamma']
        sigma = self.p['sigma']

        M_rec = (self.Wb/beta*coulomb)

        neuron_group = self.neuron_groups[0]
        input_group = self.input_groups[0]

        S2=Connection(input_group, neuron_group, 'I_AMPA')
        S2.connect(input_group, neuron_group, W = M_rec)

        self.synapses += [S2]



class NeuralSamplerCLIFMismatched(NeuralSamplerCLIF):
    def create_neuron_group(self):
        import nnf.nm.spiking_model as sm
        import scipy.io

        #------------------------------------------------------------- Neuron parameters
        psc_duration = self.p['t_ref']
        tau_noise = self.p['tau_noise']
        tau_input = self.p['tau_input']
        noise_input_rate = self.p['noise_input_rate']
        beta = self.p['beta']
        beta_fi = self.p['beta_fi']
        gamma = self.p['gamma']
        sigma = self.p['sigma']

        i_inj = ( self.p['dc_input']
                - log(float(self.p['gamma']))
                - log(float(self.p['t_ref']))
                )/self.p['beta']*amp
                      
        q_input = sigma * (2*tau_noise/noise_input_rate)**.5
        i_input = q_input/tau_noise
        w_input = i_input/beta_fi*tau_noise
        N_INPUT, N = self.N_i, self.N_u

        eqs_str = '''
        dI_NMDA/dt = -I_NMDA/tau_NMDA : amp
        dI_AMPA/dt = -I_AMPA/tau_AMPA :amp
        dv/dt = (I_tot)/Cm :volt
        I_tot = -I_lk + I_inj + I_AMPA + I_NMDA :amp
        I_inj :amp
        '''
        eqs = Equations(eqs_str, 
                Cm = 1e-12*farad,
                I_lk = w_input*beta_fi*noise_input_rate + 1e-10*amp,
                tau_NMDA = tau_noise,
                tau_AMPA = tau_input)
                                                   
        P2=PoissonGroup(N, rates = noise_input_rate)

        neuron_group = NeuronGroup(\
                self.N_u,
                model = eqs,
                threshold = 'v>1*volt',
                refractory = np.random.normal(self.p['t_ref'], self.p['t_ref']*self.p['delta_t_ref'], size = self.N_u),
                reset=0*volt
                )
        #set bias
        neuron_group.I_inj =  i_inj

        self.neuron_groups += [neuron_group]
        self.input_groups += [P2]

    def update_default_parameters(self):
        super(NeuralSamplerCLIFMismatched, self).update_default_parameters()
        self.p['delta_t_ref'] = 0.3 


class VBMSampler(Instantiable):
    default_parameters = {}
    default_parameters['N_i'] = 32
    default_parameters['N_v'] = 32

    def __init__(self, neural_sampler, N_data, N_visible, W, b, inp = None, *params, **kw_params):
        '''
        Fully visible Bolztmann machine sampler. This is essentially a wrapper around the neural sampler.

        Arguments:
        *neural_sampler*: A NeuralSamplerBase class
        *N_data*: dimension of the data vector
        *N_visible*: number of visible units
        *W*: input weight matrix
        *b*: bias of visible units
        *inp*: input vector x
        '''
        super(VBMSampler, self).__init__(*params, **kw_params)
        self.N_data = self.p['N_i'] = N_data
        self.N_visible = self.p['N_v'] = N_visible
        self.ns_class = neural_sampler
        self.ns_v = self.ns_class(
                N_inputs = self.p['N_i'],
                N_units = self.p['N_v'])
        self.W_v = W
        self.b_v = b
        self.input_v = inp


    def _set_weight_d_to_v(self, W):

        assert W.shape[0] == self.p['N_i']
        assert W.shape[1] == self.p['N_v']

        self.ns_v.p['N_inputs'] = self.p['N_i']
        self.ns_v.p['N_units'] = self.p['N_v']

        self.ns_v.Wb = W

    def _set_bias_d(self, b):
        self.ns_v.p['dc_input'] = b

    def _set_input(self, inp = None, normalize = None):            
        if inp is not None:
            self.ns_v.input_v = self.inp
        else:
            self.ns_v.input_v = self.input_v
        assert self.ns_v.input_v is not None

    def instantiate(self):
        self._set_weight_d_to_v(self.W_v)
        self._set_bias_d(self.b_v)
        self._set_input()

    def run(self, t_run = None):
        '''
        Run for time t_run
        '''
        self.instantiate()
        self.ns_v.set_trun(t_run)
        self.ns_v.run()
        return self.ns_v.output_n()
 

class RBMSampler(Instantiable):    
    default_parameters = {}
    default_parameters['trun'] = 5.0

    def __init__(self,
            neural_sampler,
            N_visible, N_hidden,
            W_h,
            b_v, b_h,
            inp,
            *params, **kw_params):
        '''
        Restricted Boltzmann Machine with one visible and one hidden layer.

        (The architecture of this class does *not* allow to combine layers.)

        Arguments:
        *neural_sampler*: A NeuralSamplerBase class that will be used to sample from each RBM layer
        *N_visible*: number of visible units
        *N_hidden*: number of hidden units
        *W_h*: visible to hidden weight matrix. Its transpose will be used for the hidden to visible weight matrix
        *b_h*: bias of hidden units
        *b_v*: bias of visible units
        *inp*: input

        Provides:
        *state*: a dictionary contaning the state of each layer if the corresponding neural sampler has been run
        '''

        super(RBMSampler, self).__init__(*params, **kw_params)
        self.N_visible = self.p['N_v'] = N_visible
        self.N_hidden = self.p['N_h'] = N_hidden

        from collections import OrderedDict
        self.ns_class = neural_sampler
        #NS Indexes correspond to the target
        self.ns = OrderedDict()
        self.state = {}
        self.b = {}
        self.W = {}
        self.b['v'] = b_v
        self.b['h'] = b_h
        self.W['h'] = W_h
        self.W['v'] = W_h.T
        self.init_k = 'v' #This layer will be clamped to input_v
        self.input_v = inp

    def update_default_parameters(self):
        super(RBMSampler, self).update_default_parameters()
        self.p['N_v'] = 784
        self.p['N_h'] = 500

    def instantiate(self):
        ### Order in which self.ns is populated is the order in which the chain of layers is run
        #neural sampler visible -> hidden
        self._instantiate_ns('h', N_inputs = self.p['N_v'], N_units = self.p['N_h'])
        #neural sampler hidden -> visible
        self._instantiate_ns('v', N_inputs = self.p['N_h'], N_units = self.p['N_v'])

    def _instantiate_ns(self, key, **kw_params):
        ns_inst = self.ns_class(**kw_params)
        self.ns[key] = ns_inst
        self._set_weight(key)
        self._set_bias(key)
        self.state[key] = np.zeros(len(ns_inst))
        ns_inst.instantiate()
        return ns_inst

    def set_trun(self, t=None):
        '''
        Set run time of all the neural samplers
        '''
        if t is not None:
            self.p['trun'] = t
        else:
            t = self.p['trun']
        print 'DBG: setting sim time to {0}'.format(t)
        for k, ns in self.ns.iteritems():
            ns.set_trun(t)

    def _set_bias(self, key):
        self.ns[key].set_bias(self.b[key])

    def _set_weight(self, key):
        self.ns[key].set_weight(self.W[key])

    def _set_input(self, key, val):
        self.ns[key].set_input(val)

    def clamp(self, state_id, value):
        '''
        Clamp the state state_id of the RBM to value.

        Arguments:
        *state_id*: Index of the RBM layer. In this case either 'h' or 'v', resp. corresponding to hidden or visible
        *value*: an iterable of the same dimension as the layer that is being clamped
        '''
        #self.input_v*50.
        print 'Clamping {0}'.format(state_id)
        self.state[state_id][:] = value
        return self.state[state_id]

    def sample(self, src, tgt):
        '''
        Sample from target (tgt) given source (src)

        Arguments:
        *src*: Index of the source RBM layer. Its state will be used to sample from tgt
        *src*: Index of target RBM layer. The corresponding neural sampler will be run, and the tgt state will be set
        '''
        print 'Sampling {0} from {1}'.format(tgt, src)
        self._set_input(tgt, self.state[src])
        s = self.ns[tgt].run()
        self.state[tgt] = s
        return s

    def _get_chain(self):
        '''
        This function provides the RBMsampler with the order in which the neural samplers will be run.
        The architecture of this class does *not* allow to combine layers
        '''
        chain = []
        idxs = [self.init_k]
        idxs += self.ns.keys()
        for i in xrange(1, len(idxs)):
            chain.append((idxs[i-1], idxs[i]))
        return chain

    def run(self):
        self.instantiate()
        self.set_trun()
        chain = self._get_chain()
        init_k = self.init_k
        self.clamp(init_k, self.input_v*(1./self.ns[init_k].p['t_ref']))
        for c in chain:
            self.sample(c[0], c[1])
        return self.state



class RBMClassifier(RBMSampler):
    
    def __init__(self,
            neural_sampler,
            N_visible, N_hidden, N_labels,
            W_h, W_c,
            b_v, b_h, b_c,
            inp,
            *params, **kw_params):
        '''
        Restricted Boltzmann Machine with one visible, one hidden and one class layer for a classification task

        Arguments:
        *neural_sampler*: A NeuralSamplerBase class that will be used to sample from each RBM layer
        *N_visible*: number of visible units
        *N_hidden*: number of hidden units
        *W_h*: visible to hidden weight matrix. Its transpose will be used for the hidden to visible weight matrix
        *W_c*: hidden to class weight matrix. Its transpose will be used for the class to visible weight matrix
        *b_h*: bias of hidden units
        *b_v*: bias of visible units
        *b_c*: bias of class units
        *inp*: input

        Provides:
        *state*: a dictionary contaning the state of each layer if the corresponding neural sampler has been run
        '''

        super(RBMClassifier, self).__init__(neural_sampler,
            N_visible, N_hidden,
            W_h,
            b_v, b_h,
            inp, 
            *params, **kw_params)

        self.N_labels = self.p['N_c'] = N_labels
        self.W['c'] = W_c
        self.b['c'] = b_c

    def update_default_parameters(self):
        super(RBMClassifier, self).update_default_parameters()
        self.p['N_c'] = 10

    def instantiate(self):
        super(RBMClassifier, self).instantiate()
        self._instantiate_ns('c', N_inputs = self.p['N_h'], N_units = self.p['N_c'])

    def _get_chain(self):            
        k = [self.init_k] 
        k += self.ns.keys()[:-1]
        chain = [(k[0], k[1]),(k[1],'c')]
        return chain

class RBMClassifierReconstructor(RBMClassifier):
    def __init__(self,
            neural_sampler,
            N_visible, N_hidden, N_labels,
            W_h, W_c,
            b_v, b_h, b_c,
            inp,
            *params, **kw_params):

        super(RBMClassifierReconstructor, self).__init__(neural_sampler,
             N_visible, N_hidden, N_labels,
             W_h, W_c,
             b_v, b_h, b_c,
             inp, 
             *params, **kw_params)

        self.W['ch'] = W_c.T
        self.b['ch'] = b_h
        self.W['hv'] = W_h.T
        self.b['hv'] = b_v

    def instantiate(self):
        super(RBMClassifierReconstructor, self).instantiate()
        self._instantiate_ns('ch', N_inputs = self.p['N_c'], N_units = self.p['N_h'])
        self._instantiate_ns('hv', N_inputs = self.p['N_h'], N_units = self.p['N_v'])
        
    def _get_chain(self):            
        chain = [('v', 'h'),('h','c'),('c','ch'),('ch','hv')]
        return chain





class ChainedRBMClassifier(RBMClassifier):
#
#    def __init__(self, *params, **kw_params):
#        super(ChainedRBMClassifier, self).__init__(*params, **kw_params)

    def instantiate(self):
        super(ChainedRBMClassifier, self).instantiate()
        chain = self._get_chain()
        for c in chain:
            self.ns[c[1]].chain(self.ns[c[0]])


    def sample(self):
        '''
        Sample from target (tgt) given source (src)

        Arguments:
        *src*: Index of the source RBM layer. Its state will be used to sample from tgt
        *src*: Index of target RBM layer. The corresponding neural sampler will be run, and the tgt state will be set
        '''
        chain = self._get_chain()
        print 'Sampling {0} from {1}'.format(chain[-1][-1], chain[0][0])
        print 'DBUG'
        for h in self.ns[chain[-1][-1]].get_netlist():
            print h
        s = self.ns[chain[-1][-1]].run()
        self.state[chain[-1][-1]] = s

    def _instantiate_ns(self, key, *params, **kw_params):
        ns_inst = ChainableNeuralSampler(self.ns_class, *params, **kw_params)
        self.ns[key] = ns_inst
        self._set_weight(key)
        self._set_bias(key)
        if key == self._get_chain()[0][0]: self._set_input(key, self.input_v*(1./self.ns['h'].p['t_ref']))
        chain = self._get_chain()
        self.state[key] = np.zeros(len(ns_inst))
        ns_inst.instantiate()
        return ns_inst

    def _get_chain(self):            
        chain = [('h','c')]
        return chain

    def run(self):
        self.sample()

def sigmoid(x):
    return 1./(1+np.exp(-x))

#RBM samplers, should go to RBMSampler?
class RBMGibbsSampler:
    def __init__(self, W, bv, bh):
        self.W = W
        self.bv = bv
        self.bh = bh

    def onestep(self, vsample):
        hmean   = sigmoid( np.dot( vsample, self.W) + self.bh)
        hsample = np.random.binomial( n = 1, p = hmean)
        vmean = sigmoid( np.dot( hsample, self.W.T) + self.bv)
        return np.random.binomial( n = 1, p = vmean), hsample

    def empty_chain(self, nsteps = 1000):
        return  np.empty([nsteps, len(self.bv)]), np.empty([nsteps, len(self.bh)])

    def manysteps(self, v, nsteps = 1000):
        '''
        *v*: initial visible sample 
        '''
        lv, lh = self.empty_chain(nsteps)
        for i in xrange(nsteps):
            lv[i,:], lh[i,:] = self.onestep(v)
            v = lv[i,:]
        return lv, lh

    def sample(self, v, nsteps = 1000):
        '''
        *v*: initial visible sample 
        '''
        lv, lh = self.manysteps(v, nsteps)
        v = lv.mean(axis=0)
        h = lh.mean(axis=0)
        pv = v/v.sum()
        ph = h/h.sum()
        return pv, ph
