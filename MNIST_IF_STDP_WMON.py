#!/bin/python
#-----------------------------------------------------------------------------
# File Name : mnist_feedback.py
# Purpose:
#
# Author: Emre Neftci
#
# Creation Date : 25-04-2013
# Last Modified : Sun 24 Nov 2013 01:36:11 PM PST
#
# Copyright : (c) 
# Licence : GPLv2
#----------------------------------------------------------------------------- 
from common import *

def main(Whv, b_v, b_c, b_h, Id, dorun = True, monitors=True, display=False, mnist_data = None):
    defaultclock.reinit()
    b_init = np.concatenate([b_v, b_c, b_h])
    netobjs = []
    #------------------------------------------ Neuron Groups
    print "Creating equation"
    
    eqs_v = Equations(eqs_str_lif_wnrd, 
            Cm = 1e-12*farad,
            I_inj = i_inj,
            g = g_leak,
            sigma = wnsigma,
            tau_rec = tau_rec)
                                               
    eqs_h = Equations(eqs_str_lif_wnr, 
            Cm = 1e-12*farad,
            I_inj = i_inj,
            g = g_leak,
            sigma = wnsigma,
            tau_rec = tau_rec)
    
    print "Creating Population"
    neuron_group_rvisible = NeuronGroup(\
            N_v+N_c,
            model = eqs_v,
            threshold = 'v>theta*volt',
            refractory = t_ref,
            reset = 0*volt
            )
    
    neuron_group_rhidden = NeuronGroup(\
            N_h,
            model = eqs_h,
            threshold = 'v>theta*volt',
            refractory = t_ref,
            reset = 0*volt
            )


    netobjs += [neuron_group_rvisible, neuron_group_rhidden]
    
    #--------------------------- Custom Network Operations
#    @network_operation(clock = defaultclock)
#    def update_mpot(when='after'):
#        neuron_group_rvisible.v[neuron_group_rvisible.v<=0.0*volt]=0.0*volt
#        neuron_group_rhidden.v[neuron_group_rhidden.v<=0.0*volt]=0.0*volt
#    netobjs.append(update_mpot)   
    
    #Bias group
    Bv = PoissonGroup(N_v+N_c, rates = bias_input_rate) #Noise injection to h
    Bh = PoissonGroup(N_h, rates = bias_input_rate) #Noise injection to h
    
    netobjs+=[Bv,Bh]
    
    #---------------------- Initialize State Variables
    neuron_group_rvisible.I_d = 0.
    
    
    #---------------------- Connections and Synapses
    #Bias units    
    Sbv = Synapses(Bv, neuron_group_rvisible, 
              model='''Afre : 1
                     Afost : 1
                     g : 1
                     w : 1''',
               pre ='''Afre=Afre*np.exp((lastupdate-t)/tau_learn)
                      Afost=Afost*np.exp((lastupdate-t)/tau_learn)
                      Afre+=deltaAbias
                      w=w+g*Afost
                      I_rec_post+= w''',
               post='''Afre=Afre*np.exp((lastupdate-t)/tau_learn)
                      Afost=Afost*np.exp((lastupdate-t)/tau_learn)
                      Afost+=deltaAbias
                      w=w+g*Afre'''
                      )
    Sbv[:,:] = 'i==j'
    Sbv.w[:] = np.concatenate([b_v,b_c])/beta/bias_input_rate/tau_rec
    
    Sbh = Synapses(Bh, neuron_group_rhidden, 
              model='''Afre : 1
                     Afost : 1
                     g : 1
                     w : 1''',
               pre ='''Afre=Afre*np.exp((lastupdate-t)/tau_learn)
                      Afost=Afost*np.exp((lastupdate-t)/tau_learn)
                      Afre+=deltaAbias
                      w=w+g*Afost
                      I_rec_post+= w''',
               post='''Afre=Afre*np.exp((lastupdate-t)/tau_learn)
                      Afost=Afost*np.exp((lastupdate-t)/tau_learn)
                      Afost+=deltaAbias
                      w=w+g*Afre'''
                      )
    Sbh[:,:] = 'i==j'
    Sbh.w[:] = b_h[:]/beta/bias_input_rate/tau_rec
    
    Srs=Synapses(neuron_group_rvisible, neuron_group_rhidden,
           model='''Afre : 1
                     Afost : 1
                     g : 1
                     w : 1''',
               pre ='''Afre=Afre*np.exp((lastupdate-t)/tau_learn)
                      Afost=Afost*np.exp((lastupdate-t)/tau_learn)
                      Afre+=deltaA
                      I_rec_post+= w
                      w=w+g*Afost''',
               post='''Afre=Afre*np.exp((lastupdate-t)/tau_learn)
                      Afost=Afost*np.exp((lastupdate-t)/tau_learn)
                      Afost+=deltaA
                      I_rec_pre+= w
                      w=w+g*Afre'''
                      )
    Srs[:,:] = True
    M_rec = Whv/beta
    for i in range(M_rec.shape[0]):
        Srs.w[i,:] = M_rec[i,:]
    
    
    netobjs+=[Sbv,Sbh,Srs]

    ev = CountingEventClock(period = dcmt*t_ref)
    @network_operation(clock = ev)
    def g_update(when='after'):
        tmod, n = ev.step()
        if tmod < 50:
            neuron_group_rvisible.I_d = Id[n]
        else:
            neuron_group_rvisible.I_d = 0.
    
        if tmod<=int(t_burn_percent):
            Srs.g = 0.
            Sbh.g = Sbv.g = 0.

        elif int(t_burn_percent)<=tmod<49:
            g_up = 1.
            Srs.g = Sbv.g = Sbh.g =  g_up
            
        elif 49<=tmod < 50+int(t_burn_percent):
            Srs.g = Sbv.g = Sbh.g = +0.
            
        elif 50+int(t_burn_percent) <= tmod <99:
            g_down = -1.
            Srs.g = Sbv.g = Sbh.g = g_down
            
        elif 99<= tmod:
            Srs.g = 0.
            Sbh.g = Sbv.g = 0.
    
        if tmod==50:
            #neuron_group_rvisible.I_DATA=0
            Srs.Afre=0
            Srs.Afost=0
            Sbv.Afre=0
            Sbv.Afost=0
            Sbh.Afre=0
            Sbh.Afost=0

    netobjs += [g_update]
        
    w_hist_v = []
    w_hist_c = []
    b_hist_vc = []
    b_hist_h = []
    
    if display:
        iv_seq, iv_l_seq, train_iv, train_iv_l, test_iv, test_iv_l = mnist_data
        figure()
        res_hist_test=[]
        res_hist_train=[]
        test_data = test_iv
        test_labels = test_iv_l
        train_data = train_iv[:200]
        train_labels = train_iv_l[:200]
        
        @network_operation(clock=EventClock(dt=10*defaultclock.dt))
        def plot_performance(when='after'):    
            n = ev.n
            Wt = Srs.w.data.reshape(N_v+N_c,N_h)
            w_hist_v.append(Wt[:N_v,:].mean())
            w_hist_c.append(Wt[N_v:,:].mean())
            b_hist_vc.append(Sbv.w.data.mean())
            b_hist_h.append(Sbh.w.data.mean())
            draw()
        
        netobjs += [plot_performance]
    
    #--------------------------- Monitors
    if monitors:
        Mh=SpikeMonitor(neuron_group_rhidden)
        Mv=SpikeMonitor(neuron_group_rvisible[:N_v])
        Mc=SpikeMonitor(neuron_group_rvisible[N_v:])
        SynMg = StateMonitor(Srs, record=[0], varname='g', timestep=10)
        SynMd = StateMonitor(neuron_group_rvisible, record=[204], varname='I_d', timestep=10)
        SynMa = StateMonitor(neuron_group_rvisible, record=[204], varname='I_rec', timestep=10)
        SynMn = StateMonitor(neuron_group_rvisible, record=[204], varname='v', timestep=10)
        netobjs += [Mh, Mv, Mc, SynMg, SynMd, SynMn, SynMa]
    #MId = StateMonitor(neuron_group_rvisible, varname='I_d', record=True)
    #MIt = StateMonitor(Sbh,varname='g',record=[0])
    net = Network(netobjs)
    if dorun:
        import time
        tic = time.time()      
        net.run(t_sim)
        toc = time.time()-tic
        print toc
        
    return locals()

if __name__ == '__main__':    
    Id = create_Id()
    W, b_v, b_c, b_h = create_rbm_parameters()
    mnist_data = load_mnist_data()
    loc = main(W, b_v, b_c, b_h, Id =create_Id(), monitors = True, display=True, mnist_data=mnist_data)
    locals().update(loc)

    



