#!/bin/python
#-----------------------------------------------------------------------------
# File Name : mnist_feedback.py
# Purpose:
#
# Author: Emre Neftci
#
# Creation Date : 25-04-2013
# Last Modified : Thu 03 Oct 2013 06:34:35 PM PDT
#
# Copyright : (c) 
# Licence : GPLv2
#----------------------------------------------------------------------------- 
from common import *

def main(Whv, b_v, b_c, b_h, Id, dorun = True, monitors=True, display=False, mnist_data=None):
    defaultclock.reinit()
    b_init = np.concatenate([b_v, b_c, b_h])
    netobjs = []
    
    eqs_v = Equations(eqs_str_lif_nrd, 
            Cm = 1e-12*farad,
            I_inj = i_inj,
            g = g_leak,
            tau_noise = tau_noise,
            tau_rec = tau_rec)
                                               
    eqs_h = Equations(eqs_str_lif_nr, 
            Cm = 1e-12*farad,
            I_inj = + i_inj,
            g = g_leak,
            tau_noise = tau_noise,
            tau_rec = tau_rec)
    
    print "Creating Population"
    neuron_group_rvisible =  NeuronGroup((N_v+N_c),
          model=eqs_v,
          threshold = SimpleFunThreshold( exp_prob_beta_gamma(defaultclock.dt, beta, g_leak, gamma, t_ref), state = 'v'),
          refractory = t_ref)
    
    
    
    neuron_group_rhidden = NeuronGroup(
          N_h,
          model=eqs_h,
          threshold = SimpleFunThreshold( exp_prob_beta_gamma(defaultclock.dt, beta, g_leak, gamma, t_ref), state = 'v'),
          refractory = t_ref)
    netobjs += [neuron_group_rvisible, neuron_group_rhidden]
    
                                                          
                                            
                                  
                                                                           
                                                                         
    
    
    #Bias group
    Bv = PoissonGroup(N_v+N_c, rates = bias_input_rate) #Noise injection to h
    Bh = PoissonGroup(N_h, rates = bias_input_rate) #Noise injection to h
    
    netobjs += [Bv, Bh] 
    
    
    #---------------------- Initialize State Variables
    neuron_group_rvisible.I_d = 0.
    neuron_group_rvisible.I_noise = 0.
    neuron_group_rhidden.I_noise = 0.
    
    
    #Bias units
    Sbv = Synapses(Bv, neuron_group_rvisible, 
              model='''w : 1''',
               pre ='''I_rec_post+= w''',
                      )
    Sbv[:,:] = 'i==j'
    Sbv.w[:] = np.concatenate([b_v,b_c])/beta/bias_input_rate/tau_rec
    
    Sbh = Synapses(Bh, neuron_group_rhidden, 
              model='''w : 1''',
               pre ='''I_rec_post+= w''',

                      )
    Sbh[:,:] = 'i==j'
    Sbh.w[:] = b_h/beta/bias_input_rate/tau_rec
    
    Srs=Synapses(neuron_group_rvisible, neuron_group_rhidden,
           model='''w : 1''',
               pre ='''I_rec_post+= w''',
               post='''I_rec_pre+= w'''
                      )
    Srs[:,:] = True
    M_rec = Whv/beta
    for i in range(M_rec.shape[0]):
        Srs.w[i,:] = M_rec[i,:]
    
    netobjs += [Sbv, Sbh, Srs] 

    Mv_update  = SpikeMonitor(neuron_group_rvisible)
    Mh_update  = SpikeMonitor(neuron_group_rhidden)

    netobjs += [Mv_update, Mh_update] 
    #M = StateMonitor(neuron_group_rvisible, varname='I_DATA', record=True)
    
    #--------------------------- Custom Network Operations
    
    deltaT = ((0.49-t_burn_percent/100)*dcmt*t_ref)#rate computation window
    
    dWSrsm_p = 0. 
    dWbvm_p = .0
    dWbhm_p = .0
    dWSrsm_m = .0 
    dWbvm_m = .0
    dWbhm_m = .0
    
    
    ev = CountingEventClock(period = dcmt*t_ref)
    @network_operation(clock = ev)
    def weight_update(when='after'):
        tmod, n = ev.step()
        
        global dWSrsm_p, dWSrsm_m
        global dWbvm_p , dWbvm_m
        global dWbhm_p , dWbhm_m
    
        if tmod < 50:
            neuron_group_rvisible.I_d = Id[n]
        else:
            neuron_group_rvisible.I_d = 0.
    
     
        if tmod == 49:
            #Compute rates
            f_v = np.array(spike_histogram(Mv_update, ev.t-deltaT, ev.t))[:,1]
            f_h = np.array(spike_histogram(Mh_update, ev.t-deltaT, ev.t))[:,1]
            dWSrsm_p = np.dot(np.array([f_v]).T,np.array([f_h])).flatten()
            dWbvm_p = f_v*bias_input_rate
            dWbhm_p = f_h*bias_input_rate
    
        elif tmod == 99:
            #Compute rates
            f_v = np.array(spike_histogram(Mv_update, ev.t-deltaT, ev.t))[:,1]
            f_h = np.array(spike_histogram(Mh_update, ev.t-deltaT, ev.t))[:,1]
            dWSrsm_m = np.dot(np.array([f_v]).T,np.array([f_h])).flatten()
            dWbvm_m = f_v*bias_input_rate
            dWbhm_m = f_h*bias_input_rate
    
            Srs.w[:] += learning_rate_decay(n,n0=4000)*epsilon*(dWSrsm_p-dWSrsm_m)
            Sbv.w[:] += learning_rate_decay(n,n0=4000)*epsilon_bias*(dWbvm_p-dWbvm_m)
            Sbh.w[:] += learning_rate_decay(n,n0=4000)*epsilon_bias*(dWbhm_p-dWbhm_m)
    
            Mv_update._spiketimes = {}
            Mv_update.spikes = []
            Mv_update.nspikes = 0
            Mh_update._spiketimes = {}
            Mh_update.spikes = []
            Mh_update.nspikes = 0

    netobjs +=[weight_update]
        
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
        plot_every = 10
        
        @network_operation(clock=EventClock(dt=plot_every*dcmt*t_ref))
        def plot_performance(when='after'):    
            n = ev.n
            Wt = Srs.w.data.reshape(N_v+N_c,N_h)
            w_hist_v.append(Wt[:N_v,:].mean())
            w_hist_c.append(Wt[N_v:,:].mean())
            b_hist_vc.append(Sbv.w.data.mean())
            b_hist_h.append(Sbh.w.data.mean())
            W=Srs.w.data.copy().reshape(N_v+N_c, N_h)*beta
            Wvh=W[:N_v,:]
            Wch=W[N_v:,:]
            mBv = Sbv.w.data*beta*tau_rec*bias_input_rate
            mBh = Sbh.w.data*beta*tau_rec*bias_input_rate
            b_c = mBv[N_v:(N_v+N_c)]
            b_v = mBv[:N_v]
            b_h = mBh
            mB = np.concatenate([mBv,mBh])
    
        
            accuracy_test = classification_free_energy(Wvh, Wch, b_h, b_c, test_data, test_labels, n_c_unit)[0]    
            res_hist_test.append(accuracy_test)
            
        
            accuracy_train = classification_free_energy(Wvh, Wch, b_h, b_c, train_data, train_labels, n_c_unit)[0]
            res_hist_train.append(accuracy_train)
        
            clf()
            plot(res_hist_train, 'go-', linewidth=2)
            plot(res_hist_test, 'ro-', linewidth=2)
            axhline(0.1)
            axhline(0.85)
            axhline(0.9, color='r')
            xlim([0,t_sim/(plot_every*dcmt*t_ref)])
            ylim([0.0,1])
            a=plt.axes([0.7,0.1,0.2,0.2])
            a.plot(w_hist_v,'b.-')
            a.plot(w_hist_c,'k.-')
            a.plot(b_hist_vc,'g.-')
            a.plot(b_hist_h,'r.-')
        
            print accuracy_test
            draw()
    
        netobjs += [plot_performance]

    #--------------------------- Monitors
    if monitors:
        Mh=SpikeMonitor(neuron_group_rhidden)
        Mv=SpikeMonitor(neuron_group_rvisible)
        netobjs+=[Mh,Mv]
    #MId = StateMonitor(neuron_group_rvisible, varname='I_d', record=True)
    #MIt = StateMonitor(Sbh,varname='g',record=[0])
    net = Network(netobjs)
    if dorun:
        import time
        tic = time.time()        
        net.run(t_sim)
        toc = time.time()-tic
        print toc
            
    #return locals()
    return locals()

if __name__ == '__main__':
    Id = create_Id()
    W, b_v, b_c, b_h = create_rbm_parameters()
    loc = main(W, b_v, b_c, b_h, Id = Id, monitors = False, display=True)
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
    



