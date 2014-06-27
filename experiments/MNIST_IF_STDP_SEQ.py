#!/bin/python
#-----------------------------------------------------------------------------
# File Name : mnist_feedback.py
# Purpose:
#
# Author: Emre Neftci
#
# Creation Date : 25-04-2013
# Last Modified : Mon 25 Nov 2013 11:52:42 AM PST
#
# Copyright : (c) 
# Licence : GPLv2
#----------------------------------------------------------------------------- 
from common import *

def main(Whv, b_v, b_c, b_h, Id, dorun = True, monitors=True, display=False, mnist_data = None):
    defaultclock.reinit()
    b_init = np.concatenate([b_v, b_c, b_h])
    netobjs = []

    Wh = Whv[:(N_v),:]
    b1 = (sigm(b_v)).mean() #0.47174785699167904
    b2 = (sigm(b_h)).mean() #0.55992711057515843
    Th = Bu * wb
    A = -np.log((-1+1./(t_ref*Bu))*gamma*t_ref)
    wmean = np.mean(Wh)#-0.023632872766664939
    wh = (A - wmean*b1 - b_h.mean())/Th
    wv = (A - wmean*b2 - b_v.mean())/Th
    N_helper = 25
    i_inj_v_helper =  Th/beta_fi
    i_inj_h_helper =  Th/beta_fi

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

    eqs_v_helper = Equations(eqs_str_lif_wnr, 
            Cm = 1e-12*farad,
            I_inj = i_inj_v_helper,
            g = 0.,
            sigma = 0.,
            tau_rec = tau_rec)
                                               
    eqs_h_helper = Equations(eqs_str_lif_wnr, 
            Cm = 1e-12*farad,
            I_inj = i_inj_h_helper,
            g = 0.,
            sigma = 0.,
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

    neuron_group_rvisible_helper = NeuronGroup(\
            N_helper,
            model = eqs_v_helper,
            threshold = 'v>theta*volt',
            reset = 0*volt
            )
    
    neuron_group_rhidden_helper = NeuronGroup(\
            N_helper,
            model = eqs_h_helper,
            threshold = 'v>theta*volt',
            reset = 0*volt
            )



    neuron_group_rvisible_helper.v = np.random.uniform(0,1,size=N_helper)
    neuron_group_rhidden_helper.v = np.random.uniform(0,1,size=N_helper)

    netobjs += [neuron_group_rvisible, neuron_group_rhidden, neuron_group_rvisible_helper, neuron_group_rhidden_helper]
    
    @network_operation(clock=defaultclock)
    def update_mpot_helper(when='after'):
        neuron_group_rvisible_helper.v[neuron_group_rvisible_helper.v<=0.0*volt]=0.0*volt
        neuron_group_rhidden_helper.v[neuron_group_rhidden_helper.v<=0.0*volt]=0.0*volt

    netobjs += [update_mpot_helper]
    #--------------------------- Custom Network Operations

    
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

    S_helper_v = Connection(neuron_group_rvisible_helper, neuron_group_rvisible, 'I_rec')
    S_helper_v.connect_random(neuron_group_rvisible_helper, neuron_group_rvisible, p=0.5, weight = -wv/N_helper/.5/beta/tau_rec)
    S_helperi_v = Synapses(neuron_group_rvisible, neuron_group_rvisible_helper, pre = "I_rec_post+= -wb/.5/N_v/beta_fi/tau_rec")
    S_helperi_v.connect_random(neuron_group_rvisible, neuron_group_rvisible_helper,sparseness=0.5)
    #
    S_helper_h = Connection(neuron_group_rhidden_helper, neuron_group_rhidden, 'I_rec')
    S_helper_h.connect_random(neuron_group_rhidden_helper, neuron_group_rhidden, p=0.5, weight = -wh/N_helper/.5/beta/tau_rec)
    S_helperi_h = Connection(neuron_group_rhidden, neuron_group_rhidden_helper, 'I_rec')
    S_helperi_h.connect_random(neuron_group_rhidden, neuron_group_rhidden_helper, p=0.5, weight = -wb/.5/N_h/beta_fi/tau_rec) 
    
    
    netobjs+= [Sbv,Sbh,Srs,S_helperi_v,S_helperi_h,S_helper_v,S_helper_h]

    ev = CountingEventClock(period = dcmt*t_ref)
    @network_operation(clock = ev)
    def g_update(when='after'):
        tmod, n = ev.step()
        if tmod < 50:
            neuron_group_rvisible.I_d = Id[n]
        else:
            neuron_group_rvisible.I_d = Id[n]
    
        if tmod<=int(t_burn_percent):
            Srs.g = 0.
            Sbh.g = Sbv.g = 0.

        elif int(t_burn_percent)<=tmod<49:
            g_up = 1.
            Srs.g = Sbv.g = Sbh.g =  learning_rate_decay(n,n0=4000)*g_up
            
        elif 49<=tmod < 50+int(t_burn_percent):
            Srs.g = Sbv.g = Sbh.g = +0.
            
        elif 50+int(t_burn_percent) <= tmod <99:
            g_down = -1.
            Srs.g = Sbv.g = Sbh.g = learning_rate_decay(n,n0=4000)*g_down
            
        elif 99<= tmod:
            Srs.g = 0.
            Sbh.g = Sbv.g = 0.
    
#        if tmod==50:
#            #neuron_group_rvisible.I_DATA=0
#            Srs.Afre=0
#            Srs.Afost=0
#            Sbv.Afre=0
#            Sbv.Afost=0
#            Sbh.Afre=0
#            Sbh.Afost=0

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
        Mv=SpikeMonitor(neuron_group_rvisible[:N_v])
        Mc=SpikeMonitor(neuron_group_rvisible[N_v:])
        Mvmem=StateMonitor(neuron_group_rvisible[N_v:], varname='v', record=True)
        netobjs += [Mh, Mv,Mc,Mvmem]
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

    



