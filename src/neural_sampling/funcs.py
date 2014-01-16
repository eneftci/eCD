import brian_no_units
import cPickle, scipy.io, time
# from neusa.funcs import ns_hist, plot_ns_hist, iter_bin_vec
import numpy as np
from copy import deepcopy

from itertools import product as iterprod
def iter_bin_vec(n):
    for i,z in enumerate(iterprod(*[[0,1]]*n)):
        yield z

def ns_hist(ns, t_burn = 0.1):
    import numpy as np
    #------Analyse-------
    MP, Mv, MvI, Mg, Ms, Mo, base_mon = ns.mons
    PSC_DURATION = ns.p['t_ref']
    INPUT_V = ns.input_v  
    N_INPUT, N = ns.p['N_inputs'], ns.p['N_units']
    Mw = ns.Wb
    T_SIM = ns.get_trun()
    binput = ns.p['dc_input']
 
    #Actual output count
    s_count = np.array(spike_histogram(Ms, t_burn, float(T_SIM))).T[1]
    b_count = np.array(spike_histogram(Mo, t_burn, float(T_SIM))).T[1]
 
    k = s_count/np.sum(s_count)
     
    b = np.dot(b_count, Mw)*PSC_DURATION+binput
    pdf = psig(np.zeros([N,N]), b)
    p_ = pdf(np.arange(N))
    p_ = p_/p_.sum()
         
    bv = np.dot(INPUT_V, Mw)*PSC_DURATION+binput
    pdfv =psig(np.zeros([N,N]), bv)
    p_v = pdfv(np.arange(N))
    p_v = p_v/p_v.sum()
 
    return s_count, k, p_, p_v
  
def plot_ns_hist(ns):
    import pylab
    import numpy as np
    s_, k, p_, p_v = ns_hist(ns)
    pylab.figure()
    pylab.bar(np.arange(len(s_)), p_, width=0.4, color='k', label='Exact')
    pylab.bar(np.arange(len(s_))+0.4,k, width=0.4, label = 'Sampled', color='g')
    pylab.legend(loc=1)
    pylab.xlabel('$p(z_k|z_{/k}$')
    pylab.ylabel('$k$')
    pylab.show()
    
def spike_histogram(spike_monitor, t_start, t_stop):
    '''
    Returns firing rate of spike_monitor between t_start and t_stop
    '''
    import numpy as np
    delta_t = t_stop - t_start
    k, v = zip(*spike_monitor.spiketimes.items())   
    def f(s):
        idx_low = s >= t_start
        idx_high = s < t_stop
        idx = idx_low * idx_high
        return np.sum(idx)
    count = np.array(map(f, v), dtype='float')/delta_t
    return np.array(zip(*[k,count]))

import numpy as np

def isiterable(obj):
    if hasattr(obj, '__iter__') or hasattr(obj, '__len__'):
        return True
    else:
        return False
        
def check_iterable(obj):
    if not isiterable(obj):
        raise TypeError('{0} is not an iterable'.format(str(obj)))
    
def check_type(obj, check_class, verbose = False):
    if isinstance(obj, check_class):
        return True
    else:
        raise TypeError('obj is of type {0}, but it shoud be of type {1}'.format(obj.__name__, check_class.__name__))

def check_iterable_shape(obj, shape):
    check_iterable(obj)
    obj_np = np.array(obj)
    if not obj_np.shape == shape:
        raise TypeError('Shape of iterable is {0}, should be {1}'.format(obj.shape, shape))

def time_slice(spike_monitor, t_start, t_stop):
    import copy
    import numpy as np
    Ms_slice = copy.copy(spike_monitor) 
    adtm = np.array(spike_monitor.it)
    if adtm.shape[1]>0:
        lo_idx = adtm[1]>t_start
        hi_idx = adtm[1]<t_stop
        idx = lo_idx * hi_idx
        Ms_slice.spikes = adtm.transpose()[idx].tolist()
    else:
        Ms_slice.spikes = []
    return Ms_slice

def isi(M):    
    import numpy as np
    l = []
    ms = np.array(M.spikes)
    id_list = np.unique(ms)
    for i in id_list:
        idx = ms[:,0] == i
        tp = np.diff(ms[idx,1])
        l.append(tp)
    return np.concatenate(l)

def isihist(M,sc):
    import numpy as np
    f=isi(M)
    b=np.linspace(0, 15, 60)
    h = np.histogram(sc*f, bins=b)
    return h


def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))

def heaviside(V, T):
    return np.array((V-T)>0, dtype='float')

def exponential_prob(dt):
    def func(V):
        return np.random.rand( len(V) ) < (1-np.exp(-np.exp(V)*float(dt)))
    return func



def save_best_W(W, b, accuracy, res_hist, best_p):
    if len(res_hist)==0:
        best_p['W'] = W
        best_p['b'] = b
    elif accuracy>np.max(res_hist):
        best_p['W'] = W
        best_p['b'] = b
    return best_p

def compute_construction(input_vector, Wh, Wc, bv, bh):
    return sigm(np.dot(sigm(np.dot(input_vector,Wc.T)+b_h),Wh.T)+b_v)

def compute_prediction(input_vector, Wh, Wc, bh, bc):
    return sigm(np.dot(sigm(np.dot(input_vector,Wh)+b_v),Wc)+b_c)

############################################################
def load_NS_v2(N_v, N_h, N_c, dataset = 'Results/062__22-05-2013/WSCD.pkl'):
    # print 'Pre-loading continuous CD'
    res = cPickle.load(file(dataset,'r'))
    Wh = res['Wh']
    Wc = res['Wc'].T
    b_vch = res['b_vch'] 
    return Wh, Wc, b_vch

def load_matlab_v1(N_v, N_h, N_c, dataset = '../data/neural_nosoftmax2.mat'):
    import scipy.io
    mat = scipy.io.loadmat(dataset)
    bias_v = b_v = mat['c'].reshape(N_v)
    bias_h = b_h = mat['b'].reshape(N_h)
    bias_c = b_c = mat['cc'].reshape(N_c) 
    b_vch = np.concatenate([bias_v, bias_c, bias_h])
    Wh = mat['W']
    Wc = mat['Wc'].T #transposed because the data is formatted differently than Wh
    return Wh, Wc, b_vch

def load_theano(N_v, N_h, N_c, dataset = '../data/WhWcbvbhbc_dict_theano.pkl'):
    res = cPickle.load(file(dataset,'r'))
    print "Loading pre-trained (Theano)"
    #res = scipy.io.loadmat('../data/model_all_digits.mat')['model_AllDigits']
    bias_v = b_v = res['bv'].reshape(N_v)
    bias_h = b_h = res['bh'].reshape(N_h)
    bias_c = b_c = res['bc'].reshape(N_c)
    b_vch = np.concatenate([bias_v, bias_c, bias_h])
    Wh = res['Wh']
    Wc = res['Wc'].T #transposed because the data is formatted differently than Wh
    return Wh, Wc, b_vch

def load_matlab_v2(N_v, N_h, N_c, model='model1', dataset = '../data/model_20units.mat'):
    print "Loading pre-trained (Matlab)"
    import scipy.io
    res = scipy.io.loadmat(dataset)[model]
    bias_v = b_v = res['c'][0,0].reshape(N_v)
    bias_h = b_h = res['b'][0,0].reshape(N_h)
    bias_c = b_c = res['cc'][0,0].reshape(N_c) 
    b_vch = np.concatenate([bias_v, bias_c, bias_h])
    Wh = res['W'][0,0]
    Wc = res['Wc'][0,0].T #transposed because the data is formatted differently than Wh
    return Wh, Wc, b_vch

def load_matlab_v3(N_v, N_h, N_c, dataset = '../data/model_20units.mat'):
    print "Loading pre-trained (Matlab)"
    import scipy.io
    res = scipy.io.loadmat(dataset)['model2']
    bias_v = b_v = res['c'][0,0].reshape(N_v)
    bias_h = b_h = res['b'][0,0].reshape(N_h)
    bias_c = b_c = res['cc'][0,0].reshape(N_c) 
    b_vch = np.concatenate([bias_v, bias_c, bias_h])
    Wh = res['W'][0,0]
    Wc = res['Wc'][0,0].T #transposed because the data is formatted differently than Wh
    return Wh, Wc, b_vch

def load_NS(N_v, N_h, N_c, dataset_W = '../data/WSCD10_2', dataset_b ='../data/WSCD10b_2'):
    import cPickle
    # print 'Pre-loading continuous CD'
    W = cPickle.load(file(dataset_W,'r'))
    Wh = W[:N_v,:]
    Wc = W[N_v:,:].T
    b_vch = cPickle.load(file(dataset_b,'r'))
    return Wh, Wc, b_vch

def compute_histograms(Mi, t_stop):
    #Actual output count
    t_burn = 0.1
    b_count = np.array(spike_histogram(Mi, t_burn, float(t_stop))).T[1]
    return b_count

def plot_hist(s_count, k_count, p, b = None, log = False):
    import pylab
    import numpy as np
    if log:
        f = lambda x: -np.log(x)
    else:
        f = lambda x: x
    pylab.figure()
    pylab.bar(np.arange(len(s_count)), f(p), width=0.3, color='k', label='Neural Sampling', linewidth=0)
    pylab.bar(np.arange(len(s_count))+0.3, f(k_count), width=0.3, label = 'Gibbs Sampling', color='g', linewidth=0)
    if b is not None:
        bonly = 1./(1+exp(-b))
        bonly /= bonly.sum()
        pylab.bar(np.arange(len(s_count))+0.6,f(bonly), width=0.3, label = 'Exact (Bias only)', color='r', linewidth=0)
    pylab.legend(loc=1)
    pylab.xlabel('$p(z_k|z_{/k}$')
    pylab.ylabel('$k$')
    pylab.show()

def build_pdf(W, b1, b2):
    '''
    Compute table of probabilities
    '''
    n1 = len(b1)
    n2 = len(b2)
    Wx = np.zeros([n1+n2,n1+n2])
    Wx[:n1,n1:(n1+n2)] = W
    Wx[n1:(n1+n2),:n1] = W.T
    bx = np.concatenate([b1, b2])
    def f(z):
        return np.exp(0.5*np.dot(z,np.dot(z,Wx))+np.dot(z,bx))
    return f, Wx, bx

def kl_divergence_pdf(distr, W, b1, b2):
    '''
    KL divergence between ideal RBM with parameters *W* *b1* and *b2*, and distribution *distr*
    '''
    n = len(b1)+len(b2)
    distr2 = np.empty([2**n])
    f,w,b = build_pdf(W, b1, b2)
    for i,z in enumerate(iter_bin_vec(n)):
        distr2[i] = f(z)
    distr2 /= distr2.sum()
    kld = 0
    for i in np.arange(2**n):
        kld+= distr2[i]*(np.log(distr2[i])-np.log(distr[i]))
    return kld

def kl(distr1, distr2):
    kld = 0
    for i in np.arange(len(distr1)):
        kld+= distr2[i]*(np.log(distr2[i])-np.log(distr1[i]))
    return kld

def clamped_input_transform(input_vector, min_p=1e-7, max_p=0.999):
    s = np.array(input_vector) #Divide by t_ref to get firing rates
    s[s<min_p] = min_p
    s[s>max_p] = max_p
    s =  -np.log(-1+1./(s))
    return s

def delay_input(input_vector, delay = .2):
    def f(t):
        if t<delay:
            return np.zeros_like(input_vector)
        else:
            return input_vector
    return f

def classification_free_energy(Wvh, Wch, b_h, b_c, test_data, test_labels, n_c_unit, n_classes = 10):
    numcases = len(test_labels);
    F = np.zeros([numcases, n_classes]);
    for i in range(n_classes):
        X= np.zeros([numcases, n_c_unit*n_classes]);
        X[:, (n_c_unit*i):(n_c_unit*(i+1))] = 1;
        F[:,i] = np.tile(b_c[i],numcases)*X[:,i]+\
                 np.sum(np.log(np.exp(np.dot(test_data, Wvh)+np.dot(X,Wch)+np.tile(b_h,numcases).reshape(numcases,-1))+1), axis=1);
    prediction= np.argmax(F, axis=1);
    accuracy = 1-float(sum(prediction!=test_labels))/numcases
    assert 1>=accuracy>=.1/n_classes
    return accuracy, prediction!=test_labels

def sigm(x):
    return 1./(1+np.exp(-x))


import numpy as np

def isiterable(obj):
    if hasattr(obj, '__iter__') or hasattr(obj, '__len__'):
        return True
    else:
        return False
        
def check_iterable(obj):
    if not isiterable(obj):
        raise TypeError('{0} is not an iterable'.format(str(obj)))
    
def check_type(obj, check_class, verbose = False):
    if isinstance(obj, check_class):
        return True
    else:
        raise TypeError('obj is of type {0}, but it shoud be of type {1}'.format(obj.__name__, check_class.__name__))

def check_iterable_shape(obj, shape):
    check_iterable(obj)
    obj_np = np.array(obj)
    if not obj_np.shape == shape:
        raise TypeError('Shape of iterable is {0}, should be {1}'.format(obj.shape, shape))

def time_slice(spike_monitor, t_start, t_stop):
    import copy
    import numpy as np
    Ms_slice = copy.copy(spike_monitor) 
    adtm = np.array(spike_monitor.it)
    if adtm.shape[1]>0:
        lo_idx = adtm[1]>t_start
        hi_idx = adtm[1]<t_stop
        idx = lo_idx * hi_idx
        Ms_slice.spikes = adtm.transpose()[idx].tolist()
    else:
        Ms_slice.spikes = []
    return Ms_slice

def ksi( vmem_mon , vmin =0., vmax = 1.):
    '''
    kuramoto_synchronization_index
    vmem_mon: statemonitor of the phase (membrane potential)
    vmin, vmax: the values in vmem_mon will be clipped to [vmin,vmax]
    '''
    import numpy as np
    x=ksi_phase(vmem_mon, vmin, vmax)
    u=np.mean(x,axis=0)
    return np.array([np.absolute(u),np.angle(u)]) 

def ksi_phase(vmem_mon , vmin =0., vmax = 1.):
    import numpy as np
    g=vmem_mon.values.copy()
    g[g>float(vmax)] = float(vmax)
    g[g<float(vmin)] = float(vmin)
    g=g.astype('complex')*2*np.pi
    j=np.sqrt(np.complex(-1))
    return np.exp(j*g)


def plot_fourier_spectrum(M, Fs=1000.):
    from numpy import sin, linspace, pi
    from pylab import plot, show, title, xlabel, ylabel, subplot
    from scipy import fft, arange

    """
    Plots a Single-Sided Amplitude Spectrum of y(t)
    """
    y=M.rate
    n = len(y) # length of the signal
    k = arange(n)
    T = n/Fs
    frq = k/T # two sides frequency range
    frq = frq[range(n/2)] # one side frequency range

    Y = fft(y)/n # fft computing and normalization
    Y = Y[range(n/2)]
    
    plot(frq,abs(Y),'r') # plotting the spectrum
    xlabel('Freq (Hz)')
    ylabel('|Y(freq)|')



def isi(M):    
    import numpy as np
    l = []
    ms = np.array(M.spikes)
    id_list = np.unique(ms)
    for i in id_list:
        idx = ms[:,0] == i
        tp = np.diff(ms[idx,1])
        l.append(tp)
    return np.concatenate(l)

def isihist(M,sc):
    import numpy as np
    f=isi(M)
    b=np.linspace(0, 15, 60)
    h = np.histogram(sc*f, bins=b)
    return h


def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))

def heaviside(V, T):
    return np.array((V-T)>0, dtype='float')

def exponential_prob(dt):
    def func(V):
        return np.random.rand( len(V) ) < (1-np.exp(-np.exp(V)*float(dt)))
    return func

#def time_slice(spike_monitor, t_start, t_stop):
#    import copy
#    Ms_slice = copy.copy(Ms) 
#    adtm = np.array(Ms.it)
#    lo_idx = adtm[1]>t_start
#    hi_idx = adtm[1]<t_stop
#    idx = lo_idx * hi_idx
#    Ms_slice.spikes = adtm.transpose()[idx].tolist()
#    return Ms_slice 
    
#def spike_histogram(spike_monitor, t_start, t_stop):
#    delta_t = t_stop - t_start
#    k, v = zip(*spike_monitor.spiketimes.items())    
#    count = np.array(map(len,spike_monitor.spiketimes.values()), dtype='float')/delta_t
#    return zip(*[k,count])
        
def build_binary_vectors(N):
    import numpy as np
    g=[None for i in range(2**N)]
    for i in range(2**N):
        g[i]=np.array(list(np.binary_repr(i,width=N)),dtype='int')
    return np.array(g, 'int')

def p(W, b):
    #return the boltzmann distribution defined by the weight matrix and the biases
    def func(z):
        return np.exp(0.5*np.dot(np.dot(W,z),z)+np.dot(b,z))
    return func

def psig(W, b):
    #return the boltzmann distribution defined by the weight matrix and the biases
    def func(i):
        return 1./(1+np.exp(-b[i]))
    return func
    
def hist_p(p, gs):
    #Histogram of gs, given distribution p
    pr = np.zeros(len(gs))
    for i, g in enumerate(gs):
        pr[i] = p(g)
    return pr

def inst_firing_rate(N, M, t_start, t_stop, dt=0.001):
    t = np.arange(t_start, t_stop, dt)
    S = np.zeros([N, len(t)])
    for i,t_ in enumerate(t):
        f = zip(*time_slice(M, t_, t_+dt).spikes)
        if len(f)>0: S[np.array(f[0],'int'),i]=1
    return S

def monitor_to_spikelist(Ms):
    import pyNCS.pyST as pyST
    s = np.array(Ms.spikes)
    id_list = range(len(Ms.source))
    s[:,1] = s[:,1] * 1000 #SpikeList takes ms
    return pyST.SpikeList(spikes = s, id_list = id_list)

def get_indexes(a):
    import numpy as np
    return np.nonzero(a)[0]
