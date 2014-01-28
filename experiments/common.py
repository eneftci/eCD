import brian_no_units
from brian.globalprefs import *
set_global_preferences(useweave=True)
    
from brian import *
import numpy as np
import scipy.io
import cPickle, pylab, matplotlib


from neusa.experimentLib import *
from neusa.funcs import *

import neusa.experimentTools as et


from meta_parameters import *
exec("from {0} import *".format(parameters_script))
#from default_parameters import *

def round_to_bits(W, n_bits, clip=[-1.5,1.5]):
    W_f = W.flatten()
    W_approx = np.empty_like(W)
    n, x = np.histogram(W_f, bins = 2**n_bits)
    clip = np.array(clip)
    lb = np.maximum(0,np.minimum(len(x)-1,(x>clip[0]).searchsorted(True)))
    hb = np.maximum(0,np.minimum(len(x)-1,(x>clip[1]).searchsorted(False, 'right')))
    new_bins = np.linspace(x[lb],x[hb],2**n_bits)
    n, x = np.histogram(W_f, bins = new_bins)
    delta = x[1]-x[0]
    x += delta/2
    idx = x.searchsorted(W_f, side='left')
    idx[idx >= len(x)]=len(x)-1
    idx[idx < 0]= 0
    w = x[idx]
    W_approx = w.reshape(W.shape)
    return W_approx

def create_rbm_parameters():
    #------------------------------------------ Bias and weights
    b_v, b_c, b_h = create_bias_vectors(N_v, N_c, N_h)
    Whv = create_weight_matrix(N_v, N_h, N_c, sigma = 0.1)    
    return Whv, b_v, b_c, b_h
    
def load_mnist_data(min_p = 1e-4, max_p=.95, binary=False, seed=None):
    #------------------------------------------ Create Input Vector
    
    mnist_data = load_MNIST(n_samples,
                            min_p = min_p,
                            max_p = max_p,
                            binary = binary,
                            seed = seed)
    iv_seq, iv_l_seq, train_iv, train_iv_l, test_iv, test_iv_l = mnist_data
    return iv_seq, iv_l_seq, train_iv, train_iv_l, test_iv, test_iv_l

ion()

def create_Id(data = True, c_min_p = 1e-4, c_max_p = .95, seed = None):
    if hasattr(data, '__len__'):
        iv_seq, iv_l_seq, train_iv, train_iv_l, test_iv, test_iv_l = data
        Idp = create_pId(iv_seq, iv_l_seq, N_v, N_c, n_c_unit, min_p = c_min_p, max_p = c_max_p)
        Id = (Idp /beta)
    elif data == True:
        iv_seq, iv_l_seq, train_iv, train_iv_l, test_iv, test_iv_l = load_mnist_data(seed = seed)
        Idp = create_pId(iv_seq, iv_l_seq, N_v, N_c, n_c_unit, min_p = c_min_p, max_p = c_max_p)
        Id = (Idp /beta)
    else:
        Id = np.zeros([n_samples,N_v+N_c])  
    return Id
