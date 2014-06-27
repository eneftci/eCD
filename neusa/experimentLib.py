import numpy as np
from brian.clock import EventClock 
from funcs import *


class CountingEventClock(EventClock):
    def __init__(self, period, mod = 100, *args, **kwargs):
        self.n = 0
        self.tmod = 0
        self.mod = mod
        self.period = period
        dt= period/mod
        EventClock.__init__(self, dt)
                 
    def step(self):
        tmod_now, n_now = self.tmod, self.n
        self.tmod = np.mod(self.tmod+1, self.mod)
        self.n = int(self.t/(self.period))
        return tmod_now, n_now


def create_weight_matrix(N_v, N_h, N_c, sigma = 0.1):
    return np.random.normal(0, sigma, size=(N_v+N_c, N_h))
    
def create_bias_vectors(N_v, N_c, N_h):
    bias_v = b_v = np.zeros(N_v)
    bias_h = b_h = np.zeros(N_h)
    bias_c = b_c = np.zeros(N_c)
    return b_v, b_c, b_h    

def select_equal_n_labels(n, data, labels, classes = None, seed=None):
    if classes is None:
        classes = range(10)    
    n_classes = len(classes)
    n_s = np.ceil(float(n)/n_classes)
    max_i = [np.nonzero(labels==i)[0] for i in classes]
    if seed is not None:
        np.random.seed(seed)
    f = lambda x, n: np.random.random_integers(0, x-1, n)
    a = np.concatenate([max_i[i][f(len(max_i[i]), n_s)] for i in classes])
    np.random.shuffle(a)
    iv_seq = data[a]
    iv_l_seq = labels[a]
    return iv_seq, iv_l_seq

def load_MNIST(n_samples, min_p = 0.0001, max_p = .95, binary = False, seed=None, datafile = 'data/mnist_reduced.pkl.gz'):

    if not binary:
        max_p_ = max_p
        min_p_ = min_p
    else:
        max_p_ = 0.5
        min_p_ = 0.5
        
    import gzip, cPickle
    mat = cPickle.load(gzip.open(datafile, 'r'))

    train_iv = mat['train']
    train_iv_l = mat['train_label']
    test_iv = mat['test']
    test_iv_l = mat['test_label']
    
    train_iv[train_iv >= max_p_] = max_p
    test_iv[test_iv >= max_p_]= max_p
    train_iv[train_iv < min_p_] = min_p
    test_iv[test_iv < min_p_]= min_p
    
    iv_seq, iv_l_seq = select_equal_n_labels(n_samples, train_iv, train_iv_l, seed = seed)
    
    return iv_seq, iv_l_seq, train_iv, train_iv_l, test_iv, test_iv_l

def create_pId(iv_seq, iv_l_seq, N_v, N_c, n_c_unit, min_p = .00001, max_p = .95):
    from funcs import clamped_input_transform       
    Id = np.ones([iv_seq.shape[0], iv_seq.shape[1]+N_c])*min_p
    
    for i in range(iv_seq.shape[0]):
        cl = np.zeros(N_c)
        cl[(iv_l_seq[i]*n_c_unit):((iv_l_seq[i]+1)*n_c_unit)] = max_p
        Id[i,N_v:] = clamped_input_transform(cl, min_p = min_p, max_p = max_p)
        Id[i,:N_v] = clamped_input_transform(iv_seq[i,:], min_p = min_p, max_p = max_p)

    return Id

def learning_rate_decay(n,n0=1):
    return float(n0)/(float(n0)+n)

eqs_str_lif_nrd_g = '''
dv/dt = (-g*v + I_inj + gate*I_rec + I_noise + I_d)/Cm :volt
dI_noise/dt = -I_noise/tau_noise : amp
dI_rec/dt = -I_rec/tau_rec : amp
I_d : amp
gate : 1
'''
eqs_str_lif_nrd = '''
dv/dt = (-g*v + I_inj + I_rec + I_noise + I_d)/Cm :volt
dI_noise/dt = -I_noise/tau_noise : amp
dI_rec/dt = -I_rec/tau_rec : amp
I_d : amp
'''

eqs_str_lif_wnrd = '''
dv/dt = (-g*v + I_inj + I_rec + sigma*xi + I_d)/Cm :volt
dI_rec/dt = -I_rec/tau_rec : amp
I_d : amp
'''


eqs_str_lif_nr = '''
dv/dt = (-g*v + I_inj + I_rec + I_noise)/Cm :volt
dI_noise/dt = -I_noise/tau_noise : amp
dI_rec/dt = -I_rec/tau_rec : amp
'''

eqs_str_lif_wnr = '''
dv/dt = (-g*v + I_inj + I_rec + sigma*xi)/Cm :volt
dI_rec/dt = -I_rec/tau_rec : amp
'''


def get_update_tick(ev, delta_t):
    ev.t

def get_visible_rates(Mv, t_start = 0., t_end = 250.):
    Sv = monitor_to_spikelist(Mv)
    Svd = Sv.time_slice(t_start, t_end)
    return Svd.mean_rates()



























    
