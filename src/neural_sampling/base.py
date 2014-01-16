import numpy as np
import scipy as sp
from nnf.global_funcs import *
from nnf import Instantiable

class WithNeuralParameters(Instantiable):
    default_parameters = {}
    default_parameters['t_ref'] = 0.020
    default_parameters['dt'] = 0.0001
    default_parameters['t_sim'] = 5
    default_parameters['dc_input'] = -3
    default_parameters['N_inputs'] = 32
    default_parameters['N_units'] = 32

    def __len__(self):
        return self.N_u
            
    @property
    def N_i(self):
        return self.p['N_inputs']

    @N_i.setter
    def N_i(self, v):
        self.p['N_inputs'] = v

    @property
    def N_u(self):
        return self.p['N_units']

    @N_u.setter
    def N_u(self, v):
        self.p['N_units'] = v

    def set_input(self, input_v=None):
        '''
        Set the input vector x.

        Arguments:
        *input_v*: np.array 
        '''
        if input_v == None:
            self.set_random_input()
        else:
            check_iterable_shape(input_v, shape = (self.N_i,))
            self.input_v = input_v

    def get_input(self):
        '''
        Get the input vector x
        '''
        if not hasattr(self, 'input_v'):
            self.input_v = np.zeros(self.N_i)
        return self.input_v

    def set_bias(self, b):
        '''
        Set bias b
        Arguments:
        *b*: can be a scalar, in which case all the units have the same bias, or an iterable of dimension N_units.
        '''
        if isiterable(b):
            check_iterable_shape(b, shape = (self.N_u,))
        self.p['dc_input'] = b

    def set_weight(self, M):
        '''
        Set weight matrix M
        Arguments:
        *M*: The feed-forward connectivity matrix multiplying the inputs x. Dimensions should be (N_i,N_u)
        '''
        check_iterable_shape(M, (self.N_i, self.N_u))
        self.Wb = M

    def get_bias(self):
        '''
        Get bias b
        '''
        return self.p['dc_input']

    def get_weight(self):
        '''
        Get weight matrix M    
        '''
        return self.Wb

    def get_trun(self):
        '''
        Get simulation time
        '''
        return self.p['t_sim']

    def set_trun(self, t = None):
        '''
        Set simulation time

        Arguments:
        *t*: simulation time in seconds
        '''
        if t!=None:
            self.p['t_sim'] = t
        print 'setting t_sim to {0}'.format(t)

