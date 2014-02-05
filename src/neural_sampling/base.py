import numpy as np
import scipy as sp

class WithDefaultParameters(object):
    '''
    A class that has default parameters as static variables. All parameters
    provided in the constructor are collected in a common parameter member
    '''
    
    default_parameters = {}
    
    def __init__(self, *params, **kw_params):
        self._parameters = {}
        self.__update_params(self.default_parameters)
        self.__update_params(kw_params)                
        self.update_default_parameters()
        super(WithDefaultParameters, self).__init__()

    def __update_params(self, dict_ = {}):
        d = self.local_params(dict_)        
        self._parameters.update(d)
    
    def local_params(self, dict_ = {}):        
        local_dict = self.parameters.copy()
        local_dict.update(dict_)
        return local_dict

    def update_default_parameters(self):
        pass
        
    @property
    def parameters(self):
        return self._parameters

    @property
    def p(self):
        return self._parameters
    
    def set_default_parameters(self, dict_):
        self._parameters = dict_.copy()

class Instantiable(WithDefaultParameters):
    '''
    A class that can be instantiated
    '''
    
    def instantiate(self, parameters={}):
        '''
        instantiates the instantiable object. parameters should be passed to the
        constructor of the object
        '''
        self.instantiated = True


    @property
    def instantiated(self):
        if not hasattr(self, '_instantiated'):
            self._instantiated = False
        else:
            return self._instantiated

    @instantiated.setter
    def instantiated(self, value = True):
        self._instantiated = value


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

