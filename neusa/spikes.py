#-----------------------------------------------------------------------------
# Purpose:
#
# Author: Emre Neftci
#
# Copyright : University of Zurich, Giacomo Indiveri, Emre Neftci, Sadique Sheik, Fabio Stefanini
# Licence : GPLv2
#-----------------------------------------------------------------------------
import warnings
"""
NeuroTools.signals.spikes
==================

A collection of functions to create, manipulate and play with spikes signals.

Classes
-------

SpikeTrain       - object representing a spike train, for one cell. Useful for plots,
                   calculations such as ISI, CV, mean rate(), ...
SpikeList        - object representing the activity of a population of neurons. Functions as a
                   dictionary of SpikeTrain objects, with methods to compute firing rate,
                   ISI, CV, cross-correlations, and so on.

Functions
---------

load_spikelist       - load a SpikeList object from a file. Expects a particular format.
                       Can also load data in a different format, but then you have
                       to write your own File object that will know how to read the data (see io.py)
load                 - a generic loader for all the previous load methods.

See also NeuroTools.signals.analogs
"""


import os
import re
import logging
import numpy
import pylab
import matplotlib
#from NeuroTools import check_dependency, check_numpy_version, analysis
#from io import *
#from NeuroTools.plotting import get_display, set_axis_limits, set_labels
#from pairs import *

HAVE_PYLAB = True


def load(user_file):
    '''
    Load a file saved by SpikeList.save
    '''
    data = numpy.fliplr(numpy.loadtxt(user_file, comments='#'))
    id_list = numpy.unique(data[:, 0])
    return SpikeList(data, id_list)


class SpikeTrain(object):
    """
    SpikeTrain(spikes_times, t_start=None, t_stop=None)
    This class defines a spike train as a list of times events.

    Event times are given in a list (sparse representation) in milliseconds.

    Inputs:
        spike_times - a list/numpy array of spike times (in milliseconds)
        t_start     - beginning of the SpikeTrain (if not, this is infered)
        t_stop      - end of the SpikeTrain (if not, this is infered)

    Examples:
        >> s1 = SpikeTrain([0.0, 0.1, 0.2, 0.5])
        >> s1.isi()
            array([ 0.1,  0.1,  0.3])
        >> s1.mean_rate()
            8.0
        >> s1.cv_isi()
            0.565685424949
    """

    @property
    def spike_times(self):
        return self._spike_times

    @spike_times.setter
    def spike_times(self, value):
        self._spike_times = numpy.sort(value).astype(numpy.float32)

    #######################################################################
    ## Constructor and key methods to manipulate the SpikeTrain objects  ##
    #######################################################################
    def __init__(self, spike_times, t_start=None, t_stop=None, presorted=False):
        """
        Constructor of the SpikeTrain object
        *options*:
        preextracted: set True for faster construction of the SpikeTrain object if the spike times are already constrained to t_start and t_stop.
        presorted: set True for faster construction of the SpikeTrain object if the spike times are already sorted

        See also
            SpikeTrain
        """

        self.t_start = t_start
        self.t_stop = t_stop
        if not hasattr(spike_times, 'size'): #Assume it is a numpy array
            self._spike_times = numpy.array(spike_times,numpy.float32)
        else:
            self._spike_times = spike_times.astype(numpy.float32)


        # We sort the spike_times if necessary. Is slower, but necessary for a lot of methods...
        if not presorted:
            self._spike_times = numpy.sort(self.spike_times, kind="quicksort")
        else:
            self._spike_times = spike_times


        # If t_start is not None, we resize the spike_train keeping only
        # the spikes with t >= t_start
        if self.t_start is not None:
            idx_t_start = numpy.searchsorted(self._spike_times>=self.t_start, True, side='left')
        else:
            idx_t_start = 0
    
        # If t_stop is not None, we resize the spike_train keeping only
        # the spikes with t <= t_stop
        if self.t_stop is not None:
            idx_t_stop = numpy.searchsorted(self._spike_times < self.t_stop, True, side='right')
        else:
            idx_t_stop = len(self._spike_times)
        
        if idx_t_start>idx_t_stop: 
            print('No event between t_start and t_stop')

        self._spike_times = self._spike_times[idx_t_start:idx_t_stop]


        # Here we deal with the t_start and t_stop values if the SpikeTrain
        # is empty, with only one element or several elements, if we
        # need to guess t_start and t_stop
        # no element : t_start = 0, t_stop = 0.1
        # 1 element  : t_start = time, t_stop = time + 0.1
        # several    : t_start = min(time), t_stop = max(time)

        size = len(self._spike_times)
        if size == 0:
            if self.t_start is None:
                self.t_start = 0
            if self.t_stop is None:
                self.t_stop = 0.1
        elif size == 1:  # spike list may be empty
            if self.t_start is None:
                self.t_start = self._spike_times[0]
            if self.t_stop is None:
                self.t_stop = self._spike_times[0] + 0.1
        elif size > 1:
            if self.t_start is None:
                self.t_start = self._spike_times[0]
            if self.t_stop is None:
                self.t_stop = self._spike_times[-1]
            if self._spike_times[-1] > self.t_stop:
                raise ValueError("Spike times must not be greater than t_stop")
            if self._spike_times[0] < self.t_start:
                raise ValueError("Spike times must not be less than t_start")

        if self.t_start > self.t_stop:
            raise Exception("Incompatible time interval : t_start = %s, t_stop = %s" %
                 (self.t_start, self.t_stop))

        elif self.t_start == self.t_stop:
            logging.debug(
                "Warning, t_stop == t_start, setting t_stop = t_start+1ms")
            self.t_stop = self.t_start + 1.

        if self.t_start < 0:
            raise ValueError("t_start must not be negative")
        if len(self._spike_times)>0:
            if self._spike_times[0] < 0:
                raise ValueError("Spike times must not be negative")


    def __str__(self):
        return str(self.spike_times)

    def __del__(self):
        pass

    def __len__(self):
        return len(self.spike_times)

    def __getslice__(self, i, j):
        """
        Return a sublist of the spike_times vector of the SpikeTrain
        """
        return self.spike_times[i:j]

    def time_parameters(self):
        """
        Return the time parameters of the SpikeTrain (t_start, t_stop)
        """
        return (self.t_start, self.t_stop)

    def is_equal(self, spktrain):
        """
        Return True if the SpikeTrain object is equal to one other SpikeTrain, i.e
        if they have same time parameters and same spikes_times

        Inputs:
            spktrain - A SpikeTrain object

        See also:
            time_parameters()
        """
        test = (self.time_parameters() == spktrain.time_parameters())
        return numpy.all(self.spike_times == spktrain.spike_times) and test

    def copy(self):
        """
        Return a copy of the SpikeTrain object
        """
        return SpikeTrain(self.spike_times, self.t_start, self.t_stop)

    def duration(self):
        """
        Return the duration of the SpikeTrain
        """
        return self.t_stop - self.t_start

    def merge(self, spiketrain):
        """
        Add the spike times from a spiketrain to the current SpikeTrain

        Inputs:
            spiketrain - The SpikeTrain that should be added

        Examples:
            >> a = SpikeTrain(range(0,100,10),0.1,0,100)
            >> b = SpikeTrain(range(400,500,10),0.1,400,500)
            >> a.merge(b)
            >> a.spike_times
                [   0.,   10.,   20.,   30.,   40.,   50.,   60.,   70.,   80.,
                90.,  400.,  410.,  420.,  430.,  440.,  450.,  460.,  470.,
                480.,  490.]
            >> a.t_stop
                500
        """
        self.spike_times = numpy.insert(
                self.spike_times,
                self.spike_times.searchsorted(spiketrain.spike_times),
                spiketrain.spike_times)
        self.t_start = min(self.t_start, spiketrain.t_start)
        self.t_stop = max(self.t_stop, spiketrain.t_stop)

    def format(self, relative=False, quantized=False):
        """
        Return an array with a new representation of the spike times

        Inputs:
            relative  - if True, spike times are expressed in a relative
                       time compared to the previsous one
            quantized - a value to divide spike times with before rounding

        Examples:
            >> st.spikes_times=[0, 2.1, 3.1, 4.4]
            >> st.format(relative=True)
                [0, 2.1, 1, 1.3]
            >> st.format(quantized=2)
                [0, 1, 2, 2]
        """
        spike_times = self.spike_times.copy()

        if relative and len(spike_times) > 0:
            spike_times[1:] = spike_times[1:] - spike_times[:-1]

        if quantized:
            assert quantized > 0, "quantized must either be False or a positive number"
            # spike_times =  numpy.array([time/self.quantized for time in
            # spike_times],int)
            spike_times = (spike_times / quantized).round().astype('int')

        return spike_times

    def jitter(self, jitter):
        """
        Returns a new SpikeTrain with spiketimes jittered by a normal distribution.

        Inputs:
              jitter - sigma of the normal distribution

        Examples:
              >> st_jittered = st.jitter(2.0)
        """

        return SpikeTrain(self.spike_times + jitter * (numpy.random.normal(loc=0.0, scale=1.0, size=self.spike_times.shape[0])), t_start=self.t_start, t_stop=self.t_stop)

    #######################################################################
    ## Analysis methods that can be applied to a SpikeTrain object       ##
    #######################################################################
    def isi(self):
        """
        Return an array with the inter-spike intervals of the SpikeTrain

        Examples:
            >> st.spikes_times=[0, 2.1, 3.1, 4.4]
            >> st.isi()
                [2.1, 1., 1.3]

        See also
            cv_isi
        """
        return numpy.diff(self.spike_times)

    def mean_rate(self, t_start=None, t_stop=None):
        """
        Returns the mean firing rate between t_start and t_stop, in Hz

        Inputs:
            t_start - in ms. If not defined, the one of the SpikeTrain object is used
            t_stop  - in ms. If not defined, the one of the SpikeTrain object is used

        Examples:
            >> spk.mean_rate()
                34.2
        """
        if (t_start == None) & (t_stop == None):
            t_start = self.t_start
            t_stop = self.t_stop
            idx = self.spike_times
        else:
            if t_start == None:
                t_start = self.t_start
            else:
                t_start = max(self.t_start, t_start)
            if t_stop == None:
                t_stop = self.t_stop
            else:
                t_stop = min(self.t_stop, t_stop)
            idx = numpy.where((self.spike_times >= t_start) & (
                self.spike_times <= t_stop))[0]
        return 1000. * len(idx) / (t_stop - t_start)

    def mean_rate_isi(self, t_start=None, t_stop=None):
        '''
        Returns the mean rate (Hz) based on the interspike intervals

        Inputs:
            t_start, t_stop - in ms. If not defined, the one of the SpikeTrain object is used
        '''
        if (t_start == None) & (t_stop == None):
            isi = self.isi()
        else:
            if t_start == None:
                t_start = self.t_start
            else:
                t_start = max(self.t_start, t_start)
            if t_stop == None:
                t_stop = self.t_stop
            else:
                t_stop = min(self.t_stop, t_stop)
            print(t_start, t_stop)
            isi = self.time_slice(t_start, t_stop).isi()
        if len(isi):
            mean_rate = 1000./numpy.mean(isi)
            return mean_rate
        else:
            return 0 # There were no spikes

    def cv_isi(self):
        """
        Return the coefficient of variation of the isis.

        cv_isi is the ratio between the standard deviation and the mean of the ISI
          The irregularity of individual spike trains is measured by the squared
        coefficient of variation of the corresponding inter-spike interval (ISI)
        distribution normalized by the square of its mean.
          In point processes, low values reflect more regular spiking, a
        clock-like pattern yields CV2= 0. On the other hand, CV2 = 1 indicates
        Poisson-type behavior. As a measure for irregularity in the network one
        can use the average irregularity across all neurons.

        http://en.wikipedia.org/wiki/Coefficient_of_variation

        See also
            isi, cv_kl

        """
        isi = self.isi()
        if len(isi) > 0:
            return numpy.std(isi) / numpy.mean(isi)
        else:
            logging.debug("Warning, a CV can't be computed because there are not enough spikes")
            return numpy.nan

    def cv_kl(self, bins=100):
        """
        Provides a measure for the coefficient of variation to describe the
        regularity in spiking networks. It is based on the Kullback-Leibler
        divergence and decribes the difference between a given
        interspike-interval-distribution and an exponential one (representing
        poissonian spike trains) with equal mean.
        It yields 1 for poissonian spike trains and 0 for regular ones.

        Reference:
            http://incm.cnrs-mrs.fr/LaurentPerrinet/Publications/Voges08fens

        Inputs:
            bins - the number of bins used to gather the ISI

        Examples:
            >> spklist.cv_kl(100)
                0.98

        See also:
            cv_isi

        """
        isi = self.isi() / 1000.
        if len(isi) < 2:
            logging.debug("Warning, a CV can't be computed because there are not enough spikes")
            return numpy.nan
        else:
            proba_isi, xaxis = numpy.histogram(isi, bins=bins, normed=True)
            xaxis = xaxis[:-1]
            proba_isi /= numpy.sum(proba_isi)
            bin_size = xaxis[1] - xaxis[0]
            # differential entropy:
            # http://en.wikipedia.org/wiki/Differential_entropy
            KL = - numpy.sum(
                proba_isi * numpy.log(proba_isi + 1e-16)) + numpy.log(bin_size)
            KL -= -numpy.log(self.mean_rate()) + 1.
            CVkl = numpy.exp(-KL)
            return CVkl

    def fano_factor_isi(self):
        """
        Return the fano factor of this spike trains ISI.

        The Fano Factor is defined as the variance of the isi divided by the mean of the isi

        http://en.wikipedia.org/wiki/Fano_factor

        See also
            isi, cv_isi
        """
        isi = self.isi()
        if len(isi) > 0:
            fano = numpy.var(isi) / numpy.mean(isi)
            return fano
        else:
            raise Exception("No spikes in the SpikeTrain !")

    def time_axis(self, time_bin=10):
        """
        Return a time axis between t_start and t_stop according to a time_bin

        Inputs:
            time_bin - the bin width

        Examples:
            >> st = SpikeTrain(range(100),0.1,0,100)
            >> st.time_axis(10)
                [ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90n 100]

        See also
            time_histogram
        """
        axis = numpy.arange(self.t_start, self.t_stop + time_bin, time_bin)
        return axis

    # def raster_plot(self, t_start=None, t_stop=None, interval=None,
    # display=True, kwargs={}):
    #    """
    #    Generate a raster plot with the SpikeTrain in a subwindow of interest,
    #    defined by t_start and t_stop.
    #
    #    Inputs:
    # t_start - in ms. If not defined, the one of the SpikeTrain object is used
    # t_stop  - in ms. If not defined, the one of the SpikeTrain object is used
    # display - if True, a new figure is created. Could also be a subplot
    # kwargs  - dictionary contening extra parameters that will be sent to the
    # plot
    #                  function
    #
    #    Examples:
    #        >> z = subplot(221)
    #        >> st.raster_plot(display=z, kwargs={'color':'r'})
    #
    #    See also
    #        SpikeList.raster_plot
    #    """
    #    if t_start is None: t_start = self.t_start
    #    if t_stop is None:  t_stop = self.t_stop
    #
    #    ====================
    #    if interval is None:
    #        interval = Interval(t_start, t_stop)
    #
    #    spikes  = interval.slice_times(self.spike_times)
    #    subplot = get_display(display)
    #    if not subplot or not HAVE_PYLAB:
    #        print PYLAB_ERROR
    #        return
    #    else:
    #        if len(spikes) > 0:
    #            subplot.plot(spikes,numpy.ones(len(spikes)),',', **kwargs)
    #            xlabel = "Time (ms)"
    #            ylabel = "Neuron"
    #            set_labels(subplot, xlabel, ylabel)
    #            pylab.draw()
    #    ====================

    def time_offset(self, offset = None, t_start=None, t_stop=None):
        """
        Add an offset to the SpikeTrain object. t_start and t_stop are
        shifted from offset, so does all the spike times.
        
        Inputs:
            offset - the time offset, in ms
            
        Returns None: changes to the SpikeList are made in-place

        Examples:
            >> spktrain = SpikeTrain(arange(0,100,10))
            >> spktrain.time_offset(50)
            >> spklist.spike_times
                [  50.,   60.,   70.,   80.,   90.,  100.,  110.,
                120.,  130.,  140.]
        """
        if isinstance(self, emptySpikeTrain) or offset is None:
            return
        
        if t_start == None:
            self.t_start += offset
        else:
            self.t_start = t_start

        if t_stop == None:
            self.t_stop += offset
        else:
            self.t_stop = t_stop

        self.spike_times += offset

    def time_slice(self, t_start, t_stop):
        """
        Return a new SpikeTrain obtained by slicing between t_start and t_stop. The new
        t_start and t_stop values of the returned SpikeTrain are the one given as arguments

        Inputs:
            t_start - begining of the new SpikeTrain, in ms.
            t_stop  - end of the new SpikeTrain, in ms.

        Examples:
            >> spk = spktrain.time_slice(0,100)
            >> spk.t_start
                0
            >> spk.t_stop
                100
        """
        spikes = numpy.extract((self.spike_times >= t_start) & (
            self.spike_times <= t_stop), self.spike_times)
        return SpikeTrain(spikes, t_start, t_stop)

    #def interval_slice(self, interval):
    #    """
    #    Return a new SpikeTrain obtained by slicing with an Interval. The new
    # t_start and t_stop values of the returned SpikeTrain are the extrema of
    # the Interval
    #
    #    Inputs:
    #        interval - The interval from which spikes should be extracted

    #    Examples:
    #        >> spk = spktrain.time_slice(0,100)
    #        >> spk.t_start
    #            0
    #        >> spk.t_stop
    #            100
    #    """

    #    ====================
    #    times           = interval.slice_times(self.spike_times)
    #    t_start, t_stop = interval.time_parameters()
    #    ====================

    #    return SpikeTrain(times, t_start, t_stop)
    #

    def time_histogram(self, time_bin=10, normalized=True):
        """
        Bin the spikes with the specified bin width. The first and last bins
        are calculated from `self.t_start` and `self.t_stop`.

        Inputs:
            time_bin   - the bin width for gathering spikes_times
            normalized - if True, the bin values are scaled to represent firing rates
                         in spikes/second, otherwise otherwise it's the number of spikes
                         per bin.

        Examples:
            >> st=SpikeTrain(range(0,100,5),0.1,0,100)
            >> st.time_histogram(10)
                [200, 200, 200, 200, 200, 200, 200, 200, 200, 200]
            >> st.time_histogram(10, normalized=False)
                [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

        See also
            time_axis
        """
        bins = self.time_axis(time_bin)
        hist, edges = numpy.histogram(self.spike_times, bins)
        if normalized and isinstance(time_bin, int):  # what about normalization if time_bin is a sequence?
            hist *= 1000.0 / time_bin
        return hist

    def relative_times(self):
        """
        Rescale the spike times to make them relative to t_start.

        Note that the SpikeTrain object itself is modified, t_start
        is substracted to spike_times, t_start and t_stop
        """
        if self.t_start != 0:
            self.spike_times -= self.t_start
            self.t_stop -= self.t_start
            self.t_start = 0.0

    def distance_victorpurpura(self, spktrain, cost=0.5):
        """
        Function to calculate the Victor-Purpura distance between two spike trains.
        See J. D. Victor and K. P. Purpura,
            Nature and precision of temporal coding in visual cortex: a metric-space
            analysis.,
            J Neurophysiol,76(2):1310-1326, 1996

        Inputs:
            spktrain - the other SpikeTrain
            cost     - The cost parameter. See the paper for more information
        """
        nspk_1 = len(self)
        nspk_2 = len(spktrain)
        if cost == 0:
            return abs(nspk_1 - nspk_2)
        elif cost > 1e9:
            return nspk_1 + nspk_2
        scr = numpy.zeros((nspk_1 + 1, nspk_2 + 1))
        scr[:, 0] = numpy.arange(0, nspk_1 + 1)
        scr[0, :] = numpy.arange(0, nspk_2 + 1)

        if nspk_1 > 0 and nspk_2 > 0:
            for i in xrange(1, nspk_1 + 1):
                for j in xrange(1, nspk_2 + 1):
                    scr[i, j] = min(scr[i - 1, j] + 1, scr[i, j - 1] + 1)
                    scr[i, j] = min(scr[i, j], scr[i - 1, j - 1] + cost *
                        abs(self.spike_times[i - 1] - spktrain.spike_times[j - 1]))
        return scr[nspk_1, nspk_2]

    def distance_vanrossum(self, spktrain, tc=5., dt=0.1):
        """
        Returns the van Rossum metric of the two spike trians.
        dt - time bins to calculate the exponentials.
        tc - time constant / cost of the exponentials.
        """
        n = int((self.t_stop - self.t_start) / dt)
        t = numpy.linspace(self.t_start, self.t_stop, n)
        z1 = numpy.zeros(n)
        z2 = numpy.zeros(n)
        tls1 = numpy.ones(n) * -numpy.inf  # Time of last spike
        tls2 = numpy.ones(n) * -numpy.inf  # Time of last spike

        # Populate array with last spike time for each time bin.
        pst_i = -1  # previous spike time index
        pst = -1  # previous spike time
        for st in self.spike_times:
            nst_i = int((st - t[0]) / (t[1] - t[0]))  # new spike time index
            if pst_i != -1:
                tls1[pst_i + 1:nst_i + 1] = pst
            pst_i = nst_i  # update previous spike time index
            pst = st  # update previous spike time
        # for the last spike emited
        if pst_i != -1:
            tls1[pst_i + 1:n] = pst

        pst_i = -1  # previous spike time index
        pst = -1  # previous spike time
        for st in spktrain.spike_times:
            nst_i = int((st - t[0]) / (t[1] - t[0]))  # new spike time index
            if pst_i != -1:
                tls2[pst_i + 1:nst_i + 1] = pst
            pst_i = nst_i  # update previous spike time index
            pst = st  # update previous spike time
        # for the last spike emited
        if pst_i != -1:
            tls2[pst_i + 1:n] = pst

        #Convolve spike trains
        z1 = numpy.exp(-(t - tls1) / tc)
        z2 = numpy.exp(-(t - tls2) / tc)
        vrm = numpy.sqrt(numpy.square(z1 - z2).sum() / tc)
        return vrm


class emptySpikeTrain(SpikeTrain):
    def __init__(self):
        super(emptySpikeTrain, self).__init__([])
        self.t_stop = 0
        self.t_start = 1000

    @property
    def spike_times(self):
        return numpy.array([], dtype=numpy.float32)

    @spike_times.setter
    def spike_times(self, value):
        pass
        # raise AttributeError("Warning, setting emptySpikeTrain spike_times!
        # has no effect")

    def merge(self, spiketrain):
        raise TypeError("Cannot merge empty spiketrains in-place, use spikes.merge function")


def merge(*spiketrains):
    """
    Merge spike times from a spiketrain

    Inputs:
        spiketrains - The SpikeTrain that should be added

    Examples:
        >> a = SpikeTrain(range(0,100,10),0.1,0,100)
        >> b = SpikeTrain(range(400,500,10),0.1,400,500)
        >> a = merge(a, b)
        >> a.spike_times
            [   0.,   10.,   20.,   30.,   40.,   50.,   60.,   70.,   80.,
            90.,  400.,  410.,  420.,  430.,  440.,  450.,  460.,  470.,
            480.,  490.]
        >> a.t_stop
            500
    """
    new_st = SpikeTrain([])

    for st in spiketrains:
        new_st.spike_times = numpy.insert(
                          new_st.spike_times,
                          new_st.spike_times.searchsorted(st.spike_times),
                          st.spike_times)
        new_st.t_start = min(new_st.t_start, st.t_start)
        new_st.t_stop = max(new_st.t_stop, st.t_stop)

    return new_st


def merge_spikelists(*lspikelist):
    #create stim sequence
    stStim = SpikeList([], [])

    #workaround for NeuroTools bug
    tt = []
    tmarray = numpy.array([[], []], dtype='float')
    adarray = numpy.array([], dtype='float')

    #transition inumpyuts
    for i, stStimPre in enumerate(lspikelist):
        tt.append(stStimPre.raw_data())
        adarray = numpy.concatenate([stStimPre.id_list(), adarray])

    #merge
    tmarray = numpy.fliplr(numpy.vstack(tt))
    adarray = numpy.sort(numpy.unique(adarray))

    stStim = SpikeList(
        tmarray, adarray, t_start=0.0, t_stop=max(tmarray[:, 1]))
    return stStim


class SpikeList(object):
    """
    SpikeList(spikes, id_list, t_start=None, t_stop=None, dims=None)

    Return a SpikeList object which will be a dict of SpikeTrain objects.

    Inputs:
        spikes  - a list of (id,time) tuples (id being in id_list)
        id_list - the list of the ids of all recorded cells (needed for silent cells)
        t_start - begining of the SpikeList, in ms. If None, will be infered from the data
        t_stop  - end of the SpikeList, in ms. If None, will be infered from the data
        dims    - dimensions of the recorded population, if not 1D population

    t_start and t_stop are shared for all SpikeTrains object within the SpikeList

    Examples:
        >> sl = SpikeList([(0, 0.1), (1, 0.1), (0, 0.2)], range(2))
        >> type( sl[0] )
            <type SpikeTrain>

    See also
        load_spikelist
    """
    #######################################################################
    ## Constructor and key methods to manipulate the SpikeList objects   ##
    #######################################################################
    def __init__(self, spikes=[], id_list=[], t_start=None, t_stop=None, dims=None):
        """
        Constructor of the SpikeList object

        See also
            SpikeList, load_spikelist
        """
        if isinstance(spikes, SpikeList):
            id_list = spikes.id_list()
            spikes = numpy.transpose(spikes.convert("[ids, times]"))
        self._t_start = t_start
        self._t_stop = t_stop
        self.dimensions = dims
        self.spiketrains = {}
        id_list = numpy.sort(id_list)
        id_set = set(id_list) #For fast membership testing

        # Implementaion base on pure Numpy arrays, that seems to be faster for
        # large spike files. Still not very efficient in memory, because we are
        # not
        ## using a generator to build the SpikeList...

        if not hasattr(spikes, 'size'):  # is not an array:
            spikes = numpy.array(spikes, 'float32')
        N = len(spikes)

        if N > 0:
            #sorting for fast array->dictionary
            spikes = spikes[numpy.argsort(spikes[:, 0],)]

            logging.debug("sorted spikes[:10,:] = %s" % str(spikes[:10, :]))

            break_points = numpy.where(numpy.diff(spikes[:, 0]) > 0)[0] + 1
            break_points = numpy.concatenate(([0], break_points))
            break_points = numpy.concatenate((break_points, [N]))

            for idx in xrange(len(break_points) - 1):
                id = spikes[break_points[idx], 0]
                if id in id_set:
                    self.spiketrains[id] = SpikeTrain(spikes[break_points[idx]:break_points[idx + 1], 1], self.t_start, self.t_stop, presorted = False)

        self.complete(id_list)

        if len(self) > 0 and (self.t_start is None or self.t_stop is None):
            self.__calc_startstop()

        del spikes

    @property
    def t_start(self):
        return self._t_start

    @t_start.setter
    def t_start(self, t_start):
        self._t_start = t_start
        for st in self:
            st.t_start = t_start

    @property
    def t_stop(self):
        return self._t_stop

    @t_stop.setter
    def t_stop(self, t_stop):
        self._t_stop = t_stop
        for st in self:
            st.t_stop = t_stop

    def __del__(self):
        pass

    def filter_duplicates(self):
        time_window = 0.01  # in ms
        for st in self:
            isi = st.isi()
            if not isinstance(st, emptySpikeTrain):
                st.spike_times = numpy.delete(
                    st.spike_times, numpy.nonzero(isi < time_window)[0])

    def id_list(self):
        """
        Return the sorted list of all the cells ids contained in the
        SpikeList object

        Examples
            >> spklist.id_list()
                [0,1,2,3,....,9999]
        """
        #return numpy.array(self.spiketrains.keys(), int)
        return numpy.array(numpy.sort(self.spiketrains.keys()))

    def copy(self):
        """
        Return a copy of the SpikeList object
        """
        spklist = SpikeList([], [], self.t_start, self.t_stop, self.dimensions)
        for id in self.id_list():
            spklist.append(id, self.spiketrains[id])
        return spklist

    def __calc_startstop(self):
        """
        t_start and t_stop are shared for all neurons, so we take min and max values respectively.
        TO DO : check the t_start and t_stop parameters for a SpikeList. Is it commun to
        all the spikeTrains within the spikelist or each spikelistes do need its own.
        """
        if len(self) > 0:
          #  if self.t_start is None:
            start_times = numpy.array([self.spiketrains[idx].
                t_start for idx in self.id_list()], numpy.float32)
            self.t_start = numpy.min(start_times)
            logging.debug("Warning, t_start is infered from the data : %f" %
                self.t_start)
            for id in self.spiketrains.keys():
                self.spiketrains[id].t_start = self.t_start
          #  if self.t_stop is None:
            stop_times = numpy.array([self.spiketrains[idx].
                t_stop for idx in self.id_list()], numpy.float32)
            self.t_stop = numpy.max(stop_times)
            logging.debug(
                "Warning, t_stop  is infered from the data : %f" % self.t_stop)
            for id in self.spiketrains.keys():
                self.spiketrains[id].t_stop = self.t_stop
        else:
            raise Exception("No SpikeTrains")

    def __getitem__(self, id):
        if id in self.id_list():
            return self.spiketrains[id]
        else:
            raise Exception(
                "id %d is not present in the SpikeList. See id_list()" % id)

    def __getslice__(self, i, j):
        """
        Return a new SpikeList object with all the ids between i and j
        """
        ids = numpy.where((self.id_list() >= i) & (self.id_list() < j))[0]
        return self.id_slice(ids)

    #def __setslice__(self, i, j):

    def __setitem__(self, id, spktrain):
        assert isinstance(spktrain, SpikeTrain), "A SpikeList object can only contain SpikeTrain objects"
        self.spiketrains[id] = spktrain
        #self.__calc_startstop()
        if (self.t_start is None) or (spktrain.t_start < self.t_start):
            for i in self.spiketrains:
                self.spiketrains[i].t_start = spktrain.t_start
            self.t_start = spktrain.t_start
        if (self.t_stop is None) or (spktrain.t_stop > self.t_stop):
            for i in self.spiketrains:
                self.spiketrains[i].t_stop = spktrain.t_stop
            self.t_stop = spktrain.t_stop

    def __iter__(self):
        return self.spiketrains.itervalues()

    def __len__(self):
        return len(self.spiketrains)

    def __sub_id_list(self, sub_list=None):
        """
        Internal function used to get a sublist for the Spikelist id list

        Inputs:
            sublist - sub_list is a list of cell in self.id_list(). If None, id_list is returned

        Examples:
            >> self.__sub_id_list(50)
        """
        if sub_list == None:
            return self.id_list()
        elif not hasattr(sub_list, '__iter__'):
            sub_list = [sub_list]
        #elif type(sub_list) == int:
        #    return numpy.random.permutation(self.id_list())[0:sub_list]
        else:
            return sub_list

    def append(self, id, spktrain):
        """
        Add a SpikeTrain object to the SpikeList

        Inputs:
            id       - the id of the new cell
            spktrain - the SpikeTrain object representing the new cell

        The SpikeTrain object is sliced according to the t_start and t_stop times
        of the SpikeLlist object

        Examples
            >> st=SpikeTrain(range(0,100,5),0.1,0,100)
            >> spklist.append(999, st)
                spklist[999]

        See also
            concatenate, __setitem__
        """
        assert isinstance(spktrain, SpikeTrain), "A SpikeList object can only contain SpikeTrain objects"
        if id in self.id_list():
            raise Exception("id %d already present in SpikeList. Use __setitem__ (spk[id]=...) instead()" % id)
        else:
            self.spiketrains[id] = spktrain.time_slice(self.
                t_start, self.t_stop)

    def time_parameters(self):
        """
        Return the time parameters of the SpikeList (t_start, t_stop)
        """
        return (self.t_start, self.t_stop)

    def time_axis(self, time_bin):
        """
        Return a time axis between t_start and t_stop according to a time_bin

        Inputs:
            time_bin - the bin width

        See also
            spike_histogram
        """
        axis = numpy.arange(self.t_start, self.t_stop + time_bin, time_bin)
        return axis

    def concatenate(self, spklists):
        """
        Concatenation of SpikeLists to the current SpikeList.

        Inputs:
            spklists - could be a single SpikeList or a list of SpikeLists

        The concatenated SpikeLists must have similar (t_start, t_stop), and
        they can't shared similar cells. All their ids have to be different.

        See also
            append, merge, __setitem__
        """
        if isinstance(spklists, SpikeList):
            spklists = [spklists]
        # We check that Spike Lists have similar time_axis
        for sl in spklists:
            if not sl.time_parameters() == self.time_parameters():
                raise Exception("Spike Lists should have similar time_axis")
        for sl in spklists:
            for id in sl.id_list():
                self.append(id, sl.spiketrains[id])

    def merge(self, spikelist, relative=False):
        """
        For each cell id in spikelist that matches an id in this SpikeList,
        merge the two SpikeTrains and save the result in this SpikeList.
        Note that SpikeTrains with ids not in this SpikeList are appended to it.

        Inputs:
            spikelist - the SpikeList that should be merged to the current one
            relative  - if True, spike times are expressed in a relative
                        time compared to the previsous one

        Examples:
            >> spklist.merge(spklist2)

        See also:
            concatenate, append, __setitem__
        """
        for id, spiketrain in spikelist.spiketrains.items():
            if id in self.id_list():
                                # Does not take relative argument, Check
                                # SpikeList.merge?
                self.spiketrains[id] = merge(self.spiketrains[id], spiketrain)
            else:
                if relative:
                    spiketrain.relative_times()
                self.append(id, spiketrain)

    def complete(self, id_list):
        """
        Complete the SpikeList by adding Sempty SpikeTrain for all the ids present in
        ids that will not already be in the SpikeList

         Inputs:
            id_list - The id_list that should be completed

        Examples:
            >> spklist.id_list()
                [0,2,5]
            >> spklist.complete(arange(5))
            >> spklist.id_list()
                [0,1,2,3,4]
        """
        id_list = set(id_list)
        missing_ids = id_list.difference(set(self.id_list()))

        if len(missing_ids) > 0:
            empty_ST = emptySpikeTrain()
            missing_sts = zip(missing_ids, [empty_ST] * len(missing_ids))
            self.spiketrains.update(missing_sts)

    def id_slice(self, id_list):
        """
        Return a new SpikeList obtained by selecting particular ids

        Inputs:
            id_list - Can be an integer (and then N random cells will be selected)
                      or a sublist of the current ids

        The new SpikeList inherits the time parameters (t_start, t_stop)

        Examples:
            >> spklist.id_list()
                [830, 1959, 1005, 416, 1011, 1240, 729, 59, 1138, 259]
            >> new_spklist = spklist.id_slice(5)
            >> new_spklist.id_list()
                [1011, 729, 1138, 416, 59]

        See also
            time_slice, interval_slice
        """
        new_SpkList = SpikeList(
            [], [], self.t_start, self.t_stop, self.dimensions)
        id_list = self.__sub_id_list(id_list)
        for id in id_list:
            try:
                new_SpkList.append(id, self.spiketrains[id])
            except Exception:
                logging.debug("id %d is not in the source SpikeList or already in the new one" % id)
        return new_SpkList

    def time_slice(self, t_start, t_stop):
        """
        Return a new SpikeList obtained by slicing between t_start and t_stop

        Inputs:
            t_start - begining of the new SpikeTrain, in ms.
            t_stop  - end of the new SpikeTrain, in ms.

        See also
            id_slice, interval_slice
        """
        new_SpkList = SpikeList([], [], t_start, t_stop, self.dimensions)
        for id in self.id_list():
            new_SpkList.append(
                id, self.spiketrains[id].time_slice(t_start, t_stop))
        new_SpkList.__calc_startstop()
        return new_SpkList

    def time_offset(self, offset=None, t_start=None, t_stop=None):
        """
        Add an offset to the whole SpikeList object. t_start and t_stop are
        shifted from offset, so does all the SpikeTrain.

        Inputs:
            offset - the time offset, in ms

        Returns None: changes to the SpikeList are made in-place
        
        Examples:
            >> spklist.t_start
                1000
            >> spklist.time_offset(50)
            >> spklist.t_start
                1050
        """
        if offset == None:
            return 
        
        if t_start == None:
            self.t_start += offset
        else:
            self.t_start = t_start

        if t_stop == None:
            self.t_stop += offset
        else:
            self.t_stop = t_stop

        for i in self.id_list():
            self.spiketrains[i].time_offset(offset, self.t_start, self.t_stop)

    def id_offset(self, offset):
        """
        Add an offset to the whole SpikeList object. All the id are shifted
        according to an offset value.

        Inputs:
            offset - the id offset

        Examples:
            >> spklist.id_list()
                [0,1,2,3,4]
            >> spklist.id_offset(10)
            >> spklist.id_list()
                [10,11,12,13,14]
        """
        id_list = numpy.sort(self.id_list())
        N = len(id_list)

        for idx in xrange(1, len(id_list) + 1):
            id = id_list[N - idx]
            spk = self.spiketrains.pop(id)
            self.spiketrains[id + offset] = spk

    def first_spike_time(self):
        """
        Get the time of the first real spike in the SpikeList
        """
        first_spike = self.t_stop
        is_empty = True
        for id in self.id_list():
            if len(self.spiketrains[id]) > 0:
                is_empty = False
                if self.spiketrains[id].spike_times[0] < first_spike:
                    first_spike = self.spiketrains[id].spike_times[0]
        if is_empty:
            raise Exception("No spikes can be found in the SpikeList object !")
        else:
            return first_spike

    def last_spike_time(self):
        """
        Get the time of the last real spike in the SpikeList
        """
        last_spike = self.t_start
        is_empty = True
        for id in self.id_list():
            if len(self.spiketrains[id]) > 0:
                is_empty = False
                if self.spiketrains[id].spike_times[-1] > last_spike:
                    last_spike = self.spiketrains[id].spike_times[-1]
        if is_empty:
            raise Exception("No spikes can be found in the SpikeList object !")
        else:
            return last_spike

    def select_ids(self, criteria):
        """
        Return the list of all the cells in the SpikeList that will match the criteria
        expressed with the following syntax.

        Inputs :
            criteria - a string that can be evaluated on a SpikeTrain object, where the
                       SpikeTrain should be named ``cell''.

        Exemples:
            >> spklist.select_ids("cell.mean_rate() > 0") (all the active cells)
            >> spklist.select_ids("cell.mean_rate() == 0") (all the silent cells)
            >> spklist.select_ids("len(cell.spike_times) > 10")
            >> spklist.select_ids("mean(cell.isi()) < 1")
        """
        selected_ids = []
        for id in self.id_list():
            cell = self.spiketrains[id]
            if eval(criteria):
                selected_ids.append(id)
        return selected_ids

    def sort_by(self, criteria, descending=False):
        """
        Return an array with all the ids of the cells in the SpikeList,
        sorted according to a particular criteria.

        Inputs:
            criteria   - the criteria used to sort the cells. It should be a string
                         that can be evaluated on a SpikeTrain object, where the
                         SpikeTrain should be named ``cell''.
            descending - if True, then the cells are sorted from max to min.

        Examples:
            >> spk.sort_by('cell.mean_rate()')
            >> spk.sort_by('cell.cv_isi()', descending=True)
            >> spk.sort_by('cell.distance_victorpurpura(target, 0.05)')
        """
        criterias = numpy.zeros(len(self), float)
        for count, id in enumerate(self.id_list()):
            cell = self.spiketrains[id]
            criterias[count] = eval(criteria)
        result = self.id_list()[numpy.argsort(criterias)]
        if descending:
            return result[numpy.arange(len(result) - 1, -1, -1)]
        else:
            return result

    def save(self, user_file):
        '''
        Save raw data to user_file
        '''
        raw_data = self.raw_data()
        arg_sort_idx = numpy.argsort(raw_data[:, 0])
        sorted_raw = raw_data[arg_sort_idx, :]
        numpy.savetxt(user_file, sorted_raw)

    #######################################################################
    ## Analysis methods that can be applied to a SpikeTrain object       ##
    #######################################################################
    def isi(self):
        """
        Return the list of all the isi vectors for all the SpikeTrains objects
        within the SpikeList.

        See also:
            isi_hist
        """
        isis = []
        for id in self.id_list():
            isis.append(self.spiketrains[id].isi())
        return isis

    def isi_hist(self, bins=50, display=False, kwargs={}):
        """
        Return the histogram of the ISI.

        Inputs:
            bins    - the number of bins (between the min and max of the data)
                      or a list/array containing the lower edges of the bins.
            display - if True, a new figure is created. Could also be a subplot
            kwargs  - dictionary contening extra parameters that will be sent to the plot
                      function

        Examples:
            >> z = subplot(221)
            >> spklist.isi_hist(10, display=z, kwargs={'color':'r'})

        See also:
            isi
        """
        isis = numpy.concatenate(self.isi())
        values, xaxis = numpy.histogram(isis, bins=bins, normed=True)
        xaxis = xaxis[:-1]
        subplot = get_display(display)
        if not subplot or not HAVE_PYLAB:
            return values, xaxis
        else:
            xlabel = "Inter Spike Interval (ms)"
            ylabel = "Probability"
            set_labels(subplot, xlabel, ylabel)
            subplot.plot(xaxis, values, **kwargs)
                        #subplot.set_yticks([]) # arbitrary units
            pylab.draw()

    def cv_isi(self, float_only=False):
        """
        Return the list of all the CV coefficients for each SpikeTrains object
        within the SpikeList. Return NaN when not enough spikes are present

        Inputs:
            float_only - False by default. If true, NaN values are automatically
                         removed

        Examples:
            >> spklist.cv_isi()
                [0.2,0.3,Nan,2.5,Nan,1.,2.5]
            >> spklist.cv_isi(True)
                [0.2,0.3,2.5,1.,2.5]

        See also:
            cv_isi_hist, cv_local, cv_kl, SpikeTrain.cv_isi

        """
        ids = self.id_list()
        N = len(ids)
        cvs_isi = numpy.empty(N)
        for idx in xrange(N):
            cvs_isi[idx] = self.spiketrains[ids[idx]].cv_isi()

        if float_only:
            cvs_isi = numpy.extract(
                numpy.logical_not(numpy.isnan(cvs_isi)), cvs_isi)
        return cvs_isi

    def mean_rate(self, t_start=None, t_stop=None):
        """
        Return the mean firing rate averaged accross all SpikeTrains between t_start and t_stop.

        Inputs:
            t_start - begining of the selected area to compute mean_rate, in ms
            t_stop  - end of the selected area to compute mean_rate, in ms

        If t_start or t_stop are not defined, those of the SpikeList are used

        Examples:
            >> spklist.mean_rate()
            >> 12.63

        See also
            mean_rates, mean_rate_std
        """
        return numpy.mean(self.mean_rates(t_start, t_stop))
    def mean_rates_isi(self, t_start=None, t_stop=None):
        '''
        Returns a list of mean rates calculated on the basis of interspike
        interval.
        
        If t_start or t_stop are not defined, those of the SpikeList are used
        '''
        mean_rate = []
        for id in self.id_list():
            mean_rate.append(self.spiketrains[id].mean_rate_isi(t_start, 
                                                                t_stop))
        return mean_rate


    def mean_rate_std(self, t_start=None, t_stop=None):
        """
        Standard deviation of the firing rates accross all SpikeTrains
        between t_start and t_stop

        Inputs:
            t_start - begining of the selected area to compute std(mean_rate), in ms
            t_stop  - end of the selected area to compute std(mean_rate), in ms

        If t_start or t_stop are not defined, those of the SpikeList are used

        Examples:
            >> spklist.mean_rate_std()
            >> 13.25

        See also
            mean_rate, mean_rates
        """
        return numpy.std(self.mean_rates(t_start, t_stop))

    def mean_rates(self, t_start=None, t_stop=None):
        """
        Returns a vector of the size of id_list giving the mean firing rate for each neuron

        Inputs:
            t_start - begining of the selected area to compute std(mean_rate), in ms
            t_stop  - end of the selected area to compute std(mean_rate), in ms

        If t_start or t_stop are not defined, those of the SpikeList are used

        See also
            mean_rate, mean_rate_std
        """
        rates = []
        for id in self.id_list():
            rates.append(self.spiketrains[id].mean_rate(t_start, t_stop))
        return numpy.array(rates, 'float')

    def rate_distribution(self, nbins=25, normalize=True, display=False, kwargs={}):
        """
        Return a vector with all the mean firing rates for all SpikeTrains.

        Inputs:
            bins    - the number of bins (between the min and max of the data)
                      or a list/array containing the lower edges of the bins.
            display - if True, a new figure is created. Could also be a subplot
            kwargs  - dictionary contening extra parameters that will be sent to the plot
                      function

        See also
            mean_rate, mean_rates
        """
        rates = self.mean_rates()
        subplot = get_display(display)
        if not subplot or not HAVE_PYLAB:
            return rates
        else:
#            if newnum:
#                values, xaxis = numpy.histogram(
#                    rates, nbins, normed=True, new=newnum)
#                xaxis = xaxis[:-1]
#            else:
            values, xaxis = numpy.histogram(rates, nbins, normed=True)
            xlabel = "Average Firing Rate (Hz)"
            ylabel = "% of Neurons"
            set_labels(subplot, xlabel, ylabel)
            subplot.plot(xaxis, values, **kwargs)
            pylab.draw()

    def spike_histogram(self, time_bin, normalized=False, display=False, kwargs={}):
        """
        Generate an array with all the spike_histograms of all the SpikeTrains
        objects within the SpikeList.

        Inputs:
            time_bin   - the time bin used to gather the data
            normalized - if True, the histogram are in Hz (spikes/second), otherwise they are
                         in spikes/bin
            display    - if True, a new figure is created. Could also be a subplot. The averaged
                         spike_histogram over the whole population is then plotted
            kwargs     - dictionary contening extra parameters that will be sent to the plot
                         function

        See also
            firing_rate, time_axis
        """
        nbins = self.time_axis(time_bin)
        N = len(self)
        M = len(nbins)
        M -= 1
        spike_hist = numpy.zeros((N, M), numpy.float32)
        subplot = get_display(display)
        for idx, id in enumerate(self.id_list()):
            s = self.spiketrains[id].time_histogram(time_bin, normalized)
            spike_hist[idx, :len(s)] = s
        if not subplot or not HAVE_PYLAB:
            return spike_hist
        else:
            if normalized:
                ylabel = "Firing rate (Hz)"
            else:
                ylabel = "Spikes per bin"
            xlabel = "Time (ms)"
            set_labels(subplot, xlabel, ylabel)
            axis = self.time_axis(time_bin)
            axis = axis[:len(axis) - 1]
            subplot.plot(axis, numpy.mean(spike_hist, axis=0), **kwargs)
            pylab.draw()

    def firing_rate(self, time_bin, display=False, average=False, kwargs={}):
        """
        Generate an array with all the instantaneous firing rates along time (in Hz)
        of all the SpikeTrains objects within the SpikeList. If average is True, it gives the
        average firing rate over the whole SpikeList

        Inputs:
            time_bin   - the time bin used to gather the data
            average    - If True, return a single vector of the average firing rate over the whole SpikeList
            display    - if True, a new figure is created. Could also be a subplot. The averaged
                         spike_histogram over the whole population is then plotted
            kwargs     - dictionary contening extra parameters that will be sent to the plot
                         function

        See also
            spike_histogram, time_axis
        """
        result = self.spike_histogram(
            time_bin, normalized=True, display=display, kwargs=kwargs)
        if average:
            return numpy.mean(result, axis=0)
        else:
            return result

    def fano_factor(self, time_bin):
        """
        Compute the Fano Factor of the population activity.

        Inputs:
            time_bin   - the number of bins (between the min and max of the data)
                         or a list/array containing the lower edges of the bins.

        The Fano Factor is computed as the variance of the averaged activity divided by its
        mean

        See also
            spike_histogram, firing_rate
        """
        firing_rate = self.spike_histogram(time_bin)
        firing_rate = numpy.mean(firing_rate, axis=0)
        fano = numpy.var(firing_rate) / numpy.mean(firing_rate)
        return fano

    def fano_factors_isi(self):
        """
        Return a list containing the fano factors for each neuron

        See also
            isi, isi_cv
        """
        fano_factors = []
        for id in self.id_list():
            try:
                fano_factors.append(self.spiketrains[id].fano_factor_isi())
            except:
                pass

        return fano_factors

    def id2position(self, id, offset=0):
        """
        Return a position (x,y) from an id if the cells are aranged on a
        grid of size dims, as defined in the dims attribute of the SpikeList object.
        This assumes that cells are ordered from left to right, top to bottom,
        and that dims specifies (height, width), i.e. if dims = (10,12), this is
        an array with 10 rows and 12 columns, and hence has width 12 units and
        height 10 units.

        Inputs:
            id - the id of the cell

        The 'dimensions' attribute of the SpikeList must be defined

        See also
            activity_map, activity_movie
        """
        if self.dimensions is None:
            raise Exception("Dimensions of the population are not defined ! Set spikelist.dimensions")
        if len(self.dimensions) == 1:
            return id - offset
        if len(self.dimensions) == 2:
            x = (id - offset) % self.dimensions[1]
            y = self.dimensions[0] - 1 - int(numpy.floor((id -
                offset) / self.dimensions[1]))
            return (x, y)

    def position2id(self, position, offset=0):
        """
        Return the id of the cell at position (x,y) if the cells are aranged on a
        grid of size dims, as defined in the dims attribute of the SpikeList object

        Inputs:
            position - a tuple with the position of the cell

        The 'dimensions' attribute of the SpikeList must be defined and have the same shape
        as the position argument

        See also
            activity_map, activity_movie, id2position
        """
        if self.dimensions is None:
            raise Exception("Dimensions of the population are not defined ! Set spikelist.dimensions")
        assert len(position) == len(tuple(
            self.dimensions)), "position does not have the correct shape !"
        if len(self.dimensions) == 1:
            return position + offset
        if len(self.dimensions) == 2:
            return (self.dimensions[0] - 1 - position[1]) * self.dimensions[1] + position[0] + offset

    def mean_rate_variance(self, time_bin):
        """
        Return the standard deviation of the firing rate along time,
        if events are binned with a time bin.

        Inputs:
            time_bin - time bin to bin events

        See also
            mean_rate, mean_rates, mean_rate_covariance, firing_rate
        """
        firing_rate = self.firing_rate(time_bin)
        return numpy.var(numpy.mean(firing_rate, axis=0))

    def mean_rate_covariance(self, spikelist, time_bin):
        """
        Return the covariance of the firing rate along time,
        if events are binned with a time bin.

        Inputs:
            spikelist - the other spikelist to compute covariance
            time_bin  - time bin to bin events

        See also
            mean_rate, mean_rates, mean_rate_variance, firing_rate
        """
        if not isinstance(spikelist, SpikeList):
            raise Exception("Error, argument should be a SpikeList object")
        if not spikelist.time_parameters() == self.time_parameters():
            raise Exception(
                "Error, both SpikeLists should share common t_start, t_stop")
        frate_1 = self.firing_rate(time_bin, average=True)
        frate_2 = spikelist.firing_rate(time_bin, average=True)
        N = len(frate_1)
        cov = numpy.sum(
            frate_1 * frate_2) / N - numpy.sum(frate_1) * numpy.sum(frate_2) / (N * N)
        return cov

    def flatten(self, id=0.0):
        """
        Create a SpikeList with only one address *id* which is the sum of all SpikeTrains in the SpikeList
        Returns a new SpikeList.
        """
        spiketimes = self.raw_data()[:, 0]
        flat_st = SpikeList([], [])
        flat_st[id] = SpikeTrain(spiketimes)
        return flat_st

    def raster_plot_flat(self, id=0.0, **kwargs):
        """
        Flattern SpikeList with flatten(), then plot the raster

        Inputs:
            id - id of the flattened SpikeList. Can be any hashable type
            kwargs - dictionary contening extra parameters that will be passed to raster_plot
        """

        flat_st = self.flatten(id)
        return flat_st.raster_plot(**kwargs)

    def raster_plot(self, id_list=None, t_start=None, t_stop=None, display=True, kwargs={}):
        """
        Generate a raster plot for the SpikeList in a subwindow of interest,
        defined by id_list, t_start and t_stop.

        Inputs:
            id_list - can be a integer (and then N cells are randomly selected) or a list of ids. If None,
                      we use all the ids of the SpikeList
            t_start - in ms. If not defined, the one of the SpikeList object is used
            t_stop  - in ms. If not defined, the one of the SpikeList object is used
            display - if True, a new figure is created. Could also be a subplot
            kwargs  - dictionary contening extra parameters that will be sent to the plot
                      function

        Examples:
            >> z = subplot(221)
            >> spikelist.raster_plot(display=z, kwargs={'color':'r'})

        See also
            SpikeTrain.raster_plot
        """
        subplot = get_display(display)
        if id_list == None:
            id_list = self.id_list()
            spk = self
        else:
            spk = self.id_slice(id_list)
        

        if t_start is None:
            t_start = spk.t_start
        if t_stop is None:
            t_stop = spk.t_stop
        if t_start != spk.t_start or t_stop != spk.t_stop:
            spk = spk.time_slice(t_start, t_stop)

        if t_start == None:
            t_start = 0
        if t_stop == None:
            t_stop = 1.

        if not subplot or not HAVE_PYLAB:
            warnings.warn('PYLAB_ERROR')
        else:
            ids, spike_times = spk.convert(format="[ids, times]")

            idx = numpy.where(
                (spike_times >= t_start) & (spike_times <= t_stop))[0]
            if len(spike_times) > 0:
                if kwargs.has_key('linestyle') or kwargs.has_key('ls'):
                    pass
                else:
                    kwargs['ls'] = ''
                if 'marker' in kwargs:
                    pass
                else:
                    kwargs['marker'] = '.'
                subplot.plot(spike_times, ids, **kwargs)
            xlabel = "Time (ms)"
            ylabel = "Neuron"
            set_labels(subplot, xlabel, ylabel)
            idlist = spk.id_list()
            if len(idlist) > 0:
                min_id = numpy.min(idlist)
                max_id = numpy.max(idlist)
            else:
                max_id = min_id = 0
                print('Empty Spiketrain')
            length = t_stop - t_start
            set_axis_limits(subplot, t_start - 0.05 *
                length, t_stop + 0.05 * length, min_id - 2, max_id + 2)
            pylab.draw()

    #######################################################################
    ## Method to convert the SpikeList into several others format        ##
    #######################################################################
    def convert(self, format="[times, ids]", relative=False, quantized=False):
        """
        Return a new representation of the SpikeList object, in a user designed format.
            format is an expression containing either the keywords times and ids,
            time and id.

        Inputs:
            relative -  a boolean to say if a relative representation of the spikes
                        times compared to t_start is needed
            quantized - a boolean to round the spikes_times.

        Examples:
            >> spk.convert("[times, ids]") will return a list of two elements, the
                first one being the array of all the spikes, the second the array of all the
                corresponding ids
            >> spk.convert("[(time,id)]") will return a list of tuples (time, id)

        See also
            SpikeTrain.format
        """
        is_times = re.compile("times")
        is_ids = re.compile("ids")
        if len(self) > 0:
            times = numpy.concatenate([st.format(relative, quantized) for st in self.spiketrains.itervalues()])
            ids = numpy.concatenate([id * numpy.ones(len(st.spike_times), int) for id, st in self.spiketrains.iteritems()])
        else:
            times = []
            ids = []
        if is_times.search(format):
            if is_ids.search(format):
                return eval(format)
            else:
                raise Exception(
                    "You must have a format with [times, ids] or [time, id]")
        is_times = re.compile("time")
        is_ids = re.compile("id")
        if is_times.search(format):
            if is_ids.search(format):
                result = []
                for id, time in zip(ids, times):
                    result.append(eval(format))
            else:
                raise Exception(
                    "You must have a format with [times, ids] or [time, id]")
            return result

    def raw_data(self):
        """
        Function to return a N by 2 array of all times and ids.

        Examples:
            >> spklist.raw_data()
            >> array([[  1.00000000e+00,   1.00000000e+00],
                      [  1.00000000e+00,   1.00000000e+00],
                      [  2.00000000e+00,   2.00000000e+00],
                         ...,
                      [  2.71530000e+03,   2.76210000e+03]])

        See also:
            convert()
        """
        if len(self) > 0:
            times = numpy.concatenate([st.spike_times for st in self.spiketrains.itervalues()])
            ids = numpy.concatenate([id * numpy.ones(len(st.spike_times), int) for id, st in self.spiketrains.iteritems()])
        else:
            times = []
            ids = []
        return numpy.column_stack([times,ids])

    def composite_plot(self, id_list=None, t_start=None, t_stop=None, t_start_rate=None, t_stop_rate=None, display=True, kwargs={}, kwargs_bar={}):
        """
        Make a nice Composite plot, *i.e.* a raster plot combined with a vertical rate plot.

        The arguments are identical to STPlotRaster, except for display. If True is given a new figure is created. If [axS,axR] is given, where axS and axR are pylab.axes objects, then the spike rater and the rate plot are plotted there.
        """
        self._SpikeList__calc_startstop()

        if id_list == None:
            id_list = self.id_list()
        if t_start is None:
            t_start = self.t_start
        if t_stop is None:
            t_stop = self.t_stop
        if t_start_rate is None:
            t_start_rate = t_start
        if t_stop_rate is None:
            t_stop_rate = t_stop

        if display == True:
            h = pylab.figure()
            axS = pylab.axes([0.12, 0.12, 0.57, 0.8])
            axR = pylab.axes([0.75, 0.12, 0.20, 0.8])
        elif isinstance(display, list):
            axS = display[0]
            axR = display[1]

        self.raster_plot(id_list=id_list, t_start=t_start,
            t_stop=t_stop, display=axS, kwargs=kwargs)

        min_addr = int(numpy.min(self.id_list()))
        max_addr = int(numpy.max(self.id_list()))
        axS.set_xlim([t_start, t_stop])
        axS.set_yticks([min_addr, (max_addr - min_addr) / 2, max_addr])
        axS.set_xticks(
            [int(t_start), 100 * (int(t_stop - t_start) / 200), int(t_stop)])
        axR.barh(id_list, self.mean_rates(t_start=t_start_rate, t_stop=t_stop_rate), linewidth=0, **kwargs_bar)  # remove space between bars, remove black lines
        axR.set_ylim(axS.set_ylim())
        axR.set_yticks([])
        axR.grid('on')
        #axS.xaxis.grid('on')

        axR.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(2))

        pylab.axes(axR)
        pylab.xlabel('Frequency (Hz)')

        return axS, axR


def set_axis_limits(subplot, xmin, xmax, ymin, ymax):
    """
    Defines the axis limits of a plot.

    Inputs:
        subplot     - the targeted plot
        xmin, xmax  - the limits of the x axis
        ymin, ymax  - the limits of the y axis

    Example:
        >> x = range(10)
        >> y = []
        >> for i in x: y.append(i*i)
        >> pylab.plot(x,y)
        >> plotting.set_axis_limits(pylab, 0., 10., 0., 100.)
    """
    if hasattr(subplot, 'xlim'):
        subplot.xlim(xmin, xmax)
        subplot.ylim(ymin, ymax)
    elif hasattr(subplot, 'set_xlim'):
        subplot.set_xlim(xmin, xmax)
        subplot.set_ylim(ymin, ymax)
    else:
        raise Exception('ERROR: The plot passed to function NeuroTools.plotting.set_axis_limits(...) does not provide limit defining functions.')


def get_display(display):
    """
    Returns a pylab object with a plot() function to draw the plots.

    Inputs:
        display - if True, a new figure is created. Otherwise, if display is a
                  subplot object, this object is returned.
    """
    if display is False:
        return None
    elif display is True:
        pylab.figure()
        return pylab
    else:
        return display


def set_labels(subplot, xlabel, ylabel):
    """
    Defines the axis labels of a plot.

    Inputs:
        subplot - the targeted plot
        xlabel  - a string for the x label
        ylabel  - a string for the y label

    Example:
        >> x = range(10)
        >> y = []
        >> for i in x: y.append(i*i)
        >> pylab.plot(x,y)
        >> plotting.set_labels(pylab, 'x', 'y=x^2')
    """
    if hasattr(subplot, 'xlabel'):
        subplot.xlabel(xlabel)
        subplot.ylabel(ylabel)
    elif hasattr(subplot, 'set_xlabel'):
        subplot.set_xlabel(xlabel)
        subplot.set_ylabel(ylabel)
    else:
        raise Exception('ERROR: The plot passed to function NeuroTools.plotting.set_label(...) does not provide labelling functions.')

def merge_sequencers(*l_seq):
    '''
    merge_sequencers(stim1, stim2 ......)
    Merges (soon to be) sequencer objects. These are typically returend by pyNCS.group spiketrain functions and used in setup.run
    '''
    from collections import defaultdict
    new_seq = defaultdict(list)
    merged_seq = dict()
    for seq in l_seq:
        for k, s in seq.iteritems():
            new_seq[k].append(s)
    for k, lseq in new_seq.iteritems():
        merged_seq[k] = merge_spikelists(*lseq)
    return merged_seq
