#!/bin/python
#-----------------------------------------------------------------------------
# File Name : plot_options.py
# Purpose:
#
# Author: Emre Neftci
#
# Creation Date : 24-04-2013
# Last Modified : Thu 19 Dec 2013 12:54:20 PM PST
#
# Copyright : (c) 
# Licence : GPLv2
#----------------------------------------------------------------------------- 
import matplotlib, pylab
matplotlib.rcParams['text.usetex']=False
matplotlib.rcParams['savefig.dpi']=800.
matplotlib.rcParams['font.size']=25.0
matplotlib.rcParams['figure.figsize']=(8.0,6.0)
matplotlib.rcParams['axes.formatter.limits']=[-10,10]
matplotlib.rcParams['figure.subplot.bottom'] = .2
pylab.ion()

