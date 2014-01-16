#from distribute_setup import use_setuptools
#use_setuptools()
# -*- coding: utf-8 -*-
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


setup(name='neusa',
	version='0',
	description='Neural Sampling',
	author='Emre Neftci',
	author_email='nemre@ucsd.edu',
	url='ssh://git@bitbucket.org/eneftci/neural_sampling_brian.git',
	packages = ['neusa'],
    package_dir = {'neusa':'src/neural_sampling'},
    )
