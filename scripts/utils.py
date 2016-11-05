# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 13:20:26 2016

@author: stephenshank
"""

import os

import numpy as np
from scipy.io import loadmat
from joblib import Memory


SAMPLING_RATE = 400
directory_root = os.getenv('EEG_DATA_DIR')
cache_directory = directory_root + 'cache'
mem = Memory(cachedir=cache_directory, verbose=0)

def load_temporal_data(path):
    try: 
        mat = loadmat(directory_root + path)
        return mat['dataStruct']['data'][0][0].transpose()
    except:
        return np.empty(0)


@mem.cache
def load_power_data(path):
    temporal_data = load_temporal_data(path)
    if temporal_data.size == 0:
        return np.empty(0), np.empty(0)
    n = temporal_data.shape[1]
    freq = np.array(np.fft.fftfreq(n)*SAMPLING_RATE)[:n//2]
    power = np.abs(np.fft.fft(temporal_data))[:,:n//2]
    return freq, power
