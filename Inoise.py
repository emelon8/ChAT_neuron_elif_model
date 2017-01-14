# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 14:59:57 2016

@author: eric
"""

import numpy as np
from scipy import randn, sqrt

def intrinsic_noise(dt,T,t,filterfrequency,gnoise):
    ######################################################### ChAT noise generation
    Inoise = np.zeros((1,len(t)))
    dt_ins = dt/1000
    df = 1/(T+dt_ins) # freq resolution
    fidx = np.arange(1,len(t)/2,1) # it has to be N/2 pts, where N=len(t); Python makes a range from 1 to np.ceil(len(t)/2)-1
    faxis = (fidx-1)*df
    #make the phases
    Rr = randn(np.size(fidx)) # ~N(0,1) over [-1,1]
    distribphases = np.exp(1j*np.pi*Rr) # on the unit circle
    #make the amplitudes - filtered
    filterf = sqrt(1/((2*np.pi*filterfrequency)**2+(2*np.pi*faxis)**2))

    fourierA = distribphases*filterf # representation in fourier domain
    # make it conj-symmetric so the ifft is real
    fourierB = fourierA.conj()[::-1]
    nss = np.concatenate(([0],fourierA,fourierB))
    Inoise[0,:] = np.fft.ifft(nss)
    scaling = np.std(Inoise, ddof=1)
    Inoise = Inoise/scaling
    Inoise = Inoise*gnoise
    ######################################################### ChAT noise generation
    return Inoise