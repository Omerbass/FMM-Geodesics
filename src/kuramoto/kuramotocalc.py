import scipy as sp
import numpy as np

def freqdist_lorenzian(omega, gamma=1.0):
    return gamma / (np.pi * (omega**2 + gamma**2))

def kuramoto_critical_coupling(gamma=1.0, D=1.0):
    return 2.0 / (gamma + D)