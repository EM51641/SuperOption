import numba as nb 
import numpy as np

@nb.jit(nopython = True, parallel = True,fastmath = True)
def BS_MC(replication_nb, path_length, variance, randn, strike, spot, rate, div, dt, expiry):
    callT = np.zeros(replication_nb, path_length)
    path = np.zeros(replication_nb, path_length)
    path[:,0] = spot
    for i in nb.prange(0, replication_nb):
        for j in nb.prange(1, path_length):
            path[i,j] = path[i,j-1] * np.exp((rate - div - 0.5 * variance) * dt + np.sqrt(variance * dt) * randn[i,j]) #Spot Simulation
        callT[i] =  np.maximum(path[i,-1] - strike, 0.0)
    callT = np.mean(callT) * np.exp(-rate * expiry) 
    return callT , path

@nb.jit(nopython = True, parallel=True, fastmath = True)
def Heston_MC(replication_nb, path_length, variance, kappa, theta, sigma, path, randn1, randn2, strike, spot, rate, div, dt, expiry):
    var = np.zeros(replication_nb, path_length)
    callT = np.zeros(replication_nb, path_length)
    path = np.zeros(replication_nb, path_length)
    var[:,0] = variance
    path[:,0] = spot
    for i in nb.prange(0, replication_nb):
        for j in nb.prange(1, path_length):
            var[i,j] = var[i,j-1] + kappa * (theta - var[i,j-1]) * dt + sigma * np.sqrt(var[i,j-1] * dt) * randn1[i,j] #Variance Simulation
            path[i,j] = path[i,j-1] * np.exp((rate - div - 0.5 * var[i,j]) * dt + np.sqrt(var[i,j] * dt) * randn2[i,j]) #Spot Simulation
        callT[i] =  np.maximum(path[i,-1] - strike, 0.0)
    callT = np.mean(callT) * np.exp(-rate * expiry) 
    return callT, path, var