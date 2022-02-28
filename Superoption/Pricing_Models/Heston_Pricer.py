from subprocess import call
import numpy as np 

def option_price(kappa, theta, sigma, rho, initilal_vol, interest_rate ,time_to_maturity ,asset_price , strike, option_type, lambda_ = 0,\
                mean = 0, sigma2 = 0):
    probability_1 = CDF_1(kappa, theta, sigma, rho, initilal_vol, interest_rate ,time_to_maturity ,asset_price , strike, lambda_,\
                mean, sigma2)
    probability_2 = CDF_2(kappa, theta, sigma, rho, initilal_vol, interest_rate ,time_to_maturity ,asset_price , strike, lambda_,\
                mean, sigma2)
    if option_type == 'C':
        option_price = np.maximum(asset_price * probability_1 - strike * np.exp(-interest_rate * time_to_maturity) * probability_2, 0)
    else : 
        option_price = np.maximum(strike * np.exp(-interest_rate * time_to_maturity) * (1 - probability_2) - asset_price * (1 - probability_1), 0)
    return option_price

def density_fcts_manager(kappa, theta, sigma, rho, initilal_vol, interest_rate ,time_to_maturity ,asset_price , strike, status, lambda_ = 0,\
                mean = 0, sigma2 = 0):
    if status == 1:
        f_inv = density_fcts(-1j, kappa, theta, sigma, rho, initilal_vol, interest_rate ,time_to_maturity ,asset_price , strike, lambda_, mean , sigma2)
        integrand = lambda phi: (np.exp(-1j * phi * np.log(strike)) * density_fcts(phi - 1j, kappa, theta, sigma, rho, initilal_vol, interest_rate ,time_to_maturity ,asset_price , strike, lambda_, mean , sigma2) /(1j * phi * f_inv))
    else :
        integrand = lambda phi: (np.exp(-1j * phi * np.log(strike)) * density_fcts(phi, kappa, theta, sigma, initilal_vol, interest_rate ,time_to_maturity ,asset_price , strike, lambda_, mean , sigma2) /1j)

def CDF_1(kappa, theta, sigma, rho, initilal_vol, interest_rate ,time_to_maturity ,asset_price , strike, lambda_,\
                mean, sigma2):
    return density_fcts_manager(kappa, theta, sigma, rho, initilal_vol, interest_rate ,time_to_maturity ,asset_price , strike, lambda_,\
                mean, sigma2)

def CDF_2(kappa, theta, sigma, rho, initilal_vol, interest_rate ,time_to_maturity ,asset_price , strike, lambda_,\
                mean, sigma2):
    return density_fcts_manager(kappa, theta, sigma, rho, initilal_vol, interest_rate ,time_to_maturity ,asset_price , strike, lambda_,\
                mean, sigma2)

def density_fcts(phi, kappa, theta, sigma, rho, initilal_vol, interest_rate ,time_to_maturity ,asset_price , strike, lambda_,\
                mean, sigma2):

    x = np.log(asset_price)
    d = np.sqrt((rho * sigma * phi * 1j - kappa)**2 + sigma**2 * (phi * 1j + phi**2))
    g = (kappa - rho * sigma * phi * 1j - d) / (kappa - rho * sigma * phi * 1j + d)
    C = (initilal_vol/sigma**2) * (kappa - rho * sigma * phi * 1j - d) * (1 - np.exp(-d * time_to_maturity)) / (1 - g * np.exp(-d * time_to_maturity))
    B = (theta * kappa / sigma**2) * (time_to_maturity * (kappa - rho * sigma * phi * 1j - d) - 2 * np.log((1 - g * np.exp(-d * time_to_maturity))/( 1 - g )))
    H = lambda_ * time_to_maturity *((( 1 + mean )**(1j * phi)) * np.exp(.5 * 1j * phi * sigma2**2 * (1j * phi -1 ) ) - 1 )
    D = -lambda_  * mean * phi * 1j * time_to_maturity  + H
    res = np.exp(C + B + 1j * phi * x + D)
    return res