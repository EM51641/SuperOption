from US_EU_Pricing_Functions.EuropeanCallPricing import EuropeanCall
from US_EU_Pricing_Functions.EuropeanPutPricing import EuropeanPut
import scipy 


def Optimize_call(asset_volatility,args):

    market_price, asset_price, strike_price, time_to_expiration, risk_free_rate,\
        dividend_yield = args
    replicator = EuropeanCall(asset_price, asset_volatility, strike_price,\
                              time_to_expiration, risk_free_rate, dividend_yield)
    argmin = replicator - market_price

    return argmin


def Optimize_put(asset_volatility,args):
    
    market_price, asset_price, strike_price, time_to_expiration, risk_free_rate,\
        dividend_yield = args
    replicator = EuropeanPut(asset_price, asset_volatility, strike_price,\
                              time_to_expiration, risk_free_rate, dividend_yield)
    argmin = replicator - market_price

    return argmin

def implied_volatility(price, asset_price, strike_price, time_to_expiration, \
    risk_free_rate, dividend_yield,Option_type):

    args = [price, asset_price, strike_price, time_to_expiration, risk_free_rate, dividend_yield]
    if Option_type == 'C' :
        res = scipy.optimize.brentq(Optimize_call, 0.0001, 1000, args, maxiter=10000)
    else : 
        res = scipy.optimize.brentq(Optimize_put, 0.0001, 1000, args, maxiter=10000)
    return res