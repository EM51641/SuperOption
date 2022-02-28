import math 
import numpy as np
import scipy 

class EuropeanPut:

    def put_delta(
        self, asset_price, asset_volatility, strike_price,
        time_to_expiration, risk_free_rate, dividend_yield
            ):
        b = math.exp(-risk_free_rate*time_to_expiration)
        x1 = math.log(asset_price/(strike_price)) + (.5*(asset_volatility**2)+risk_free_rate-dividend_yield)*time_to_expiration
        x1 = x1/(asset_volatility*(time_to_expiration**.5))
        z1 = scipy.stats.norm.cdf(x1)
        return z1 - 1

    def put_price(
        self, asset_price, asset_volatility, strike_price,
        time_to_expiration, risk_free_rate,dividend_yield
            ):
        b  = math.exp(-risk_free_rate*time_to_expiration)
        x1 = math.log(asset_price/(strike_price))+(.5*(asset_volatility**2)+risk_free_rate-dividend_yield)*time_to_expiration
        x1 = x1/(asset_volatility*(time_to_expiration**.5))
        z1 = scipy.stats.norm.cdf(-x1)
        z1 = z1*asset_price*math.exp(-q*time_to_expiration)
        x2 = math.log(asset_price/(strike_price)) - (.5*(asset_volatility**2)-risk_free_rate+dividend_yield)*time_to_expiration
        x2 = x2/(asset_volatility*(time_to_expiration**.5))
        z2 = scipy.stats.norm.cdf(-x2)
        z2 = b*strike_price*z2
        return z2-z1

    def __init__(
        self, asset_price, asset_volatility, strike_price,
        time_to_expiration, risk_free_rate,dividend_yield
            ):
        self.asset_price = asset_price
        self.asset_volatility = asset_volatility
        self.strike_price = strike_price
        self.time_to_expiration = time_to_expiration
        self.risk_free_rate = risk_free_rate
        self.dividend_yield=dividend_yield
        self.price = self.put_price(asset_price, asset_volatility, strike_price, time_to_expiration, risk_free_rate, dividend_yield)
        self.delta = self.put_delta(asset_price, asset_volatility, strike_price, time_to_expiration, risk_free_rate, dividend_yield)

