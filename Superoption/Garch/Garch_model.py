from arch import arch_model
import numpy as np 
import pandas as pd

class GARCH_MODEL_AR_1:
    
    def GARCH_AR_1(self,returns, horizon):
        
        am = arch_model(returns * 100, vol='Garch', rescale=False, p=1, q=1, o=0, dist='StudentsT', mean='AR', lags=1) #Use Garch with whatever distribution you want
        res = am.fit(update_freq=5,disp='off')
        fitted = pd.DataFrame(.1 * np.sqrt(res.params['omega'] + res.params['alpha[1]'] * res.resid**2 + res.conditional_volatility**2 * res.params['beta[1]']))
        v = (returns.rolling(21).std()) * 252 ** (1/2)  
        forecasts = res.forecast(horizon = horizon)
        A = .1 * np.sqrt(forecasts.variance.dropna().T)
        A.index = pd.date_range(start = returns.index[-1] + pd.DateOffset(days = 1), end=returns.index[-1] + pd.DateOffset(days = horizon))
        return res.summary(), fitted, A, v, A.iloc[-1,0]
        
    
    def __init__(self,returns,horizon):
        self.returns = returns
        self.horizon = horizon
        self.garch = self.GARCH_AR_1(returns, horizon)
