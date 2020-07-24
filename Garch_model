class GARCH_MODEL_AR_1:
    
    def GARCH_AR_1(self,s,d):
        
        returns=s.pct_change().dropna()*100
        am = arch_model(returns,vol='Garch',rescale=False,p=1,q=1,o=0,dist='StudentsT',mean='AR', lags=1)#Here I preferred to use a student t distribution as it reflects better the real distribution of returns 
        res = am.fit(update_freq=5,disp='off')
        fitted = 0.1*np.sqrt(res.params['omega'] +res.params['alpha[1]'] *res.resid**2 +res.conditional_volatility**2 *res.params['beta[1]'])
        fitted = pd.DataFrame(fitted)
        v = (returns.rolling(21).std())*252**(1/2)  
        forecasts = res.forecast(horizon=d)
        A = 0.1*np.sqrt(forecasts.variance.dropna().T)
        A.index = pd.date_range(start=s.index[-1]+ pd.DateOffset(days=1), end=s.index[-1] + pd.DateOffset(days=d))
        
        return res.summary(),fitted,A,v,A.iloc[-1,0]
        
    
    def __init__(self,s,d):
        self.s=s
        self.d=d
        self.garch=self.GARCH_AR_1(s,d)
