#Stock Pressures and vizualisation algorithms using a bollinger band:
class technical_chart:
    
    
    def stock_analysis(self,stock_analyzed):
        stock_analyzed['LB']=(stock_analyzed.Close.rolling(21).mean()-2*stock_analyzed.Close.rolling(21).std())
        stock_analyzed['HB']=(stock_analyzed.Close.rolling(21).mean()+2*stock_analyzed.Close.rolling(21).std())
        stock_analyzed['Hpercent']=((stock_analyzed.HB-stock_analyzed.Close)/(stock_analyzed.Close))
        stock_analyzed['Lpercent']=((stock_analyzed.LB-stock_analyzed.Close)/(stock_analyzed.Close))
        return stock_analyzed#['Hpercent'],stock_analyzed['Lpercent'],stock_analyzed['Close']
    
    def percentB_belowzero(self,stock_analyzed):
        percentB=self.stock_analysis(stock_analyzed)['Lpercent']
        price=self.stock_analysis(stock_analyzed)['Close']
        signal   = []
        previous = -1.0
        for date,value in percentB.iteritems():
            if value > 0 and previous < 0:
                signal.append(price[date]*0.99)
            else:
                signal.append(np.nan)
            previous = value
        return signal

    def percentH_abovezero(self,stock_analyzed):
        percentH=self.stock_analysis(stock_analyzed)['Hpercent']
        price=self.stock_analysis(stock_analyzed)['Close']
        signal   = []
        previous = 1.0
        for date,value in percentH.iteritems():
            if value < 0 and previous >= 0:
                signal.append(price[date]*1.01)
            else:
                signal.append(np.nan)
            previous = value
        return signal
    
    def __init__(self,stock_analyzed):
        
        self.stock_analyzed = stock_analyzed
        self.below_pressure=self.percentB_belowzero(stock_analyzed)
        self.above_pressure=self.percentH_abovezero(stock_analyzed)
        self.analysis=self.stock_analysis(stock_analyzed)
