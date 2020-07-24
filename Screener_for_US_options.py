class Screener_US:
    
    def American_Screening(self,_list_):
        i_r=TNX[-1]/100
        S_name=[]
        C_value=[]
        P_value=[]
        S_Price=[]
        for stock in _list_:
            print(("\n pulling {} from Yahoo").format(stock))
            df=pd.DataFrame()
            try:
                df[stock]=wb.DataReader(stock,start=start_date,end=end_date,data_source='yahoo')['Adj Close']
                v=GARCH_MODEL_AR_1(df,63).garch[-1]
            except:
                print("N/A stock")
        
            if yn=='Y':
                try:
                    d_y=yf.Ticker(stock).info['dividendYield']
                    if d_y is None:
                        d_y=0
                except:
                    print("This stock doesn't distribute dividends")
                    d_y=0
            else:
                d_y=0
            
            try:
                last_sport_price = df.iloc[-1][0]
                returns=df.pct_change().dropna()
                drift=(returns.mean()*252-(1/2)*(v**2))[0]
                T = 1
                N = 252 # Number of points, number of subintervals = N-1
                dt = 1/N # Time step
                t = np.linspace(0,T,N)
                M = 500 # Number of walkers/paths
                dX = np.exp((drift)*dt-(np.sqrt(dt) * np.random.randn(M, N)*v))
                X = df.iloc[-1][0]*np.cumprod(dX, axis=1)[:,:63]
                Strike_Price=X.mean()
                result_set=American_Option_Pricing(N=int(round(63/21,1)),T=90/360,q=d_y,sigma=v,r=i_r,K=Strike_Price,S0=last_sport_price).binomial_tree_Pricing_call_d[-1]
                c = result_set
                print("Probability of exercice of the call: "+str(c))
                result_set=American_Option_Pricing(N=int(round(63/21,1)),T=90/360,q=d_y,sigma=v,r=i_r,K=Strike_Price,S0=last_sport_price).binomial_tree_Pricing_put_d[-1]
                p = result_set
                print("Probability of exercice of the Put: "+str(p))
                if customer=='B':
                    if p>0.70 or c>0.70:
                        S_name.append(stock)
                        C_value.append(c)
                        P_value.append(p)
                        S_Price.append(Strike_Price)
                elif customer=='S':
                    if p<0.30 or c<0.30:
                        S_name.append(stock)
                        C_value.append(c)
                        P_value.append(p)
                        S_Price.append(Strike_Price)
            except:
                print('No Available datas')
        
        print("Your document is registred under the name:'ScreenOutput.xlsx'")        
        exportList=pd.DataFrame({'Stock':S_name, "Probability of executing a call":C_value, "Probability of executing a put":P_value,"Strike Price":S_Price})
        print(exportList)
        writer = ExcelWriter("ScreenOutput.xlsx")
        exportList.to_excel(writer, "Sheet1")
        writer.save()
        return exportList
        
    
        
    def __init__(self,_list_):
        self.liste=_list_
        self.AMScreen=self.American_Screening(_list_)
