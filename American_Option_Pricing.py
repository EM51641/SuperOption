class American_Option_Pricing:
    def binomial_tree_call_draw(self,N,q,T,S0,sigma,r,K,call=True):
        global MCMCC
        dt=T/N#setting the steps
        u=np.exp(sigma*np.sqrt(dt))
        d=1/u
        p=(np.exp((r-q)*dt)-d)/(u-d)
        MCMCC=p
    
        price_tree=np.zeros([N+1,N+1])
        for i in range(N+1):
            for j in range(i+1):
                price_tree[j,i]=S0*(d**j)*(u**(i-j))#
    #Setting option values
        option=np.zeros([N+1,N+1])
        if call:#Payoffs
            option[:,N]=np.maximum(np.zeros(N+1),price_tree[:,N]-K)
        else:
            option[:,N]=np.maximum(np.zeros(N+1),K-price_tree[:,N])
    
        for i in np.arange(N-1,-1,-1):
            for j in np.arange(0,i+1):
                option[j,i]=np.exp(-r*dt)*(p*option[j,i+1]+(1-p)*option[j+1,i+1])
                
        CT=pd.DataFrame(price_tree[:,-1])
        P_call=len(CT.mask(CT<=K).dropna())/len(CT)
        
        return[option[0,0],price_tree,option,P_call]
    def binomial_tree_put_draw(self,N,q,T,S0,sigma,r,K,call=False):
        
        dt=T/N #setting the steps
        u=np.exp(sigma*np.sqrt(dt))
        d=1/u
        p=(np.exp((r-q)*dt)-d)/(u-d)#Probability to execute the option,q is the dividend yield for the stock
    
        price_tree=np.zeros([N+1,N+1])
        for i in range(N+1):
            for j in range(i+1):
                price_tree[j,i]=S0*(d**j)*(u**(i-j))#
    #Setting option values
        option=np.zeros([N+1,N+1])
        if call:#Payoffs
            option[:,N]=np.maximum(np.zeros(N+1),price_tree[:,N]-K)
        else:
            option[:,N]=np.maximum(np.zeros(N+1),K-price_tree[:,N])
    
        for i in np.arange(N-1,-1,-1):
            for j in np.arange(0,i+1):
                option[j,i]=np.exp(-r*dt)*(p*option[j,i+1]+(1-p)*option[j+1,i+1])
                
        CT=pd.DataFrame(price_tree[:,-1])
        P_put=len(CT.mask(CT>=K).dropna())/len(CT)
        
        return[option[0,0],price_tree,option,P_put]
        
        
        
    def __init__(self, N, q, T,S0, sigma,r,K):
        
        self.N = N
        self.q = q
        self.T = T
        self.S0 = S0
        self.sigma = sigma
        self.r=r
        self.K=K
        self.binomial_tree_Pricing_call_d = self.binomial_tree_call_draw(N,q,T,S0,sigma,r,K)
        self.binomial_tree_Pricing_put_d = self.binomial_tree_put_draw(N,q,T,S0,sigma,r,K)

