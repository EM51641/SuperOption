#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as wb
from datetime import timedelta, date
import time
from yahoo_fin.options import *
import yfinance as yf
from pandas import ExcelWriter
from yahoo_fin import stock_info as si
from arch import arch_model
import math
import random
from scipy.stats import norm
from pandas.plotting import register_matplotlib_converters
import mplfinance as mpf
from numba import jit
register_matplotlib_converters()
yf.pdr_override()


# In[ ]:


class OptionTools:

    def __init__(self):
        pass

    # Return annualized remaining time to maturity and days to maturity for simulations
    def compute_time_to_expiration(self, Y, M, D):
        d0 = date.today()
        d1 = date(Y, M, D)
        delta = d1 - d0
        return delta.days/365, delta.days
    
    # Testing the basic implmentation for a call
    def generate_random_option(self, n,call=True):
        options = []
        for i in range(0, n):
            # NOTE: These parameters will determine the model's performance and capabilities...
            asset_price = random.randrange(10, 30)
            asset_volatility = random.random()
            strike_price = random.randrange(10, 30)
            time_to_expiration = random.randrange(30, 364)/365 # If we have to many observations expiring tomorrow the model may just predict zero as the option is almost worthless
            risk_free_rate = random.random()
            if call:
                options.append(EuropeanCall(asset_price, asset_volatility, strike_price, time_to_expiration, risk_free_rate))
            else:
                options.append(EuropeanPut(asset_price, asset_volatility, strike_price, time_to_expiration, risk_free_rate))
        return options

    # Simulate options, returns a set of OptionSimulations
    def simulate_calls(self, n_time_steps, n_options, strike_price, initial_asset_price, drift, delta_t, asset_volatility, risk_free_rate, time_to_expiration,q):
        # List of stochastic processes modeling the underlying asset
        stochastic_processes = []
        # Generate a stochastic process for each option
        for i in range(n_options):
            stochastic_processes.append(StochasticProcess(initial_asset_price, drift, delta_t, asset_volatility)) # Note delta t is annualized

        # Make n_time_steps for each stochastic process
        for stochastic_process in stochastic_processes:
            # Make n_time_steps for each process
            for i in range(n_time_steps):
                # Take a time step in the stochastic process
                stochastic_process.time_step()

        # List of option simulations holding realized option variables at every time step
        option_simulations = []
        # Generate n_options simulations classes to hold each observation
        for i in range(n_options):
            # Create an option simulation for every sample path to hold the option variables (prie, delta, etc...)
            option_simulations.append(OptionSimulation(initial_asset_price, asset_volatility, strike_price, time_to_expiration, risk_free_rate))

        # For each stochastic process realization and option simulation
        for z in range(n_options):
            # Reset the decrement for the next option simulation
            time_to_expiration_var = time_to_expiration
            # Price the option for each asset price in the stochsatic process given by z stored in the option simulation given by z
            for i in range(len(stochastic_processes[z].asset_prices)):
                # Check if we still have time in the option
                if (time_to_expiration_var - stochastic_processes[z].delta_t) > 0: # Avoid loss of percision down to 0
                    # Create a european call to record the variables at the z stochsatic processes's i asset price and other static variables with the z stochastic process
                    e = EuropeanCall(stochastic_processes[z].asset_prices[i], stochastic_processes[z].asset_volatility, strike_price, time_to_expiration_var, risk_free_rate,q)
                    # Append all variables for the i asset price in this z stochastic process
                    option_simulations[z].option_prices.append(e.price)
                    option_simulations[z].option_deltas.append(e.delta)
                    option_simulations[z].asset_prices.append(stochastic_processes[z].asset_prices[i])
                # Decrement the time_to_expiration by the step in time within the stochastic process, even though z iterates through each stochasstic process the step in time is constant acorss all of them
                if (time_to_expiration_var - stochastic_processes[z].delta_t) > 0:
                    time_to_expiration_var -= stochastic_processes[z].delta_t
                # Break the loop if we are out of time steps, go to the next stochastic process and price an option simulation for it
                else:
                    break
        # Return the option simulations for further analysis
        return option_simulations
    def simulate_puts(self, n_time_steps, n_options, strike_price, initial_asset_price, drift, delta_t, asset_volatility, risk_free_rate, time_to_expiration,q):
        # List of stochastic processes modeling the underlying asset
        stochastic_processes = []
        # Generate a stochastic process for each option
        for i in range(n_options):
            stochastic_processes.append(StochasticProcess(initial_asset_price, drift, delta_t, asset_volatility)) # Note delta t is annualized

        # Make n_time_steps for each stochastic process
        for stochastic_process in stochastic_processes:
            # Make n_time_steps for each process
            for i in range(n_time_steps):
                # Take a time step in the stochastic process
                stochastic_process.time_step()

        # List of option simulations holding realized option variables at every time step
        option_simulations = []
        # Generate n_options simulations classes to hold each observation
        for i in range(n_options):
            # Create an option simulation for every sample path to hold the option variables (prie, delta, etc...)
            option_simulations.append(OptionSimulation(initial_asset_price, asset_volatility, strike_price, time_to_expiration, risk_free_rate))

        # For each stochastic process realization and option simulation
        for z in range(n_options):
            # Reset the decrement for the next option simulation
            time_to_expiration_var = time_to_expiration
            # Price the option for each asset price in the stochsatic process given by z stored in the option simulation given by z
            for i in range(len(stochastic_processes[z].asset_prices)):
                # Check if we still have time in the option
                if (time_to_expiration_var - stochastic_processes[z].delta_t) > 0: # Avoid loss of percision down to 0
                    # Create a european put to record the variables at the z stochsatic processes's i asset price and other static variables with the z stochastic process
                    e = EuropeanPut(stochastic_processes[z].asset_prices[i], stochastic_processes[z].asset_volatility, strike_price, time_to_expiration_var, risk_free_rate,q)
                    # Append all variables for the i asset price in this z stochastic process
                    option_simulations[z].option_prices.append(e.price)
                    option_simulations[z].option_deltas.append(e.delta)
                    option_simulations[z].asset_prices.append(stochastic_processes[z].asset_prices[i])
                # Decrement the time_to_expiration by the step in time within the stochastic process, even though z iterates through each stochasstic process the step in time is constant acorss all of them
                if (time_to_expiration_var - stochastic_processes[z].delta_t) > 0:
                    time_to_expiration_var -= stochastic_processes[z].delta_t
                # Break the loop if we are out of time steps, go to the next stochastic process and price an option simulation for it
                else:
                    break
        # Return the option simulations for further analysis
        return option_simulations

    # Takes a set of option simulations returns a vector output of average option price at end of option life, max simulated price, initial simulated price, and min simulated price
    def simulation_analysis(self, option_simulations):
        initial_option_price = 0
        max_option_price = 0
        average_option_price = 0
        min_option_price = 0
        options_in_the_money = 0
        options_out_of_the_money = 0
        ending_prices = []
        # For each option simulation
        for option_simulation in option_simulations:
            # Set initial option price
            initial_option_price = option_simulation.option_prices[0]
            # Get Max Option Price
            if option_simulation.option_prices[len(option_simulation.option_prices)-1] > max_option_price:
                max_option_price = option_simulation.option_prices[len(option_simulation.option_prices)-1]
            # Get Min option price
            if option_simulation.option_prices[len(option_simulation.option_prices)-1] < min_option_price:
                min_option_price = option_simulation.option_prices[len(option_simulation.option_prices)-1]
            # Store for average ending option price
            ending_prices.append(option_simulation.option_prices[len(option_simulation.option_prices)-1])
        return sum(ending_prices)/len(option_simulations), max_option_price, initial_option_price, min_option_price

    # Returns the probability of exerise after simulation, takes set of option simulations
    def probability_of_exercise_calls(self, option_simulations,call=True):#call=True
        exercised = 0
        for option_simulation in option_simulations:
            exercised = exercised +  option_simulation.exercise_on_expiration(call)
        return exercised/len(option_simulations)
    
    def probability_of_exercise_puts(self, option_simulations,call=False):#call=False
        exercised = 0
        for option_simulation in option_simulations:
            exercised = exercised +  option_simulation.exercise_on_expiration_puts(call)
        return exercised/len(option_simulations)

    # Takes an option simulation set, chart each sample path and the respective variable
    def aggregate_chart_option_simulation(self, option_simulations, asset_prices, option_prices, option_deltas):
        # Sum the amount of variables we are plotting
        subplots = asset_prices + option_prices + option_deltas
        # Create subplots for each variable we are plotting
        fig, axs = plt.subplots(subplots,figsize=(15,10))
        fig.suptitle('Option Simulation Outcome')
        # If the variables is to be charted chart it on an independent axis
        if asset_prices:
            axs[0].set_title('Simulated Asset Prices')
            for o in option_simulations:
                axs[0].plot(o.asset_prices)
                # pick any option simulation and fetch the strike price (same for all simulations)
            axs[0].axhline(y=option_simulations[0].strike_price, color='r', linestyle='-', label='Strike Price')
            # To show strike price label
            axs[0].legend()
        if option_prices:
            axs[1].set_title('Option Prices Consequence of Asset Price Change')
            for o in option_simulations:
                axs[1].plot(o.option_prices)
        if option_deltas:
            axs[2].set_title('Option Deltas Consequence of Asset Price Change')
            for o in option_simulations:
                axs[2].plot(o.option_deltas)

        fig.subplots_adjust(hspace=.5)
        plt.show()


# Models the underling asset assuming geometetric brownian motion
class StochasticProcess:

    # Probability of motion in a certain direction
    def motion_probability(self, motion_to):
        if motion_to > self.current_asset_price:
            pass
        elif motion_to <= self.current_asset_price:
            pass
    def time_step(self):
        # Brownian motion is ~N(0,1)
        dW = np.random.normal()
        dS = self.drift*self.current_asset_price*self.delta_t + self.asset_volatility*self.current_asset_price*dW*math.sqrt(self.delta_t) 
        self.asset_prices.append(self.current_asset_price + dS)
        # Reassign the new current asset price for next time step
        self.current_asset_price = self.current_asset_price + dS

    def __init__(self, asset_price, drift, delta_t, asset_volatility):
        self.current_asset_price = asset_price
        self.asset_prices = []
        self.asset_prices.append(asset_price)
        self.drift = drift
        self.delta_t = delta_t
        self.asset_volatility = asset_volatility


class EuropeanCall:

    def call_delta(
        self, asset_price, asset_volatility, strike_price,
        time_to_expiration, risk_free_rate,q
            ):
        b = math.exp(-risk_free_rate*time_to_expiration)
        x1 = math.log(asset_price/(strike_price)) + .5*(asset_volatility*asset_volatility+risk_free_rate-q)*time_to_expiration
        x1 = x1/(asset_volatility*(time_to_expiration**.5))
        z1 = norm.cdf(x1)
        return z1

    def call_price(
        self, asset_price, asset_volatility, strike_price,
        time_to_expiration, risk_free_rate,q
            ):
        b  = math.exp(-risk_free_rate*time_to_expiration)
        x1 = math.log(asset_price/(strike_price))+(.5*(asset_volatility**2)+risk_free_rate-q)*time_to_expiration
        x1 = x1/(asset_volatility*(time_to_expiration**.5))
        z1 = norm.cdf(x1)
        z1 = z1*asset_price*math.exp(-q*time_to_expiration)
        x2 = math.log(asset_price/(strike_price)) - (.5*(asset_volatility**2)-risk_free_rate+q)*time_to_expiration
        x2 = x2/(asset_volatility*(time_to_expiration**.5))
        z2 = norm.cdf(x2)
        z2 = b*strike_price*z2
        return z1 - z2

    def __init__(
        self, asset_price, asset_volatility, strike_price,
        time_to_expiration, risk_free_rate,q
            ):
        self.asset_price = asset_price
        self.asset_volatility = asset_volatility
        self.strike_price = strike_price
        self.time_to_expiration = time_to_expiration
        self.risk_free_rate = risk_free_rate
        self.q = q
        self.price = self.call_price(asset_price, asset_volatility, strike_price, time_to_expiration, risk_free_rate,q)
        self.delta = self.call_delta(asset_price, asset_volatility, strike_price, time_to_expiration, risk_free_rate,q)
        


class EuropeanPut:

    def put_delta(
        self, asset_price, asset_volatility, strike_price,
        time_to_expiration, risk_free_rate,q
            ):
        b = math.exp(-risk_free_rate*time_to_expiration)
        x1 = math.log(asset_price/(strike_price)) + (.5*(asset_volatility**2)+risk_free_rate-q)*time_to_expiration
        x1 = x1/(asset_volatility*(time_to_expiration**.5))
        z1 = norm.cdf(x1)
        return z1 - 1

    def put_price(
        self, asset_price, asset_volatility, strike_price,
        time_to_expiration, risk_free_rate,q
            ):
        b  = math.exp(-risk_free_rate*time_to_expiration)
        x1 = math.log(asset_price/(strike_price))+(.5*(asset_volatility**2)+risk_free_rate-q)*time_to_expiration
        x1 = x1/(asset_volatility*(time_to_expiration**.5))
        z1 = norm.cdf(-x1)
        z1 = z1*asset_price*math.exp(-q*time_to_expiration)
        x2 = math.log(asset_price/(strike_price)) - (.5*(asset_volatility**2)-risk_free_rate+q)*time_to_expiration
        x2 = x2/(asset_volatility*(time_to_expiration**.5))
        z2 = norm.cdf(-x2)
        z2 = b*strike_price*z2
        return z2-z1

    def __init__(
        self, asset_price, asset_volatility, strike_price,
        time_to_expiration, risk_free_rate,q
            ):
        self.asset_price = asset_price
        self.asset_volatility = asset_volatility
        self.strike_price = strike_price
        self.time_to_expiration = time_to_expiration
        self.risk_free_rate = risk_free_rate
        self.q=q
        self.price = self.put_price(asset_price, asset_volatility, strike_price, time_to_expiration, risk_free_rate,q)
        self.delta = self.put_delta(asset_price, asset_volatility, strike_price, time_to_expiration, risk_free_rate,q)


class OptionSimulation:

    def exercise_on_expiration(self,call=True):#call=True
        # Call
        if call:
            if self.asset_prices[len(self.asset_prices)-1] > self.strike_price:
                return True
            else:
                return False
        # Put
        else:
            if self.asset_prices[len(self.asset_prices)-1] < self.strike_price:
                return True
            else:
                return False
            
    def exercise_on_expiration_puts(self,call=False):#call=False
        # Call
        if call:
            if self.asset_prices[len(self.asset_prices)-1] > self.strike_price:
                return True
            else:
                return False
        # Put
        else:
            if self.asset_prices[len(self.asset_prices)-1] < self.strike_price:
                return True
            else:
                return False

    def __init__(
        self, initial_asset_price, asset_volatility, strike_price,
        time_to_expiration, risk_free_rate
            ):
        self.initial_asset_price = initial_asset_price
        self.asset_volatility = asset_volatility
        self.strike_price = strike_price
        self.time_to_expiration = time_to_expiration
        self.risk_free_rate = risk_free_rate
        self.asset_prices = []
        self.option_prices = []
        self.option_deltas = []


# In[ ]:


#American option pricing using binomial trees:
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


# In[ ]:


#Stock Pressures and vizualisation algorithms:
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


# In[ ]:


class Grading_Bank_Average:
    
    def Final_Grade_Computation(self,lst):
        a=0
        Count=lst.count()
        for l in lst:
            if l=='Strong Buy':
                a=a+2
            elif l=='Outperform':
                a=a+2
            elif l=='Buy':
                a=a+1
            elif l=='Overweight':
                a=a+2
            elif l=='Sell':
                a=a-1
            elif l=='Hold':
                a=a-1
            elif l=='Underweight':
                a=a-2
            elif l=='Market Outperform':
                a=a+2
            elif l=='Positive':
                a=a+1
            else:
                a=a
                        
        grade=round(a/Count,2)
        #We use a degressig barema for the strong buy,making hard to have top notch grades          
        if grade<2 and grade>=1.5:
            a='Strong Buy Graded by Firms'
        elif grade<1.5 and grade>=0.5:
            a='Buy Graded by Firms'
        elif grade<0.5 and grade >=-0.5 :
            a='Hold Graded by Firms'
        elif grade>-1 and grade <=-0.5:
            a='Sell Graded by Firms'
        else:
            a='Strong Sell Graded by Firms'
        return a
        
    def __init__(self,lst):
        self.lst = lst
        self.Final_Grade_Given=self.Final_Grade_Computation(lst)
        


# In[ ]:


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


# In[ ]:


class Screener_EU:
    
    def European_Screening(self,_list_):
        
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
                print('Please retry')
                
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
                result_set=OptionTools().simulate_calls(100, 50, Strike_Price, last_sport_price, drift, 1/365, v, i_r, (90/365),d_y)
                c = OptionTools().probability_of_exercise_calls(result_set)
                print("Probability of exercice of the call: "+str(c))
                result_set=OptionTools().simulate_puts(100, 50, Strike_Price, last_sport_price, drift, 1/365, v, i_r, (90/365),d_y)
                p = OptionTools().probability_of_exercise_puts(result_set)
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
        self.EUScreen=self.European_Screening(_list_)
        


# In[ ]:


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
        return 
        
    
        
    def __init__(self,_list_):
        self.liste=_list_
        self.AMScreen=self.American_Screening(_list_)


# In[ ]:


TNX=wb.DataReader('^IRX',data_source='yahoo',start='2020-04-10')['Adj Close']


# In[ ]:


while True:
    today = date.today()
    user = input("Please enter a new command('help' for a summary of the commands): ")
    # guide him
    if user == "help":
        print("=" * 90 + "\nTo estimate an European option type 1:\n"
              + "=" * 90 +
              "\nTo estimate the expected future forecast of a volatility of a stock/commodity type 2:\n"
              + "=" * 90 +
              "\nTo use a screen option selector type 3:\n"
              + "=" * 90 +
              "\nTo estimate the price of an American Option type 4:\n"
              + "=" * 90 +
              "\nTo Visualize a stock/commodity movement type 5:\n"
              + "=" * 90 +
              "\nto leave the app, simply type 'exit'\n\n")
    elif user =="1":
        print("Use a monte-carlo simulator(type:'MC') or the Standard Estimation using Black and Sholes(type:'NMC')")
        a = input()
        while a!="MC" and a!="NMC":
            print("choose a correct combinaison(MC/NMC)")
            a = input()
        else:
            print("Estimate a Put(type P) or a Call(type C):")
            b=input()
            
            while b!="C" and b!="P":
                print("choose a correct combinaison(C/P)")
                b=input()
            
            print("Enter the ticker stock you choose :")
            s=input()
            Name=s
            
            try:
                s=wb.DataReader(s,data_source='yahoo',start='2019-01-01')['Adj Close']
            except:
                print('Retry and choose a Correct Ticker')
                break
                
                
            print('Do you want to use the dividends yield or not(Y/N)?')
            yn=input()
            while yn!='Y' and yn!='N':
                print('You can type Y or N')
                yn=input()
                
            if yn=='Y':
                try:
                    d_y=yf.Ticker(Name).info['dividendYield']
                    
                    if d_y is None:
                        d_y=0
                except:
                    print("This stock doesn't distribute dividends")
                    d_y=0
            else:
                d_y=0
    
            try:
                print("Enter your Strike Price(it's price today is {}),choose a rounded number:".format(round(s[-1],3)))
                SPR=input()
                
                while SPR.isnumeric()!=True:
                    print('Please insert only Numbers')
                    SPR=input()
                
                while SPR.isnumeric()==True and round(float(SPR),0)<0:
                    print('Select positive numbers')
                    SPR=input()
                    while SPR.isnumeric()!=True:
                        print('Please insert only Numbers')
                        SPR=input() 
                        
                SPR=round(float(SPR),0)
            
                print('Expiration Date:')
                print('input the year:(YYYY)')
                year=input()
                
                while year.isnumeric()!=True:
                    print('Please insert only Numbers')
                    year=input()
             
                while year.isnumeric()==True and int(year)<today.year:
                        print('Please insert a correct year,at least 2020 (YYYY)')
                        year=(input())
                        while year.isnumeric()!=True:
                            print('Please insert only valid Numbers')
                            year=input()
                            
                year=int(year)
                
                print('input the month:(MM)')
                month=input()
                
                while month.isnumeric()!=True:
                    print('Please insert only Numbers')
                    month=input()
             
                while month.isnumeric()==True:
                    if int(month)<today.month and year==today.year:
                        print('Please insert a correct month')
                        month=(input())
                        while month.isnumeric()!=True:
                            print('Please insert only valid Numbers')
                            month=input()
                            
                    elif int(month)>12 or int(month)<1 :
                        print('Please insert a correct month')
                        month=(input())
                        while month.isnumeric()!=True:
                            print('Please insert only valid Numbers')
                            month=input()
                            
                    else:
                        month=int(month)
                        break
                
                print('input the Day:(DD)')
                day=input()
                
                while day.isnumeric()!=True:
                    print('Please insert only Numbers')
                    day=input()
                    
                while day.isnumeric()==True:
                    if int(day)<today.day and year==today.year and month==today.month:
                        print('Please insert a correct day at least 1 day from now {}'.format(today.day+1))
                        day=(input())
                        while day.isnumeric()!=True:
                            print('Please insert only valid Numbers')
                            day=input()
                            
                    elif int(day)>30 or int(day)<1:
                        print('Please insert a correct day number')
                        day=(input())
                        while month.isnumeric()!=True:
                            print('Please insert only valid Numbers')
                            day=input()
                            
                    else:
                        day=int(day)
                        break
                        
                    
                m, d = OptionTools().compute_time_to_expiration((year),(month),(day))
                returns=s.pct_change().dropna()
                i_r=TNX[-1]/100
                v=GARCH_MODEL_AR_1(s,d).garch[-1]
                drift=returns.mean()*252-(1/2)*(v**2)
                
                if b=="P" and a=="NMC":
                    e_P=EuropeanPut(s[-1], v, SPR, m, i_r,d_y)
                    print("The Price of the Option assuming a volatility of {}% ,a strike price of {}$ and an interest rate of {}% is :".format(round(v*100,1),SPR,round(i_r*100,1)))
                    print('{}$'.format(round(e_P.price,2)))
                    
                elif b=="C" and a=="NMC":
                    e_C=EuropeanCall(s[-1],v, SPR, m, i_r,d_y)
                    print("The Price of the Option assuming a volatility of {}% ,a strike price of {}$ and an interest rate of {}% is :".format(round(v*100,1),SPR,round(i_r*100,1)))
                    print('{}$'.format(round(e_C.price,2)))
                    
                elif b=="C" and a=="MC":
                    result_set=OptionTools().simulate_calls(100, 1000, SPR, s[-1], drift, 1/365, v, i_r,m,d_y)
                    OptionTools().aggregate_chart_option_simulation(result_set, True, True, True)
                    h = OptionTools().simulation_analysis(result_set)
                    k = OptionTools().probability_of_exercise_calls(result_set)
                    print("Average Price of the option: "+str(round(h[0],3))+'$')
                    print("Maximum Price of the option: "+str(round(h[1],3))+'$')
                    print("Initial Price of the option: "+str(round(h[2],3))+'$')
                    print("Minimal Price of the option: "+str(round(h[3],3))+'$')
                    print("Probability of exercice: "+str(k))
                    
                elif b=="P" and a=="MC":
                    result_set=OptionTools().simulate_puts(100, 1000, SPR, s[-1], drift, 1/365, v, i_r,m,d_y)
                    OptionTools().aggregate_chart_option_simulation(result_set, True, True, True)
                    h = OptionTools().simulation_analysis(result_set)
                    k = OptionTools().probability_of_exercise_puts(result_set)
                    print("Average Price of the option: "+str(round(h[0],3))+'$')
                    print("Maximal Price of the option: "+str(round(h[1],3))+'$')
                    print("Initial Price of the option: "+str(round(h[2],3))+'$')
                    print("Minimal Price of the option: "+str(round(h[3],3))+'$')
                    print("Probability of exercice: "+str(k))
                    
            except:
                print('Error')
                
            try:
             
                RC = yf.Ticker(Name).recommendations['{}-01-01'.format(today.year):]
                M=RC.sort_values('Firm').drop_duplicates(['Firm'], keep='last')
                L=M['To Grade']
                FG=Grading_Bank_Average(L).Final_Grade_Given
                print(80*'=')
                print('\t\t'+str(FG))
                print(80*'=')
                print(M)
                
            except:
                print('No Grades available from Investment Banks')
                
    elif user=='2':
        print("Enter the ticker of the stock:")
        s=input()
        Name=s
        try:
            s=wb.DataReader(s,data_source='yahoo',start='2010-01-01')['Adj Close']
            
            print('What is your expiration date for the stock {} (You have to choose a day after {})'.format(Name,today.isoformat()))
            print('input the year:(YYYY)')
            
            year=input()     
                
            while year.isnumeric()!=True:
                print('Please insert only Numbers')
                year=input()
             
            while year.isnumeric()==True and int(year)<today.year:
                print('Please insert a correct year,at least 2020 (YYYY)')
                year=(input())
                while year.isnumeric()!=True:
                    print('Please insert only valid Numbers')
                    year=input()
                            
            year=int(year)
                
            print('input the month:(MM)')
            month=input()
                
            while month.isnumeric()!=True:
                print('Please insert only Numbers')
                month=input()
             
            while month.isnumeric()==True:
                if int(month)<today.month and year==today.year:
                    print('Please insert a correct month')
                    month=input()
                    while month.isnumeric()!=True:
                        print('Please insert only valid Numbers')
                        month=input()
                            
                elif int(month)>12 or int(month)<1 :
                    print('Please insert a correct month')
                    month=input()
                    while month.isnumeric()!=True:
                        print('Please insert only valid Numbers')
                        month=input()
                            
                else:
                    month=int(month)
                    break
                
            print('input the Day:(DD)')
            day=input()
                
            while day.isnumeric()!=True:
                print('Please insert only Numbers')
                day=input()
                    
            while day.isnumeric()==True:
                if int(day)<today.day and year==today.year and month==today.month:
                    print('Please insert a correct day at least 1 day from now {}'.format(today.day+1))
                    day=(input())
                    while day.isnumeric()!=True:
                        print('Please insert only valid Numbers')
                        day=input()
                            
                elif int(day)>30 or int(day)<1:
                    print('Please insert a correct day number')
                    day=(input())
                    while month.isnumeric()!=True:
                        print('Please insert only valid Numbers')
                        day=input()
                            
                else:
                    day=int(day)
                    break
                    
            m, d = OptionTools().compute_time_to_expiration((year),(month),(day))
            summary,fitted,for_vol,real_vol,last_for_vol=GARCH_MODEL_AR_1(s,d).garch
            print(summary)
        
            fig=plt.figure(figsize=(15,5))
            plt.plot(fitted,label='Train',lw=2,color='blue')
            plt.plot(for_vol,label='Forecasted value',lw=2,color='red')
            plt.title("Volatility forecast of the {}'s stock ".format(Name))
            plt.legend()
            plt.show()
            print('\t\t Conditional Volatility Forecast the {} is {}%'.format(for_vol.index[-1].strftime('%Y-%m-%d'),round(last_for_vol*100,2)))
        except:
            print('Error 404')
        
    elif user=='3':
        print('Welcome to the Screening menu for European options !')
        print('This screener is made for 3 months option')
        print("Buyer(B) or Seller(S)")
        print(80*'=')
        print('During the process it is not able to stop the machine.You will have to stop the kernel patch for that ')
        print('This screener is made for 3 months option')
        customer=input()
        
        while customer!='B' and customer!='S' :
            print('Please choose the right option')
            customer=input()
            
        print(80*'=')
        print('Choose the index analysis:Our analysis will be made on stocks composites of index you want,type:(DOW has the smallest index with 30 components)') 
        print('SP500  : for S&P 500 composites')
        print('DOW    : for DOW JONES composites')
        print('NASDAQ : for NASDAQ composites')
        print(80*'=')
        print('What is your choice?')
        stocklist=input('NASDAQ/DOW/SP500 : ')
        while stocklist!='NASDAQ' and stocklist!='DOW' and stocklist!='SP500' :
            print('Please choose the right option')
            stocklist=input()
            
        if stocklist=='NASDAQ':
            stocklist=si.tickers_nasdaq()
        elif stocklist=='DOW':
            stocklist=si.tickers_dow()
        else:
            stocklist=si.tickers_sp500()
            
        start_date=date.today()-timedelta(days=360)
        end_date=date.today()
        
        print('Do you want to use the dividend yields or not(Y/N)?')
        yn=input()
        while yn!='Y' and yn!='N':
            print('You can type Y or N')
            yn=input()
        
        print('Do you want to use the pricing for American Options or European ones?(type A or E)')
        opt=input('You can type A or E : ')
        while opt!='A' and opt!='E':
            print('You can type A or E')
            opt=input()
            
        if opt=='E':
            Screener_EU(stocklist).EUScreen
        else:
            Screener_US(stocklist).AMScreen
    
    elif user=='4':
            print('Do you want to price a Call(type C) or a Put(type P) option?')
            am_c_p=input()
            while am_c_p!='P' and am_c_p!='C':
                print('Wrong command.Please retry')
                am_c_p=input()

            print('insert the ticker of the Stock')
            stk=input()
            Name=stk
            try:
                s=wb.DataReader(stk,data_source='yahoo',start='2010-01-01')['Adj Close']
            except:
                print("You must select a valid ticker")
                break
                
            print('Do you want to use the dividend yields or not(Y/N)?')
            yn=input()
            while yn!='Y' and yn!='N':
                print('You can type Y or N')
                yn=input()
            if yn=='Y' :
                try:
                    d_y=yf.Ticker(Name).info['dividendYield']
                    if d_y is None:
                        d_y=0
                
                except:
                    d_y=0
            else:
                d_y=0
                
            try:
                print("Enter your Strike Price(it's price today is {}):".format(round(s[-1],3)))
                
                SPR=input()
                
                while SPR.isnumeric()!=True:
                    print('Please insert only Numbers')
                    SPR=input()
                
                while SPR.isnumeric()==True and round(float(SPR),0)<0:
                    print('Select positive numbers')
                    SPR=input()
                    while SPR.isnumeric()!=True:
                        print('Please insert only Numbers')
                        SPR=input() 
                        
                SPR=round(float(SPR),0)
                
                print('Expiration Date:')
                print('input the year:(YYYY)')
                year=input()
                
                while year.isnumeric()!=True:
                    print('Please insert only Numbers')
                    year=input()
             
                while year.isnumeric()==True and int(year)<today.year:
                        print('Please insert a correct year,at least 2020 (YYYY)')
                        year=(input())
                        while year.isnumeric()!=True:
                            print('Please insert only valid Numbers')
                            year=input()
                            
                year=int(year)
                
                print('input the month:(MM)')
                month=input()
                
                while month.isnumeric()!=True:
                    print('Please insert only Numbers')
                    month=input()
             
                while month.isnumeric()==True:
                    if int(month)<today.month and year==today.year:
                        print('Please insert a correct month')
                        month=(input())
                        while month.isnumeric()!=True:
                            print('Please insert only valid Numbers')
                            month=input()
                            
                    elif int(month)>12 or int(month)<1 :
                        print('Please insert a correct month')
                        month=(input())
                        while month.isnumeric()!=True:
                            print('Please insert only valid Numbers')
                            month=input()
                            
                    else:
                        month=int(month)
                        break
                
                print('input the Day:(DD)')
                day=input()
                
                while day.isnumeric()!=True:
                    print('Please insert only Numbers')
                    day=input()
                    
                while day.isnumeric()==True:
                    if int(day)<today.day and year==today.year and month==today.month:
                        print('Please insert a correct day at least 1 day from now {}'.format(today.day+1))
                        day=(input())
                        while day.isnumeric()!=True:
                            print('Please insert only valid Numbers')
                            day=input()
                            
                    elif int(day)>30 or int(day)<1:
                        print('Please insert a correct day number')
                        day=(input())
                        while month.isnumeric()!=True:
                            print('Please insert only valid Numbers')
                            day=input()
                            
                    else:
                        day=int(day)
                        break
                
            except:
                print('Retry and use exclusively Numbers')
            
            try:
                m, d = OptionTools().compute_time_to_expiration((year),(month),(day))
                v=GARCH_MODEL_AR_1(s,d).garch[-1]
                American_O_Pricing=American_Option_Pricing(N=int(round(d/30,0)),q=d_y,T=m,S0=s[-1],sigma=v,r=TNX[-1]/100,K=SPR)
                if am_c_p=='P':
                    b_put=American_O_Pricing.binomial_tree_Pricing_put_d
                    print('Price of the American option = '+str(round(b_put[0],2)))
                    print(80*'=')
                    print('\t\t Binomial Tree for stock prices :')
                    print(pd.DataFrame(b_put[1]))
                    print(80*'=')
                    print('\t\t Binomial Tree for the option :')
                    print(pd.DataFrame(b_put[2]))
                    print(80*'=')
                    print('\t\t Probability of execution the last month :'+str(b_put[-1]))
                      
                elif am_c_p=='C':
                    b_call=American_O_Pricing.binomial_tree_Pricing_call_d
                    print('Price of the American option = '+str(round(b_call[0],2)))
                    print(80*'=')
                    print('\t\t\t Binomial Tree for stock prices :')
                    print(pd.DataFrame(b_call[1]))
                    print(80*'=')
                    print('\t\t\t Binomial Tree for the option :')
                    print(pd.DataFrame(b_call[2]))
                    print(80*'=')
                    print('\t\t Probability of execution the last month :'+str(b_call[-1]))
            except:
                print('Error')
                    
            try:
                RC = yf.Ticker(Name).recommendations['{}-01-01'.format(today.year):]
                M=RC.sort_values('Firm').drop_duplicates(['Firm'], keep='last')
                L=M['To Grade']
                FG=Grading_Bank_Average(L).Final_Grade_Given
                print(80*'=')
                print('\t\t\t'+str(FG))
                print(80*'=')
                print(M)
            except:
                print('No Grades available from Investment Banks')
        
    elif user=='5':
        
        print("What stock do you want to visualize (type it's ticker) ?")
        stk=input()
        try:
            print('From what date do you want to visualize the stock {} (You have to choose an anterior date to 2019-01-01)?'.format(stk))
            print('input the year:(YYYY)')
            year=input()
            
            while year.isnumeric()!=True:
                    print('Please insert only Numbers(YYYY)')
                    year=input()
             
            while year.isnumeric()==True :
                if int(year)>=2019 or int(year)<1980:
                    print('Please insert a correct year between 1980 and 2018')
                    year=input()
                    while year.isnumeric()!=True:
                        print('Please insert only valid Numbers(YYYY)')
                        year=input()
                else:
                    break
            
            year=int(year)
                
            print('input the month:(MM)')
            month=input()
            
            while month.isnumeric()!=True:
                    print('Please insert only Numbers(MM)')
                    month=input()
             
            while month.isnumeric()==True :
                if int(month)>12 or int(month)<1:
                    print('Please insert a correct month')
                    month=input()
                    while month.isnumeric()!=True:
                        print('Please insert only valid Numbers(MM)')
                        month=input()
                else:
                    break
                    
            month=int(month)
                
            print('input the Day:(DD)')
            day=input()
            
            while day.isnumeric()!=True:
                    print('Please insert only Numbers(DD)')
                    day=input()
             
            while day.isnumeric()==True:
                if int(day)>30 or int(day)<0:#We supresss 31 by precautions
                    print('Please insert a correct day number')
                    day=input()
                    while month.isnumeric()!=True:
                        print('Please insert only valid Numbers(DD)')
                        day=input()
                else:
                    break
                    
            day=int(day)
                
            start_date=date(year,month,day).isoformat()
            stock_analyzed=wb.DataReader(stk,data_source='yahoo',start=start_date)
            TCH=technical_chart(stock_analyzed)
            
            signal_Buy=TCH.below_pressure
            signal_Sell=TCH.above_pressure
    

            apdict = [mpf.make_addplot(TCH.analysis[['LB','HB']]),
                  mpf.make_addplot(signal_Buy,scatter=True,markersize=200,marker='^'),
                  mpf.make_addplot(signal_Sell,scatter=True,markersize=200,marker='v')]
        
            mpf.plot(stock_analyzed,type='candle',volume=True,style='charles',addplot=apdict,figscale=2)
            print('\t\t\t Low Bollinger Band : Blue')
            print('\t\t\t High Bollinger Band : Orange')
            
            print('Do you want to dowload available Calls and Puts in the Market?')
            CP=input('Y/N :')
            while CP!='Y'and CP!='N':
                print('Please input the right command(Y/N)')
                CP=input('Y/N :')
            if CP=='Y':
                Calls=get_calls(stk).sort_values('Open Interest')
                Puts=get_puts(stk).sort_values('Open Interest')
                Calls.to_excel('Calls{}.xlsx'.format(stk))
                Puts.to_excel('Puts{}.xlsx'.format(stk)) 
                print('Thank you,files are registred under the names:')
                print('Calls{}.xlsx'.format(stk))
                print('Puts{}.xlsx'.format(stk))
            

        except:
            print('Error 404')
        
    elif user == "exit":
        print("\t\t Sucess you are out")
        break
            
    else:
        print('You typed on a wrong command.Please retry')           
        


# In[ ]:




