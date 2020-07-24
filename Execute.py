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

TNX=wb.DataReader('^IRX',data_source='yahoo',start='2020-04-10')['Adj Close']

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
