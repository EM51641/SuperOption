
from pandas_datareader import data as wb
from yahoo_fin.stock_info import *
from yahoo_fin.options import *
import yfinance as yf
from Grading_Function.Grading_Bank_System import Grading_Bank_Average

def rate_ext(ticker, start_date = '2020-01-01'):
    rate = get_data(ticker,data_source='yahoo',start = start_date)['Adj Close']
    return rate

def yahoo_finance_dividend(ticker, start_date = '2020-01-01'):
    dividend = get_dividends(ticker,start_date = '2020-01-01')['dividend']
    return dividend

def invetment_grades(ticker, date):
    recommendations = yf.Ticker(ticker).recommendations['{}-01-01'.format(date):]
    recommendations = recommendations.sort_values('Firm').drop_duplicates(['Firm'], keep='last')['To Grade']
    grade = Grading_Bank_Average(recommendations).Final_Grade_Given
    return grade

def stocklister(ticker = 'SP500'):
    if ticker=='NASDAQ':
        stocklist= tickers_nasdaq()
    elif ticker=='DOW':
        stocklist= tickers_dow()
    else:
        stocklist= tickers_sp500()

        return stocklist

def Option_writer(ticker):
    Calls = get_calls(ticker).sort_values('Open Interest')
    Puts = get_puts(ticker).sort_values('Open Interest')
    Calls.to_excel('Calls{}.xlsx'.format(ticker))
    Puts.to_excel('Puts{}.xlsx'.format(ticker)) 
    print('Thank you,files are registred under the names:')
    print('Calls{}.xlsx'.format(ticker))
    print('Puts{}.xlsx'.format(ticker))



