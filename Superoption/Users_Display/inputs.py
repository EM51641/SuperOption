from genericpath import exists

from numpy import float16
from Parsers.date_parser import date_parse, day_check, month_check, year_check
from Rate_extractor.rate_extractor import rate_ext
from datetime import date, datetime 
import re  
import string


def date_input(history = False):
    print('Expiration date (DD/MM/YYYY): \n' + 80*'=')
    date = str(input())
    while date_definition_option(date, history) == False:
        date = str(input())
    return date_parse(date)

def date_definition_option(date, history):

    if set(re.sub('[/]', '', date)).issubset(set(string.digits)) is False:
        print('Please insert appropriate digits: \n' + 80*'=')
        return False
     #   date = str(input())

    day, month, year = date_parse(date)

    if day_check(day, month, year) == False or month_check(month) == False or year_check(year) == False :
        print('2')
        print('Please insert an appropriate date: \n' + 80*'=')
        return False
     #   day, month, year = date_parse(date)

    if history == True :
         while datetime(year, month, day) < date.today():
             print('Please insert a future date: \n' + 80*'=')
             return False

    day, month, year = date_parse(date)

    return day, month, year


def strike_input(last_price):
    print("Enter your Strike Price (It's price today is {})\
        , choose a rounded number:\n".format(last_price) + 80*'=')
    strike = input()
    while positive_digit_verifier(strike, False) != True:
        strike = input()
    return float(strike)


def digit_verifier(inpt, positive):

    if set(inpt).issubset(set(string.digits + '.')) is False:
        print('Please insert a float or an integer number: \n' + 80*'=')
        return False

    if inpt.count('.') > 1:
        print('Please insert only one . :\n' + 80*'=')

    if positive :
        if float(inpt) < 0:
            print('Please insert a positive number: \n' + 80*'=')
            return False

    return True 


def dividend():
    print('Do you want to use the dividends yield or not (yes/no)?\n' + 80*'=')
    Answer = str(input())
    while dividend_verifier(Answer) == False:
        Answer = str(input())
    return Answer


def dividend_verifier(Answer):
    if set(Answer).issubset(set(string.ascii_lowercase)) is False:
        print('Only use lowercase characters\n' + 80*'=')
        return False
    
    if Answer != 'yes' and Answer != 'no':
        print('Please only use yes/no \n'+ 80*'=')
        return False

    return Answer


def How_to_evaluate_option():
    print("Use a Monte-carlo (MC) simulation or a semi-closed formula (NMC)\
         to simulate your option price?")
    Answer = str(input())
    while How_to_evaluate_option_verifier(Answer) == False:
        Answer = str(input())
    return Answer

def How_to_evaluate_option_verifier(Answer):
    if set(Answer).issubset(set(string.ascii_uppercase)) is False:
        print('Only use uppercase characters\n' + 80*'=')
        return False
    
    if Answer != 'MC' and Answer != 'NMC':
        print('Please only use MC/NMC only \n'+ 80*'=')
        return False

    return Answer

def What_option():
    print('Put(type P) or a Call(type C)')
    Answer = str(input())
    while What_option_verifier(Answer) == False:
        Answer = str(input())
    return Answer

def What_option_verifier(Answer):
    if set(Answer).issubset(set(string.ascii_uppercase)) is False:
        print('Only use uppercase characters\n' + 80*'=')
        return False
    
    if Answer != 'C' and Answer != 'P':
        print('Please only use C/P only \n'+ 80*'=')
        return False

    return Answer

def Choose_ticker():
    print("Enter the ticker stock you choose :\n" + 80*'=')
    Ticker = str(input())
    return Ticker

def Choose_volatility():
    print("Enter your implied volatility \n".format(0) + 80*'=')
    implied_vol = input()
    while digit_verifier(implied_vol, True) == False:
        implied_vol = input()
    return float(implied_vol)

#Heston
def input_initial_vol():
    print("Enter your initial volatility :\n".format(0) + 80*'=')
    inital_vol = input()
    while digit_verifier(inital_vol, True) == False:
        inital_vol = input()
    return float(inital_vol)

def input_kappa():
    print("Enter your kappa parameter :\n".format(0) + 80*'=')
    kappa = input()
    while digit_verifier(kappa, True) == False:
        kappa = input()
    return float(kappa)

def input_theta():
    print("Enter your theta parameter :\n".format(0) + 80*'=')
    theta = input()
    while digit_verifier(theta, True) == False:
        theta = input()
    return float(theta)

#Heston+Jumps
def add_lambda_():
    print("Enter your lambda parameter :\n".format(0) + 80*'=')
    lambda_ = input()
    while digit_verifier(lambda_, True) == False:
        lambda_ = input()
    return float(lambda_)

def add_mean_jump_size():
    print("Enter your mean jump size parameter :\n".format(0) + 80*'=')
    mean = input()
    while digit_verifier(mean, False) == False:
        mean = input()
    return float(mean)

def jump_volatility():
    print("Enter your jump volatility parameter :\n".format(0) + 80*'=')
    sigma = input()
    while digit_verifier(sigma, True) == False:
        sigma = input()
    return float(sigma)













     
