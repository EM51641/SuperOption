from Users_Display.inputs import What_option, How_to_evaluate_option
from Users_Display.extractor import extract_dividend, data_extractor
from Users_Display.inputs import date_input, strike_input
from Tools.Option_tools import OptionTools
from Mc_generator.generator import data_prep 



def User_1(interest_rate):

    Condition1 = How_to_evaluate_option()

    Condition2 = What_option()

    data, dividend = data_extractor()

    strike = strike_input(round(data[-1],3))

    year, month, day = date_input(history = False)

    ttm, horizon = OptionTools().compute_time_to_expiration(year, month, day)

    # volatility_process = add inputs 

    #Option_price = Choice(Condition1, volatility_process, data, horizon)

    



    #Add volatility choice


    pass



def Choice(choice, volatility_process, data, horizon):
    if choice == 'NMC':
        return Closed_fml(volatility_process, data, horizon)
    else : 
        return Monte_carlos(volatility_process, data, horizon)



def Monte_carlos(volatility_process, data, horizon):

    if volatility_process == 'Heston Model':
        volatility = data_prep(data, horizon)[-1]
    
    elif volatility_process == 'Rough':
        volatility = 
    else :  
        volatility =
    return

    
def Closed_fml(volatility_process, data, horizon):
    if volatility_process == 'Garch':
        volatility = data_prep(data, horizon)[-1]

    elif volatility_process == 'Heston Model':
        volatility = data_prep(data, horizon)[-1]
    
    else :  
        volatility =
    return