from Users_Display.inputs import What_option, How_to_evaluate_option
from Users_Display.extractor import extract_dividend, data_extractor
from Users_Display.inputs import date_input, strike_input
from Tools.Option_tools import OptionTools


def User_1(interest_rate):

    Condition1 = How_to_evaluate_option()

    Condition2 = What_option()

    data, dividend = data_extractor()

    strike = strike_input(round(data[-1],3))

    year, month, day = date_input(history = False)

    ttm, horizon = OptionTools().compute_time_to_expiration(year, month, day)

    #Add volatility choice


    pass






    
