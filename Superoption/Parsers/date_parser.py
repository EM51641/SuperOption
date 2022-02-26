
import re 

def date_parse(strg):
    string_list = re.split('/', strg)
    day, month, year = list( map( int, string_list ) )
    return day, month, year

def day_check(day, month, year): 
    if month in (1, 3, 5, 7, 8, 10, 12):
        if day > 31 and day <1 : 
            return False 
    elif month in (4, 6, 9, 11):
        if day > 30 and day <1 : 
            return False 
    elif year %400 == 0 :
        if day > 29: 
            return False 
    else:
        if day > 28:
            return False  
    return True 

def month_check(month):
    if month < 0 or month > 12:
        return False
    return True

def year_check(year):
    if year < 1980:
        return False 
    return True 