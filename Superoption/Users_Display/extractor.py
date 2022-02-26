from Rate_extractor.rate_extractor import rate_ext, yahoo_finance_dividend
from Users_Display.inputs import Choose_ticker


def data_extractor(include_dividend = True):
    Ticker = Choose_ticker()
    data = rate_ext(Ticker)

    while data.size == 0:
        print('Retry and choose a Correct Ticker \n' + 80*'=')
        Ticker = Choose_ticker()
        data = rate_ext(Ticker)

    if include_dividend == True:
        dividend = yahoo_finance_dividend(Ticker)
    else:
        dividend = 0 

    return data, dividend


