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
