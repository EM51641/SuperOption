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
