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
