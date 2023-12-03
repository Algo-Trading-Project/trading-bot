import numpy as np
import pandas as pd
from numba import njit, prange

@njit
def simulate_equity_curves(returns, initial_equity, num_simulations):
    """
    Simulates multiple equity curves by randomly sampling historical returns with replacement.

    Parameters:
    ----------
    returns : numpy.ndarray
        An array of historical returns.

    initial_equity : float
        The initial value of equity at the start of the simulation.
        
    num_simulations : int
        The number of simulated equity curves to generate.

    Returns:
    -------
    numpy.ndarray
        A 2D array where each column represents a simulated equity curve and each row corresponds 
        to a time point in the simulation.
    """
    num_days = len(returns)
    simulated_curves = np.empty((num_days, num_simulations))

    for sim in range(num_simulations):
        # Randomly select returns with replacement
        sampled_indices = np.random.randint(0, num_days, num_days)
        sampled_returns = returns[sampled_indices]

        # Generate an equity curve for this simulation
        equity_curve = np.empty(num_days)
        equity_curve[0] = initial_equity * (1 + sampled_returns[0])
        for i in range(1, num_days):
            equity_curve[i] = equity_curve[i - 1] * (1 + sampled_returns[i])

        simulated_curves[:, sim] = equity_curve

    return simulated_curves

def run_monte_carlo_simulation(equity_curve, num_simulations = 1000):
    """
    Runs a Monte Carlo simulation on an equity curve.

    Given an equity curve, this function calculates daily returns and then uses these returns
    to simulate multiple possible future equity curve trajectories.

    Parameters:
    ----------
    equity_curve : pandas.DataFrame
        A DataFrame with a column 'equity' representing the value of an equity over time.
    num_simulations : int, default 1000
        The number of simulated equity curves to generate.

    Returns:
    -------
    pandas.DataFrame
        A DataFrame where each column is a simulated equity curve, indexed by the same dates as the input equity_curve.

    Notes:
    -----
    - The initial equity value for simulations is taken from the first value of the 'equity' column in the input DataFrame.
    """

    # Calculate daily returns from the equity curve
    equity_curve['returns'] = equity_curve['equity'].pct_change().fillna(0)

    # Extract the initial equity value for the simulation
    initial_equity = equity_curve['equity'].iloc[0]

    # Generate simulated equity curves
    simulated_curves_np = simulate_equity_curves(equity_curve['returns'].values, initial_equity, num_simulations)

    # Convert the numpy array of simulated curves to a pandas DataFrame
    dates = equity_curve.index
    simulated_curves_df = pd.DataFrame(simulated_curves_np, index=dates)

    c = simulated_curves_df.columns

    return simulated_curves_df
