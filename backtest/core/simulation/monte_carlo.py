import numpy as np
import pandas as pd
from numba import njit, prange

from numba import njit
import numpy as np

@njit
def simulate_equity_curves_with_block_bootstrap(
    returns, 
    initial_equity, 
    num_simulations, 
    block_size
):
    """
    Simulates multiple equity curves by block bootstrapping historical returns.

    Parameters:
    ----------
    returns : numpy.ndarray
        An array of historical returns.

    initial_equity : float
        The initial equity value for the simulated curves.
        
    num_simulations : int
        The number of simulated equity curves to generate.

    block_size : int
        The size of each contiguous block of returns the returns array will be partitioned
        into for bootstrapping.

    Returns:
    -------
    numpy.ndarray
        A 2D array where each column represents a simulated equity curve and each row represents 
        a time step in the simulation.
    """
    num_days = len(returns)
    num_blocks = num_days // block_size
    simulated_curves = np.empty((num_days, num_simulations))

    for sim in range(num_simulations):
        # Initialize equity curve for this simulation
        equity_curve = np.empty(num_days)
        equity_curve[0] = initial_equity

        # Perform block bootstrapping
        for block_start in range(0, num_days, block_size):
            block_end = min(block_start + block_size, num_days)
            sampled_block_start = np.random.randint(0, num_days - block_size + 1)
            sampled_block_end = sampled_block_start + block_end - block_start

            sampled_returns = returns[sampled_block_start:sampled_block_end]

            # Compute the equity values for this block
            for i in range(block_start, block_end):
                if i == 0:
                    equity_curve[i] = initial_equity * (1 + sampled_returns[i - block_start])
                else:
                    equity_curve[i] = equity_curve[i - 1] * (1 + sampled_returns[i - block_start])

        simulated_curves[:, sim] = equity_curve

    return simulated_curves

def run_monte_carlo_simulation(equity_curve, num_simulations = 1000):
    """
    Runs a Monte Carlo simulation on an equity curve.

    Given an equity curve, this function calculates returns at each timestep and then samples blocks of returns
    to generate multiple simulated equity curves.  This is useful for estimating the distribution of possible
    outcomes for the equity curve over time.

    Parameters:
    ----------
    equity_curve : pandas.DataFrame
        A DataFrame with a column 'equity' representing the equity curve to simulate. The index should be a datetime index.
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

    # Calculate returns from the equity curve
    equity_curve['returns'] = equity_curve['equity'].pct_change().fillna(0)

    # Extract the initial equity value for the simulation
    initial_equity = equity_curve['equity'].iloc[0]

    # Generate simulated equity curves
    simulated_curves_np = simulate_equity_curves_with_block_bootstrap(
        returns = equity_curve['returns'].values, 
        initial_equity = initial_equity, 
        num_simulations = num_simulations,
        block_size = 24
    )

    # Convert the numpy array of simulated curves to a pandas DataFrame
    dates = equity_curve.index
    simulated_curves_df = pd.DataFrame(simulated_curves_np, index=dates)

    return simulated_curves_df
