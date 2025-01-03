import numpy as np
import pandas as pd
from numba import njit

@njit
def custom_diff(arr):
    return np.array([(arr[i] - arr[i - 1]) / arr[i - 1] for i in range(1, len(arr))])

@njit
def custom_rolling_max(arr):
    result = np.empty(arr.shape)
    max_value = arr[0]
    for i in range(len(arr)):
        if arr[i] > max_value:
            max_value = arr[i]
        result[i] = max_value
    return result

@njit
def simulate_equity_curves_with_block_bootstrap(
    returns, 
    initial_equity, 
    n, 
    block_size
):
    """
    Simulates multiple equity curves by block bootstrapping historical returns.

    Parameters:
    ----------
    returns : numpy.ndarray
        An array of historical returns.

    initial_equity : float
        The initial value of equity at the start of the simulation.
        
    num_simulations : int
        The number of simulated equity curves to generate.

    block_size : int
        The size of each block to sample. 

    Returns:
    -------
    numpy.ndarray
        A 2D array where each column represents a simulated equity curve and each row corresponds 
        to a time point in the simulation.
    """
    num_days = len(returns)
    num_blocks = num_days // block_size
    simulated_curves = np.empty((num_days, n))

    for sim in range(n):
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

@njit(nopython=True)
def value_at_risk(equity_curve, holding_period, confidence_level):
    # Calculate daily returns
    returns = custom_diff(equity_curve) / equity_curve[:-1]

    # Calculate the VaR percentage using np.percentile
    var_percent = -np.percentile(returns, (1 - confidence_level) * 100)

    # Scale the VaR to the holding period
    var_scaled_percent = var_percent * np.sqrt(holding_period)

    return abs(var_scaled_percent)

@njit(nopython=True)
def conditional_value_at_risk(equity_curve, holding_period, confidence_level):
    # Calculate daily returns
    returns = custom_diff(equity_curve) / equity_curve[:-1]

    # Calculate the VaR percentage using np.percentile
    var_percent = -np.percentile(returns, (1 - confidence_level) * 100)

    # Identify the returns that are worse than the VaR
    worse_than_var_returns = returns[returns <= -var_percent]

    # Calculate the mean of these returns using np.mean
    mean_worse_than_var_returns = np.mean(worse_than_var_returns)

    # Scale the CVaR to the holding period
    cvar_scaled_percent = mean_worse_than_var_returns * np.sqrt(holding_period)

    return abs(cvar_scaled_percent)

@njit
def max_drawdown(equity):
    rolling_max = custom_rolling_max(equity)
    drawdown = equity / rolling_max - 1.0
    max_dd_pct = drawdown.min() * 100
    return max_dd_pct
    
@njit
def avg_drawdown(equity):
    rolling_max = custom_rolling_max(equity)
    drawdowns = equity / rolling_max - 1.0

    # Compute average drawdown
    # Exclude the periods with no drawdown (drawdowns >= 0)
    negative_drawdowns = drawdowns[drawdowns < 0]
    if len(negative_drawdowns) > 0:
        avg_dd_pct = negative_drawdowns.mean() * 100
    else:
        avg_dd_pct = 0.0

    return avg_dd_pct

@njit
def sharpe_ratio(equity):
    # Sharpe Ratio of non-zero returns
    returns = custom_diff(equity)
    mean_returns = np.mean(returns)
    std_returns = np.std(returns)

    if std_returns != 0:
        sharpe = np.sqrt(365) * mean_returns / std_returns
        return sharpe
    else:
        return np.nan

@njit
def sortino_ratio(equity):
    # Sortino Ratio of non-zero returns
    returns = custom_diff(equity)
    mean_returns = np.mean(returns)
    negative_returns = returns[returns < 0]
    std_negative_returns = np.std(negative_returns)

    if std_negative_returns != 0 and len(negative_returns) > 0:
        try:
            sortino = np.sqrt(365) * mean_returns / (std_negative_returns)
            return sortino
        except:
            return np.nan
    else:
        return np.nan

@njit
def calmar_ratio(equity):
    num_years = len(equity) / (365)
    returns = custom_diff(equity)
    cum_ret_final = np.prod(1 + returns)

    if num_years != 0:
        annual_returns = cum_ret_final ** (1 / num_years) - 1
    else:
        return np.nan

    try:
        max_dd = max_drawdown(equity)  # Assuming max_drawdown is also Numba compliant
        if max_dd != 0:
            calmar = annual_returns / abs(max_dd / 100)
            return calmar
        else:
            return np.nan
    except:
        return np.nan

def run_monte_carlo_simulation(equity_curve, num_simulations = 100_000):
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
    simulated_curves_np = simulate_equity_curves_with_block_bootstrap(equity_curve['returns'].values, initial_equity, num_simulations, 7)

    # Convert the numpy array of simulated curves to a pandas DataFrame
    dates = equity_curve.index
    simulated_curves_df = pd.DataFrame(simulated_curves_np, index=dates)

    return simulated_curves_df

def calculate_monte_carlo_performance_metrics(monte_carlo_equity_curves):
    """
    Calculates performance metrics for each Monte Carlo simulated equity curve.

    Parameters:
    ----------
    monte_carlo_equity_curves : pandas.DataFrame
        A DataFrame where each column represents a simulated equity curve.

    Returns:
    -------
    metrics_dict : dict
        A dictionary where keys are metric names and values are lists of metric values for each simulated equity curve.
    """
    
    # Define holding periods and confidence level
    holding_periods = [1, 7, 30]  # 1 day, 1 week, 1 month in days
    confidence_level = 0.99

    # Initialize dictionary to store metrics
    metrics_dict = {'1_day_var': [], '1_week_var': [], '1_month_var': [],
                    '1_day_cvar': [], '1_week_cvar': [], '1_month_cvar': [],
                    'sharpe_ratio': [], 'sortino_ratio': [], 'calmar_ratio': [],
                    'max_dd': [], 'avg_dd': []}

    # Iterate over each simulated equity curve
    i = 1
    for curve in monte_carlo_equity_curves.columns:

        # print('\r {} / {}'.format(i, len(monte_carlo_equity_curves.columns)), end = '', flush = True)

        i += 1

        equity_curve = monte_carlo_equity_curves[curve].values
        
        sharpe = sharpe_ratio(equity_curve)
        try:
            sortino = sortino_ratio(equity_curve)
        except:
            sortino = float('-inf')
            
        calmar = calmar_ratio(equity_curve)
        max_dd = max_drawdown(equity_curve)
        avg_dd = avg_drawdown(equity_curve)

        metrics_dict['sharpe_ratio'].append(sharpe)
        metrics_dict['sortino_ratio'].append(sortino)
        metrics_dict['calmar_ratio'].append(calmar)
        metrics_dict['max_dd'].append(max_dd)
        metrics_dict['avg_dd'].append(avg_dd)

        # Calculate VaR and CVaR for each holding period
        for holding_period in holding_periods:
            var = value_at_risk(equity_curve, holding_period, confidence_level)
            cvar = conditional_value_at_risk(equity_curve, holding_period, confidence_level)
            
            # Assign to appropriate keys in the dictionary
            if holding_period == 1:  # 1 day
                metrics_dict['1_day_var'].append(var)
                metrics_dict['1_day_cvar'].append(cvar)
            elif holding_period == 7:  # 1 week
                metrics_dict['1_week_var'].append(var)
                metrics_dict['1_week_cvar'].append(cvar)
            elif holding_period == 30:  # 1 month
                metrics_dict['1_month_var'].append(var)
                metrics_dict['1_month_cvar'].append(cvar)

    return metrics_dict