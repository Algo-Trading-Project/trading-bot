from numba import njit, prange
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from numba import njit

# Numba implementation of the permutation test
@njit(parallel = True)
def permutation_test(y_pred, trade_returns, n_simulations = 1000):
    results = np.zeros(n_simulations)

    for i in prange(n_simulations):
        # Shuffle the predictions (not in place)
        y_pred_permuted = y_pred.copy()
        np.random.shuffle(y_pred_permuted)

        # Calculate the sample statistic
        sample_means = np.zeros(2)
        sample_means[0] = np.mean(trade_returns[y_pred_permuted == 0])
        sample_means[1] = np.mean(trade_returns[y_pred_permuted == 1])

        results[i] = sample_means[1] - sample_means[0]

    return results

# Numba implementation of the bootstrap distribution
@njit(parallel = True)
def calculate_bootstrap_distribution_numba(X_test, n_simulations = 1000):
    diff_distrubtion = np.zeros(n_simulations)
    pos_pred_mean_returns = np.zeros(n_simulations)
    neg_pred_mean_returns = np.zeros(n_simulations)

    for i in prange(n_simulations):
        # Sample with replacement
        X_test_sample = X_test[np.random.randint(0, X_test.shape[0], size = X_test.shape[0]), :]

        # Calculate the mean returns for positive and negative predictions
        mean_returns_pos = np.mean(X_test_sample[X_test_sample[:, 0] == 1, 1])
        mean_returns_neg = np.mean(X_test_sample[X_test_sample[:, 0] == 0, 1])

        pos_pred_mean_returns[i] = mean_returns_pos
        neg_pred_mean_returns[i] = mean_returns_neg

        # Calculate the sample statistic
        diff_distrubtion[i] = mean_returns_pos - mean_returns_neg

    return diff_distrubtion, pos_pred_mean_returns, neg_pred_mean_returns

# Function to run the OOS statistical tests
def run_oos_statistical_tests(y_pred, trade_returns, n_simulations, sample_statistic):
    # Calculate the permutation test
    permutation_results = permutation_test(y_pred, trade_returns, n_simulations = n_simulations)
    empirical_p_value = (permutation_results >= sample_statistic).mean()

    # Calculate the bootstrap distribution
    mean_returns_diff_dist, mean_returns_pos_pred_dist, mean_returns_neg_pred_dist = calculate_bootstrap_distribution_numba(np.column_stack((y_pred, trade_returns)), n_simulations = n_simulations)

    # Calculate the confidence interval for each distribution
    ci_mean_returns_diff = np.percentile(mean_returns_diff_dist, [0.5, 99.5])
    ci_mean_returns_diff[0] = round(ci_mean_returns_diff[0], 3)
    ci_mean_returns_diff[1] = round(ci_mean_returns_diff[1], 3)
    median_mean_returns_diff = np.median(mean_returns_diff_dist)

    ci_mean_pos_pred = np.percentile(mean_returns_pos_pred_dist, [0.5, 99.5])
    ci_mean_pos_pred[0] = round(ci_mean_pos_pred[0], 3)
    ci_mean_pos_pred[1] = round(ci_mean_pos_pred[1], 3)
    median_mean_pos_pred = np.median(mean_returns_pos_pred_dist)

    ci_mean_neg_pred = np.percentile(mean_returns_neg_pred_dist, [0.5, 99.5])
    ci_mean_neg_pred[0] = round(ci_mean_neg_pred[0], 3)
    ci_mean_neg_pred[1] = round(ci_mean_neg_pred[1], 3)
    median_mean_neg_pred = np.median(mean_returns_neg_pred_dist)

    fig, ax = plt.subplots(2, 2, figsize = (18, 6))

    # Plot the permutation test distribution
    sns.histplot(permutation_results, kde = True, color = 'black', alpha = 0.7, ax = ax[0][0])
    ax[0][0].axvline(x = sample_statistic, color = 'red', linestyle = '--', label = 'OOS test statistic')
    ax[0][0].set_title(f'Permutation Test Distribution of OOS Mean Return Differences (N = {n_simulations:,}), p = {empirical_p_value:.4f}')
    ax[0][0].set_xlabel('Mean Trade Returns Difference')
    ax[0][0].set_ylabel('Frequency')
    ax[0][0].legend()

    # Plot the bootstrap distribution
    sns.histplot(mean_returns_diff_dist, kde = True, color = 'blue', alpha = 0.7, ax = ax[0][1])
    ax[0][1].axvline(x = ci_mean_returns_diff[0], color = 'red', linestyle = '--', label = '0.5th Percentile')
    ax[0][1].axvline(x = median_mean_returns_diff, color = 'black', linestyle = '--', label = 'Median')
    ax[0][1].axvline(x = ci_mean_returns_diff[1], color = 'green', linestyle = '--', label = '99.5th Percentile')
    ax[0][1].set_title(f'Bootstraped Distribution of OOS Mean Return Differences (99% CI {ci_mean_returns_diff}, N = {n_simulations:,})')
    ax[0][1].set_xlabel('Mean Trade Returns Difference')
    ax[0][1].set_ylabel('Frequency')
    ax[0][1].legend()

    # Plot the distribution of mean returns for positive predictions
    sns.histplot(mean_returns_pos_pred_dist, kde = True, color = 'green', alpha = 0.7, ax = ax[1][0])
    ax[1][0].axvline(x = ci_mean_pos_pred[0], color = 'red', linestyle = '--', label = '0.5th Percentile')
    ax[1][0].axvline(x = median_mean_pos_pred, color = 'blue', linestyle = '--', label = 'Median')
    ax[1][0].axvline(x = ci_mean_pos_pred[1], color = 'green', linestyle = '--', label = '99.5th Percentile')
    ax[1][0].set_title(f'Bootstrapped Distribution of Mean Returns for Positive Predictions (99% CI {ci_mean_pos_pred}, N = {n_simulations:,})')
    ax[1][0].set_xlabel('Mean Trade Returns')
    ax[1][0].set_ylabel('Frequency')
    ax[1][0].legend()

    # Plot the distribution of mean returns for negative predictions
    sns.histplot(mean_returns_neg_pred_dist, kde = True, color = 'red', alpha = 0.7, ax = ax[1][1])
    ax[1][1].axvline(x = ci_mean_neg_pred[0], color = 'red', linestyle = '--', label = '0.5th Percentile')
    ax[1][1].axvline(x = median_mean_neg_pred, color = 'blue', linestyle = '--', label = 'Median')
    ax[1][1].axvline(x = ci_mean_neg_pred[1], color = 'green', linestyle = '--', label = '99.5th Percentile')
    ax[1][1].set_title(f'Bootstrapped Distribution of Mean Returns for Negative Predictions (99% CI {ci_mean_neg_pred}, N = {n_simulations:,})')
    ax[1][1].set_xlabel('Mean Trade Returns')
    ax[1][1].set_ylabel('Frequency')
    ax[1][1].legend()

    plt.tight_layout()
    plt.legend()
    plt.show()

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

@njit(nopython=True)
def value_at_risk(equity_curve, holding_period, confidence_level):
    # Calculate daily returns
    returns = custom_diff(equity_curve)

    # Calculate the VaR percentage using np.percentile
    var_percent = -np.percentile(returns, (1 - confidence_level) * 100)

    # Scale the VaR to the holding period
    var_scaled_percent = var_percent * np.sqrt(holding_period)

    return abs(var_scaled_percent)


@njit(nopython=True)
def conditional_value_at_risk(equity_curve, holding_period, confidence_level):
    # Calculate daily returns
    returns = custom_diff(equity_curve)
    
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
    num_years = len(equity) / 365
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

@njit
def alpha(equity, benchmark):
    """
    Calculates the alpha of an equity curve relative to a benchmark.

    Parameters:
    ----------
    equity : numpy.ndarray
        An array representing the equity curve.

    benchmark : numpy.ndarray
        An array representing the benchmark returns.

    Returns:
    -------
    float
        The alpha value.
    """
    # Calculate daily returns
    equity_returns = custom_diff(equity)
    benchmark_returns = benchmark

    # Calculate covariance and variance
    covariance = np.cov(equity_returns, benchmark_returns[1:])[0, 1]
    variance = np.var(benchmark_returns)

    if variance != 0:
        beta_value = covariance / variance
        alpha_value = np.mean(equity_returns) - beta_value * np.mean(benchmark_returns)
        return alpha_value
    else:
        return np.nan

@njit
def beta(equity, benchmark):
    """
    Calculates the beta of an equity curve relative to a benchmark.

    Parameters:
    ----------
    equity : numpy.ndarray
        An array representing the equity curve.

    benchmark : numpy.ndarray
        An array representing the benchmark returns.

    Returns:
    -------
    float
        The beta value.
    """
    # Calculate daily returns
    equity_returns = custom_diff(equity)
    benchmark_returns = benchmark

    # Calculate covariance and variance
    covariance = np.cov(equity_returns, benchmark[1:])[0, 1]
    variance = np.var(benchmark_returns)

    if variance != 0:
        beta_value = covariance / variance
        return beta_value
    else:
        return np.nan

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

def run_block_bootstrap(equity_curve, num_simulations=100_000):
    """
    Runs a block bootstrap on an equity curve.

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
    simulated_curves_np = simulate_equity_curves_with_block_bootstrap(equity_curve['returns'].values, initial_equity, num_simulations, 30)

    # Convert the numpy array of simulated curves to a pandas DataFrame
    dates = equity_curve.index
    simulated_curves_df = pd.DataFrame(simulated_curves_np, index=dates)

    return simulated_curves_df


def calculate_block_bootstrap_performance_metrics(monte_carlo_equity_curves, benchmark):
    """
    Calculates performance metrics for each block bootstrapped equity curve.

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
                    'max_dd': [], 'avg_dd': [], 'alpha': [], 'beta': []}

    # Iterate over each simulated equity curve
    i = 1
    for curve in monte_carlo_equity_curves.columns:
        i += 1

        equity_curve = monte_carlo_equity_curves[curve].values
        sharpe = sharpe_ratio(equity_curve)
        sortino = sortino_ratio(equity_curve)
        calmar = calmar_ratio(equity_curve)
        max_dd = max_drawdown(equity_curve)
        avg_dd = avg_drawdown(equity_curve)
        
        alpha_value = alpha(equity_curve, benchmark)
        beta_value = beta(equity_curve, benchmark)
        metrics_dict['sharpe_ratio'].append(sharpe)
        metrics_dict['sortino_ratio'].append(sortino)
        metrics_dict['calmar_ratio'].append(calmar)
        metrics_dict['max_dd'].append(max_dd)
        metrics_dict['avg_dd'].append(avg_dd)
        metrics_dict['alpha'].append(alpha_value)
        metrics_dict['beta'].append(beta_value)

        # Calculate VaR and CVaR for each holding period
        for holding_period in holding_periods:
            try:
                var = value_at_risk(equity_curve, holding_period, confidence_level)
            except:
                var = float('-inf')
            try:
                cvar = conditional_value_at_risk(equity_curve, holding_period, confidence_level)
            except:
                cvar = float('-inf')

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

def run_monte_carlo_simulation(oos_trades, equity_curve, num_simulations=100_000):
    """
    Runs a Monte Carlo simulation on out-of-sample trades.

    Given out-of-sample trades, this function resamples the trades and calculates
    various trade-level performance metrics for each resampled set of trades.

    Parameters:
    ----------
    oos_trades : pandas.DataFrame
        A DataFrame with columns:
            - 'entry_date': The entry date of the trade.
            - 'exit_date': The exit date of the trade.
            - 'symbol_id': The symbol identifier of the token traded.
            - 'size': The size of the trade.
            - 'entry_fees': The fees paid when entering the trade.
            - 'exit_fees': The fees paid when exiting the trade.
            - 'pnl': The profit and loss of the trade.
            - 'pnl_pct': The profit and loss percentage of the trade.
            -  'is_long': A boolean indicating if the trade is long (True) or short (False).

    num_simulations : int, default 1000
        The number of simulated equity curves to generate.

    Returns:
    -------
    pandas.DataFrame
        A DataFrame with columns:
            - 'avg_pnl': The average profit and loss of the resampled trades.
            - 'avg_pnl_pct': The average profit and loss percentage of the resampled trades.
            - 'expectancy': The expectancy of the resampled trades.
            - 'win_rate': The win rate of the resampled trades.
    """
    i = 0
    results = np.zeros((num_simulations, 4))

    while i < num_simulations:
        # print(f'Running simulation {i + 1}/{num_simulations}', end='\r', flush=True)

        # Resample the trades
        resampled_trades = oos_trades.sample(frac=1, replace=True)

        # Calculate the trade-level performance metrics
        avg_pnl = resampled_trades['pnl'].mean()
        avg_pnl_pct = resampled_trades['pnl_pct'].mean()
        expectancy = avg_pnl_pct
        win_rate = (resampled_trades['pnl'] > 0).mean()

        results[i] = [avg_pnl, avg_pnl_pct, expectancy, win_rate]
        i += 1

    return pd.DataFrame(results, columns=['avg_pnl', 'avg_pnl_pct', 'expectancy', 'win_rate'])
