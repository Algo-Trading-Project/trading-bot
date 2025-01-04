from numba import njit, prange
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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
def returns_nb(arr: np.ndarray) -> np.ndarray:
    """
    Calculate the returns of an array of values.

    Args:
        arr: numpy.ndarray
            An array of values.

    Returns:
        numpy.ndarray
            An array of returns.
    """
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
    returns = returns_nb(equity_curve)

    # Calculate the VaR percentage using np.percentile
    var_percent = -np.percentile(returns, (1 - confidence_level) * 100)

    # Scale the VaR to the holding period
    var_scaled_percent = var_percent * np.sqrt(holding_period)

    return abs(var_scaled_percent)

@njit(nopython=True)
def conditional_value_at_risk(equity_curve, holding_period, confidence_level):
    # Calculate daily returns
    returns = returns_nb(equity_curve)

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
    returns = returns_nb(equity)
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
    returns = returns_nb(equity)
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
    returns = returns_nb(equity)
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

def calculate_maximum_adverse_excursion(equity_curve):
    pass

def calculate_maximum_favorable_excursion(equity_curve):
    pass

def calculate_edge_ratio(equity_curve):
    pass

def calculate_single_monte_carlo_equity_curve(resampled_trades, equity_curve):
    """
    Calculates a single equity curve trajectory by resampling trades.

    Args:
        resampled_trades: pandas.DataFrame
            A DataFrame with columns:
                - 'entry_date': The entry date of the trade.
                - 'exit_date': The exit date of the trade.
                - 'symbol_id': The symbol identifier of the token traded.
                - 'size': The size of the trade.
                - 'entry_fees': The fees paid when entering the trade.
                - 'exit_fees': The fees paid when exiting the trade.
                - 'pnl': The profit and loss of the trade.
                - 'pnl_pct': The profit and loss percentage of the trade.
                - 'is_long': A boolean indicating if the trade is long (True) or short (False).

        equity_curve: pandas.DataFrame
            A DataFrame with a column 'equity' representing the value of an equity over time.

    Returns:
        pandas.Series
            A Series representing the simulated equity curve trajectory.
    """
    initial_equity = equity_curve['equity'][0]
    new_equity_curve = pd.Series(index=equity_curve.index)
    equity_curve['returns'] = equity_curve['equity'].pct_change().fillna(0)
    start_index_date = equity_curve.index[0]

    for i, trade in resampled_trades.iterrows():
        # Get the entry and exit dates of the trade
        entry_date = trade['entry_date']
        exit_date = trade['exit_date']
        position_len_days = (exit_date - entry_date).days

        # Fill in the next n days with the original equity curve returns using the
        # position length of the trade and the start index date
        min_end_date = min(start_index_date + pd.Timedelta(days=position_len_days), equity_curve.index[-1])
        new_equity_curve.loc[start_index_date:min_end_date] = equity_curve.loc[entry_date:exit_date]['returns']

        # Update the start index date
        start_index_date = min_end_date + pd.Timedelta(days=1)

    # Fill in the remaining days with 0 returns
    new_equity_curve.loc[start_index_date:] = 0

    # Calculate the equity curve
    new_equity_curve = (1 + new_equity_curve).cumprod() * initial_equity
    new_equity_curve = new_equity_curve.fillna(method='ffill')

    return new_equity_curve

def calculate_monte_carlo_equity_curves(oos_trades, equity_curve, num_simulations=100_000):
    """
    Simulates multiple equity curves by resampling trades.

    Given out-of-sample trades, this function resamples the trades and calculates
    multiple possible future equity curve trajectories.

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

    equity_curve : pandas.DataFrame
        A DataFrame with a column 'equity' representing the value of an equity over time.

    num_simulations : int, default 1000
        The number of simulated equity curves to generate.

    Returns:
    -------
    pandas.DataFrame
        A DataFrame where each column is a simulated equity curve, indexed by the same dates as the input
        equity_curve.

    pandas.DataFrame
        A DataFrame with columns:
            - 'avg_pnl': The average profit and loss of the resampled trades.
            - 'avg_pnl_pct': The average profit and loss percentage of the resampled trades.
            - 'expectancy': The expectancy of the resampled trades.
            - 'win_rate': The win rate of the resampled trades.
            - 'maximum_adverse_excursion': The maximum adverse excursion of the resampled trades.
            - 'maximum_favorable_excursion': The maximum favorable excursion of the resampled trades.
            - 'edge_ratio': The edge ratio of the resampled trades.
    """
    # Initialize dictionary to store trade-level metrics
    metrics_dict = {
        'avg_pnl': [], 'avg_pnl_pct': [], 'expectancy': [], 'win_rate': [],
        'maximum_adverse_excursion': [], 'maximum_favorable_excursion': [], 'edge_ratio': []
    }

    # Initialize DataFrame to store simulated equity curves
    simulated_equity_curves = pd.DataFrame(index=equity_curve.index)

    for i in range(num_simulations):
        print(f'Simulating equity curve {i + 1}/{num_simulations}...', end='\r', flush=True)
        # Sample trades with replacement
        sampled_trades = oos_trades.sample(frac=1, replace=True)

        # Calculate trade-level metrics
        avg_pnl = sampled_trades['pnl'].mean()
        avg_pnl_pct = sampled_trades['pnl_pct'].mean()
        avg_win_pnl_pct = sampled_trades[sampled_trades['pnl'] > 0]['pnl_pct'].mean()
        avg_loss_pnl_pct = sampled_trades[sampled_trades['pnl'] < 0]['pnl_pct'].mean()
        win_rate = (sampled_trades['pnl'] > 0).mean()
        expectancy = avg_win_pnl_pct * win_rate - abs(avg_loss_pnl_pct) * (1 - win_rate)
        # max_adverse_excursion = calculate_maximum_adverse_excursion(sampled_trades)
        # max_favorable_excursion = calculate_maximum_favorable_excursion(sampled_trades)
        # edge_ratio = calculate_edge_ratio(sampled_trades)

        # Assign to appropriate keys in the dictionary
        metrics_dict['avg_pnl'].append(avg_pnl)
        metrics_dict['avg_pnl_pct'].append(avg_pnl_pct)
        metrics_dict['win_rate'].append(win_rate)
        metrics_dict['expectancy'].append(expectancy)
        # metrics_dict['maximum_adverse_excursion'].append(max_adverse_excursion)
        # metrics_dict['maximum_favorable_excursion'].append(max_favorable_excursion)
        # metrics_dict['edge_ratio'].append(edge_ratio)

        # Calculate the simulated equity curve
        simulated_equity_curve = calculate_single_monte_carlo_equity_curve(sampled_trades, equity_curve)
        simulated_equity_curves[f'sim_{i}'] = simulated_equity_curve

    return simulated_equity_curves, metrics_dict

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
        A DataFrame where each column is a simulated equity curve, indexed by the same dates as the input
        equity_curve.

    pandas.DataFrame
        A DataFrame with columns:
            - 'avg_pnl': The average profit and loss of the resampled trades.
            - 'avg_pnl_pct': The average profit and loss percentage of the resampled trades.
            - 'expectancy': The expectancy of the resampled trades.
            - 'win_rate': The win rate of the resampled trades.
            - 'maximum_adverse_excursion': The maximum adverse excursion of the resampled trades.
            - 'maximum_favorable_excursion': The maximum favorable excursion of the resampled trades.
            - 'edge_ratio': The edge ratio of the res
    """
    # Run Monte Carlo simulation on resampled trades
    simulated_equity_curves, metrics_df = calculate_monte_carlo_equity_curves(oos_trades, equity_curve, num_simulations=num_simulations)
    return simulated_equity_curves, metrics_df