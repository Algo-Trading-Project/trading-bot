from datetime import timedelta
from scipy.stats import norm

import pandas as pd
import numpy as np

# STANDARD PERFORMANCE METRICS

def exposure_time(duration, trades):
    exposure = timedelta(days = 0, hours = 0)

    for i in range(len(trades)):
        entry_date = pd.to_datetime(trades.at[i, 'entry_date'])
        exit_date = pd.to_datetime(trades.at[i, 'exit_date'])
        trade_duration = exit_date - entry_date
        exposure += trade_duration
    
    return round(exposure / duration * 100, 2)

def buy_and_hold_return(data):
    token1_start_value = data.at[data.index[0], data.columns[0]]
    token1_end_value = data.at[data.index[-1], data.columns[0]]
    buy_and_hold_return_token_1 = round((token1_end_value - token1_start_value) / token1_start_value * 100, 2)
    
    token2_start_value = data.at[data.index[0], data.columns[1]]
    token2_end_value = data.at[data.index[-1], data.columns[1]]
    buy_and_hold_return_token_2 = round((token2_end_value - token2_start_value) / token2_start_value * 100, 2)

    return max([buy_and_hold_return_token_1, buy_and_hold_return_token_2])

def sharpe_ratio(equity):
    returns = equity.pct_change()
    mean_returns = returns.mean()
    std_returns = returns.std()
    
    try:
        return np.sqrt(8760) * mean_returns / std_returns 
    except:
        return np.nan
    
def sortino_ratio(equity):
    returns = equity.pct_change()
    negative_returns = returns[returns['equity'] < 0]
    
    mean_returns = returns.mean()
    std_negative_returns = negative_returns.std()
    
    try:
        return np.sqrt(8760) * mean_returns / std_negative_returns 
    except:
        return np.nan
    
def calmar_ratio(equity):
    num_years = len(equity) / 8760
    cum_ret_final = (1 + equity.pct_change()).prod().squeeze()
    annual_returns = cum_ret_final ** (1 / num_years) - 1
    
    try:
        return annual_returns / abs(max_drawdown(equity) / 100)
    except:
        return np.nan

def cagr_over_avg_drawdown(equity):
    num_years = len(equity) / 8760
    cum_ret_final = (1 + equity.pct_change()).prod().squeeze()
    annual_returns = cum_ret_final ** (1 / num_years) - 1
    
    try:
        return annual_returns / abs(avg_drawdown(equity) / 100)
    except:
        return np.nan

def profit_factor(trades):
    if len(trades) == 0:
        return np.nan
    
    gross_profit = trades[trades['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(trades[trades['pnl'] < 0]['pnl'].sum())
    
    return gross_profit / gross_loss

def max_drawdown(equity):
    # Prepare for Maximum Drawdown calculation
    rolling_max = equity['equity'].expanding(min_periods=1).max()
    drawdown = equity['equity'] / rolling_max - 1

    # Maximum Drawdown percentage calculation for each simulation
    max_dd_pct = drawdown.min() * 100

    return max_dd_pct

def avg_drawdown(equity):
    # Prepare for Avg. Drawdown calculation
    rolling_max = equity['equity'].expanding(min_periods=1).max()
    drawdown = equity['equity'] / rolling_max - 1

    # Avg. Drawdown percentage calculation for each simulation
    avg_dd_pct = drawdown.mean() * 100

    return avg_dd_pct

def max_drawdown_duration(equity_curve_df):
    equity = equity_curve_df['equity']
    rolling_max = equity.cummax()
    drawdown = (equity / rolling_max) - 1

    in_drawdown = drawdown < 0
    drawdown_starts = in_drawdown & (~in_drawdown.shift(1, fill_value=False))
    drawdown_ends = (~in_drawdown) & in_drawdown.shift(1, fill_value=False)

    if drawdown_starts.sum() > drawdown_ends.sum():
        drawdown_ends.iloc[-1] = True

    drawdown_durations = (drawdown_ends[drawdown_ends].index - drawdown_starts[drawdown_starts].index).total_seconds() / (24 * 3600)

    if drawdown_durations.empty:
        return 0
    else:
        max_dd_duration_days = drawdown_durations.max()
        return max_dd_duration_days

def avg_drawdown_duration(equity_curve_df):
    equity = equity_curve_df['equity']
    rolling_max = equity.cummax()
    drawdown = (equity / rolling_max) - 1

    in_drawdown = drawdown < 0
    drawdown_starts = in_drawdown & (~in_drawdown.shift(1, fill_value=False))
    drawdown_ends = (~in_drawdown) & in_drawdown.shift(1, fill_value=False)

    if drawdown_starts.sum() > drawdown_ends.sum():
        drawdown_ends.iloc[-1] = True

    drawdown_durations = (drawdown_ends[drawdown_ends].index - drawdown_starts[drawdown_starts].index)

    drawdown_durations_days = drawdown_durations.total_seconds() / (24 * 3600)

    if not drawdown_durations_days.size:
        return 0
    else:
        avg_dd_duration_days = np.mean(drawdown_durations_days)
        return avg_dd_duration_days

def win_rate(trades):
    if len(trades) == 0:
        return np.nan
    
    num_winning_trades = len(trades[trades['pnl_pct'] > 0])
    num_trades_total = len(trades)

    return round(num_winning_trades / num_trades_total * 100, 2)

def best_trade(trades):
    if len(trades) == 0:
        return np.nan
    
    return round(trades['pnl_pct'].max() * 100, 2)
    
def worst_trade(trades):
    if len(trades) == 0:
        return np.nan
    
    return round(trades['pnl_pct'].min() * 100, 2)

def avg_trade(trades):
    if len(trades) == 0:
        return np.nan
    
    return round(trades['pnl_pct'].mean() * 100, 2)

def max_trade_duration(trades):
    if len(trades) == 0:
        return np.nan
    
    return (pd.to_datetime(trades['exit_date']) - pd.to_datetime(trades['entry_date'])).dt.total_seconds().max()            
    
def avg_trade_duration(trades):
    if len(trades) == 0:
        return np.nan
    
    return (pd.to_datetime(trades['exit_date']) - pd.to_datetime(trades['entry_date'])).dt.total_seconds().mean()           

def calculate_performance_metrics(oos_equity_curve, 
                                  oos_trades, 
                                  oos_price_data):
                
    start = oos_price_data.index[0]
    end = oos_price_data.index[-1]
    duration = pd.to_datetime(end) - pd.to_datetime(start)

    return_pct = (oos_equity_curve['equity'].iloc[-1] - oos_equity_curve['equity'].iloc[0]) / oos_equity_curve['equity'].iloc[0]

    print('Max DD Duration: ', max_drawdown_duration(oos_equity_curve))
    print('Avg DD Duration: ', avg_drawdown_duration(oos_equity_curve))
    print()
    
    metrics_dict = {
        'duration':duration, 
        'exposure_pct': round(exposure_time(duration, oos_trades), 2),
        'equity_final':round(oos_equity_curve.dropna().iloc[-1]['equity'], 2),
        'equity_peak':round(oos_equity_curve['equity'].max(), 2),
        'return_pct':round(return_pct * 100, 2),
        'buy_and_hold_return_pct':round(buy_and_hold_return(oos_price_data), 2),
        'sharpe_ratio':round(sharpe_ratio(oos_equity_curve), 2), 
        'sortino_ratio':round(sortino_ratio(oos_equity_curve), 2),
        'calmar_ratio':round(calmar_ratio(oos_equity_curve), 2),
        'cagr_over_avg_drawdown':round(cagr_over_avg_drawdown(oos_equity_curve), 2),
        'profit_factor':round(profit_factor(oos_trades), 2),
        'max_dd_pct':round(max_drawdown(oos_equity_curve), 2),
        'avg_dd_pct':round(avg_drawdown(oos_equity_curve), 2),
        'max_dd_duration':max_drawdown_duration(oos_equity_curve), 
        'avg_dd_duration':avg_drawdown_duration(oos_equity_curve),
        'num_trades':len(oos_trades), 
        'win_rate_pct':round(win_rate(oos_trades), 2), 
        'best_trade_pct':round(best_trade(oos_trades), 2),
        'worst_trade_pct':round(worst_trade(oos_trades), 2), 
        'avg_trade_pct':round(avg_trade(oos_trades), 2),
        'max_trade_duration':max_trade_duration(oos_trades),
        'avg_trade_duration':avg_trade_duration(oos_trades),
        '1_day_value_at_risk':value_at_risk(oos_equity_curve['equity'], 24, 0.95),
        '1_week_value_at_risk':value_at_risk(oos_equity_curve['equity'], 24 * 7, 0.95),
        '1_month_value_at_risk':value_at_risk(oos_equity_curve['equity'], 24 * 30, 0.95),
        '1_day_cvar':conditional_value_at_risk(oos_equity_curve['equity'], 24, 0.95),
        '1_week_cvar':conditional_value_at_risk(oos_equity_curve['equity'], 24 * 7, 0.95),
        '1_month_cvar':conditional_value_at_risk(oos_equity_curve['equity'], 24 * 30, 0.95)
    }

    return pd.DataFrame(metrics_dict).reset_index(drop = True)

def value_at_risk(equity_curve, holding_period, confidence_level):
    # Calculate daily returns from the equity curve
    returns = equity_curve.pct_change().dropna()

    # Calculate the VaR percentage using np.percentile
    var_percent = np.percentile(returns, (1 - confidence_level) * 100)

    # Scale the VaR to the holding period
    var_scaled_percent = var_percent * np.sqrt(holding_period)

    # Calculate VaR in dollars
    current_equity = equity_curve.iloc[-1]
    var_dollars = current_equity * var_scaled_percent

    var_pct = abs(var_dollars) / current_equity # Ensure positive value

    return var_pct

def conditional_value_at_risk(equity_curve, holding_period, confidence_level):
    # Calculate daily returns from the equity curve
    returns = equity_curve.pct_change().dropna()

    # Calculate the VaR percentage using np.percentile
    var_percent = np.percentile(returns, (1 - confidence_level) * 100)

    # Identify the returns that are worse than the VaR
    worse_than_var_returns = returns[returns <= var_percent]

    # Calculate the mean of these returns
    mean_worse_than_var_returns = worse_than_var_returns.mean()

    # Scale the CVaR to the holding period
    cvar_scaled_percent = mean_worse_than_var_returns * np.sqrt(holding_period)

    # Calculate CVaR in dollars
    current_equity = equity_curve.iloc[-1]
    cvar_dollars = current_equity * cvar_scaled_percent

    cvar_pct = abs(cvar_dollars) / current_equity # Ensure positive value

    return cvar_pct

# CALCULATES DEFLATED SHARPE RATIO

def _approximate_expected_maximum_sharpe(mean_sharpe, 
                                        var_sharpe, 
                                        nb_trials):
    # universal constants
    gamma = 0.5772156649015328606
    e = np.exp(1)

    return mean_sharpe + np.sqrt(var_sharpe) * (
        (1 - gamma) * norm.ppf(1 - 1 / nb_trials) + gamma * norm.ppf(1 - 1 / (nb_trials * e)))

def compute_deflated_sharpe_ratio(estimated_sharpe,
                                  sharpe_variance,
                                  nb_trials,
                                  backtest_horizon, 
                                  skew, 
                                  kurtosis):
    
    SR0 = _approximate_expected_maximum_sharpe(0, sharpe_variance, nb_trials)
    
    return (norm.cdf(((estimated_sharpe - SR0) * np.sqrt(backtest_horizon - 1))) 
            / np.sqrt(1 - skew * estimated_sharpe + ((kurtosis - 1) / 4) * estimated_sharpe**2))

# PERFORMANCE METRICS FOR MONTE CARLO SIMULATIONS

def calculate_monte_carlo_metrics(simulated_curves_df):
    """
    Calculates various performance metrics for each Monte Carlo simulation and returns them in a dictionary.

    Each key in the dictionary corresponds to a performance metric (e.g., 'return_pct', 'sharpe_ratio'), and
    the value is a list containing the computed metric for each Monte Carlo simulation, rounded to 3 decimal places.

    Parameters:
    ----------
    simulated_curves_df : pandas.DataFrame
        A DataFrame with each column representing a Monte Carlo simulated equity curve.
        Rows are indexed by dates or time periods, and values represent the simulated equity value.

    Returns:
    -------
    metrics_dict : dict
        A dictionary where each key is a metric name and the value is a list of metric values,
        one for each Monte Carlo simulation, rounded to 3 decimal places.

    Notes:
    -----
    - Assumes hourly data for annualization purposes.
    - Assumes that the input DataFrame contains no NaN values at the start.
    """

    metrics_dict = {}

    # Sharpe and Sortino Ratios for each simulation, rounded to 3 decimal places
    daily_returns = simulated_curves_df.pct_change().fillna(0)
    annualized_return = daily_returns.mean() * 8760
    annualized_std = daily_returns.std() * np.sqrt(8760)
    negative_std = daily_returns[daily_returns < 0].std() * np.sqrt(8760)

    metrics_dict['sharpe_ratio'] = [round(val, 3) for val in (annualized_return / annualized_std)]
    metrics_dict['sortino_ratio'] = [round(val, 3) for val in (annualized_return / negative_std)]

    # Prepare for Maximum Drawdown calculation
    rolling_max = simulated_curves_df.expanding(min_periods=1).max()
    drawdown = simulated_curves_df / rolling_max - 1

    # Maximum Drawdown percentage calculation for each simulation
    max_dd_pct = drawdown.min() * 100
    metrics_dict['max_dd_pct'] = [round(val, 3) for val in max_dd_pct]
   
    # Corrected calculation for Annualized Return (compounded) for Calmar Ratio
    compounded_annualized_return = 100 * (((1 + daily_returns.mean()) ** 8760) - 1)

    # Calmar Ratio calculation, rounded to 3 decimal places
    metrics_dict['calmar_ratio'] = [round(val, 3) for val in (compounded_annualized_return / abs(max_dd_pct))]

    #Average Drawdown percentage, rounded to 3 decimal places
    avg_dd_pct = drawdown.mean() * 100
    metrics_dict['avg_dd_pct'] = [round(val, 3) for val in avg_dd_pct]

    return metrics_dict