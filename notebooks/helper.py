import json
import vectorbt as vbt
import warnings
warnings.filterwarnings('ignore')

from utils.db_utils import QUERY
from analysis.stats import run_monte_carlo_simulation
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

def plot_in_sample_vs_out_of_sample_sharpe_ratios(all_metrics, strat):
    is_sharpe = all_metrics['backtest_results'].apply(lambda x: json.loads(x)['is_sharpe']).sum()
    oos_sharpe = all_metrics['backtest_results'].apply(lambda x: json.loads(x)['oos_sharpe']).sum()
    df = pd.DataFrame({
        'is_sharpe': is_sharpe,
        'oos_sharpe': oos_sharpe
    })
    # Scatter plot of In-Sample Sharpe Ratios vs. Out-of-Sample Sharpe Ratios
    r = df.corr().iloc[0, 1]
    r_2 = r ** 2

    plt.figure(figsize=(10, 5))
    plt.title(f'In-Sample vs. Out-of-Sample Sharpe Ratios for {strat} (r = {r:.2f}, r^2 = {r_2:.2f})')
    plt.xlabel('In-Sample Sharpe Ratio')
    plt.ylabel('Out-of-Sample Sharpe Ratio')
    plt.grid()

    sns.regplot(x='is_sharpe', y='oos_sharpe', data=df)

    plt.tight_layout();

def plot_strategy_equity_curve(btc, eth, strategy_equity_curve, trades_data, strat, symbol_id):
    # Calculate buy and hold equity curve for BTC as a benchmark
    btc['equity'] = 10_000 * (1 + btc['close'].pct_change().fillna(0)).cumprod()
    btc = btc.set_index('time_period_end')
    btc = btc.loc[strategy_equity_curve['date'].min():strategy_equity_curve['date'].max()]

    # Calculate buy and hold equity curve for ETH as a benchmark
    eth['equity'] = 10_000 * (1 + eth['close'].pct_change().fillna(0)).cumprod()
    eth = eth.set_index('time_period_end')
    eth = eth.loc[strategy_equity_curve['date'].min():strategy_equity_curve['date'].max()]

    # Calculate 50/50 portfolio equity curve of BTC and ETH
    w1 = 0.5
    w2 = 0.5
    port_returns = (w1 * btc['close'].pct_change().fillna(0)) + (w2 * eth['close'].pct_change().fillna(0))
    port_returns = (1 + port_returns).cumprod() * 10_000
    port_returns = port_returns.loc[strategy_equity_curve['date'].min():strategy_equity_curve['date'].max()]

    # Plot buy and hold equity curve for ETH, BTC, and portfolio
    # eth['equity'].plot(figsize=(10, 5), label='ETH Buy & Hold', color='green')
    # btc['equity'].plot(figsize=(10, 5), label='BTC Buy & Hold', color='orange')
    port_returns.plot(figsize=(10, 5), label='50/50 Portfolio', color='blue')

    plt.title(f'Equity Curve for {strat} on {symbol_id}')
    plt.xlabel('Date')
    plt.ylabel('Equity')

    # Plot strategy equity curve
    seq = strategy_equity_curve.copy()
    seq = seq.set_index('date')

    # for i in range(len(trades_data)):
    #     entry_date = trades_data.iloc[i]['entry_date']
    #     exit_date = trades_data.iloc[i]['exit_date']
    #
    #     try:
    #         entry_equity = seq.loc[entry_date, 'equity']
    #         exit_equity = seq.loc[exit_date, 'equity']
    #     except:
    #         continue
    #
    #     plt.scatter(entry_date, entry_equity, color = 'green', marker = '^')
    #     plt.scatter(exit_date, exit_equity, color = 'red', marker = 'v')

    plt.plot(seq.index, seq['equity'], label=strat, color='purple')

    plt.grid()
    plt.tight_layout()
    plt.legend()
    plt.show()

def plot_price_data(price_data, trades_data, symbol_id):
    # Plot Price Data
    plt.figure(figsize=(10, 5))
    plt.title(f'Price Data for {symbol_id}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid()

    # Scatter plot of trades and price data
    for i in range(len(trades_data)):
        entry_date = trades_data.iloc[i]['entry_date']
        exit_date = trades_data.iloc[i]['exit_date']

        try:
            entry_price = price_data.loc[entry_date, 'close']
            exit_price = price_data.loc[exit_date, 'close']
        except:
            continue

        plt.scatter(entry_date, entry_price, color='green', marker='^')
        plt.scatter(exit_date, exit_price, color='red', marker='v')

    plt.plot(price_data.index, price_data['close'])

    plt.tight_layout();

def plot_monte_carlo_equity_curves(monte_carlo_equity_curves):
    sampled_equity_curves = np.random.choice(monte_carlo_equity_curves.columns, 1000)
    title = 'Monte Carlo Simulations of Equity Curves Over Time'
    # Normalize the equity curves to the initial value
    monte_carlo_equity_curves = monte_carlo_equity_curves / monte_carlo_equity_curves.iloc[0]
    monte_carlo_equity_curves[sampled_equity_curves].plot(figsize=(20, 5), legend=False)

    plt.title(title)
    plt.xlabel('Date')
    plt.grid()
    plt.ylabel('Simulated Equity Value (USD)');

def plot_bootstrapped_sharpe_sortino_calmar_ratios(monte_carlo_risk_metrics):
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))

    # Probability of Sharpe Ratio being greater than 1 and 0
    sharpe_ratio_prob_0 = (np.array(monte_carlo_risk_metrics['sharpe_ratio']) > 0).mean()
    sharpe_ratio_prob_1 = (np.array(monte_carlo_risk_metrics['sharpe_ratio']) > 1).mean()
    sortino_ratio_prob_0 = (np.array(monte_carlo_risk_metrics['sortino_ratio']) > 0).mean()
    sortino_ratio_prob_1 = (np.array(monte_carlo_risk_metrics['sortino_ratio']) > 1).mean()
    calmar_ratio_prob_0 = (np.array(monte_carlo_risk_metrics['calmar_ratio']) > 0).mean()
    calmar_ratio_prob_1 = (np.array(monte_carlo_risk_metrics['calmar_ratio']) > 1).mean()

    # Mean Sharpe Ratio
    mean_sharpe_ratio = np.mean(monte_carlo_risk_metrics['sharpe_ratio'])
    mean_sortino_ratio = np.mean(monte_carlo_risk_metrics['sortino_ratio'])
    mean_calmar_ratio = np.mean(monte_carlo_risk_metrics['calmar_ratio'])

    # 99% Confidence Interval for Sharpe Ratio, Sortino Ratio, and Calmar Ratio
    sharpe_ratio_ci_99 = np.percentile(monte_carlo_risk_metrics['sharpe_ratio'], [0.5, 99.5])
    sortino_ratio_ci_99 = np.percentile(monte_carlo_risk_metrics['sortino_ratio'], [0.5, 99.5])
    calmar_ratio_ci_99 = np.percentile(monte_carlo_risk_metrics['calmar_ratio'], [0.5, 99.5])

    print(f'Probability of Sharpe Ratio > 0: {sharpe_ratio_prob_0:.2f}')
    print(f'Probability of Sharpe Ratio > 1: {sharpe_ratio_prob_1:.2f}')
    print()
    print(f'Probability of Sortino Ratio > 0: {sortino_ratio_prob_0:.2f}')
    print(f'Probability of Sortino Ratio > 1: {sortino_ratio_prob_1:.2f}')
    print()
    print(f'Probability of Calmar Ratio > 0: {calmar_ratio_prob_0:.2f}')
    print(f'Probability of Calmar Ratio > 1: {calmar_ratio_prob_1:.2f}')
    print()
    print(f'Mean Sharpe Ratio: {mean_sharpe_ratio:.2f}')
    print(f'Mean Sortino Ratio: {mean_sortino_ratio:.2f}')
    print(f'Mean Calmar Ratio: {mean_calmar_ratio:.2f}')
    print()
    print(f'99% Confidence Interval for Sharpe Ratio: {sharpe_ratio_ci_99}')
    print(f'99% Confidence Interval for Sortino Ratio: {sortino_ratio_ci_99}')
    print(f'99% Confidence Interval for Calmar Ratio: {calmar_ratio_ci_99}')

    # Histogram of Monte Carlo Sharpe Ratios
    axs[0].set_title('Distribution of 1,000,000 Monte Carlo Sharpe Ratios')
    axs[0].set_xlabel('Sharpe Ratio')
    axs[0].grid()
    sns.histplot(monte_carlo_risk_metrics['sharpe_ratio'], ax=axs[0], stat='probability', color='green')

    # Histogram of Monte Carlo Sortino Ratios
    axs[1].set_title('Distribution of 1,000,000 Monte Carlo Sortino Ratios')
    axs[1].set_xlabel('Sortino Ratio')
    axs[1].grid()
    sns.histplot(monte_carlo_risk_metrics['sortino_ratio'], ax=axs[1], stat='probability', color='blue');

    # Histogram of Monte Carlo Calmar Ratios
    axs[2].set_title('Distribution of 1,000,000 Monte Carlo Calmar Ratios')
    axs[2].set_xlabel('Calmar Ratio')
    axs[2].grid()
    sns.histplot(monte_carlo_risk_metrics['calmar_ratio'], ax=axs[2], stat='probability', color='red')

    plt.tight_layout();

def plot_bootstrapped_drawdowns(monte_carlo_risk_metrics):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))

    # Risk of Ruin
    risk_of_ruin = (np.array(monte_carlo_risk_metrics['max_dd']) <= -25).mean()

    # Mean Avg. Drawdown and Max Drawdown
    mean_avg_dd = np.mean(monte_carlo_risk_metrics['avg_dd'])
    mean_max_dd = np.mean(monte_carlo_risk_metrics['max_dd'])

    # 99% Confidence Interval for Avg. Drawdown and Max Drawdown
    avg_dd_ci_99 = np.percentile(monte_carlo_risk_metrics['avg_dd'], [0.5, 99.5])
    max_dd_ci_99 = np.percentile(monte_carlo_risk_metrics['max_dd'], [0.5, 99.5])

    print(f'Risk of Ruin: {risk_of_ruin:.2f}')
    print()
    print(f'Mean Avg. Drawdown: {mean_avg_dd:.2f}%')
    print(f'Mean Max Drawdown: {mean_max_dd:.2f}%')
    print()
    print(f'99% Confidence Interval for Avg. Drawdown: {avg_dd_ci_99}')
    print(f'99% Confidence Interval for Max Drawdown: {max_dd_ci_99}')

    plt.subplot(121)
    axs[0].set_title('Distribution of 1,000,000 Monte Carlo Avg. Drawdowns (%)')
    axs[0].set_xlabel('Avg. Drawdown (%)')
    axs[0].grid()

    # Histogram of Avg. Drawdown Pcts.
    sns.histplot(monte_carlo_risk_metrics['avg_dd'], ax=axs[0], stat='probability')
    plt.grid()

    plt.subplot(121)
    axs[1].set_title('Distribution of 1,000,000 Monte Carlo Max Drawdowns (%)')
    axs[1].set_xlabel('Max Drawdown (%)')
    axs[1].grid()

    # Histogram of Monte Carlo Max Drawdown Pcts.
    sns.histplot(monte_carlo_risk_metrics['max_dd'], ax=axs[1], stat='probability')
    plt.grid();

def plot_boot_strapped_var_and_cvar(monte_carlo_risk_metrics):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))

    # Value at Risk (VaR) and Conditional Value at Risk (CVaR)
    var_1_month = monte_carlo_risk_metrics['1_month_var'] * 100
    cvar_1_month = monte_carlo_risk_metrics['1_month_cvar'] * 100

    # Median VaR and CVaR
    median_var = np.median(var_1_month)
    median_cvar = np.median(cvar_1_month)

    # 95% Confidence Interval for VaR and CVaR
    var_ci = np.percentile(var_1_month, [2.5, 97.5])
    cvar_ci = np.percentile(cvar_1_month, [2.5, 97.5])

    print(f'Median 1-Year VaR: {median_var:.2f}%')
    print(f'Median 1-Year CVaR: {median_cvar:.2f}%')
    print()
    print(f'95% Confidence Interval for 1-Year VaR: {var_ci}')
    print(f'95% Confidence Interval for 1-Year CVaR: {cvar_ci}')

    plt.subplot(121)
    axs[0].set_title('Distribution of 100,000 Monte Carlo 1-Year VaR (%)')
    axs[0].set_xlabel('1-Year VaR (%)')
    axs[0].grid()

    # Histogram of 1-Year VaR Pcts.
    sns.histplot(var_1_month, ax=axs[0], stat='probability')
    plt.grid()

    plt.subplot(121)
    axs[1].set_title('Distribution of 100,000 Monte Carlo 1-Year CVaR (%)')
    axs[1].set_xlabel('1-Year CVaR (%)')
    axs[1].grid()

    # Histogram of Monte Carlo 1-Year CVaR Pcts.
    sns.histplot(cvar_1_month, ax=axs[1], stat='probability')
    plt.grid();