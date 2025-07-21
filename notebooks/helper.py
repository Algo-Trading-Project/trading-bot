import json
import vectorbtpro as vbt
import warnings
warnings.filterwarnings('ignore')

from utils.db_utils import QUERY
from analysis.stats import run_monte_carlo_simulation
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

def plot_strategy_equity_curve(benchmark, strategy_equity_curve, trades_data, strat, symbol_id):
    cagr = (strategy_equity_curve['equity'].iloc[-1] / strategy_equity_curve['equity'].iloc[0]) ** (365 / len(strategy_equity_curve)) - 1
    sharpe_ratio = strategy_equity_curve['equity'].pct_change().mean() / strategy_equity_curve['equity'].pct_change().std() * np.sqrt(365)
    sortino_ratio = strategy_equity_curve['equity'].pct_change().mean() / strategy_equity_curve['equity'].pct_change()[strategy_equity_curve['equity'].pct_change() < 0].std() * np.sqrt(365)
    total_return = (strategy_equity_curve['equity'].iloc[-1] / strategy_equity_curve['equity'].iloc[0]) - 1

    cagr_benchmark = (benchmark['equity'].iloc[-1] / benchmark['equity'].iloc[0]) ** (365 / len(benchmark)) - 1
    sharpe_ratio_benchmark = benchmark['equity'].pct_change().mean() / benchmark['equity'].pct_change().std() * np.sqrt(365)
    sortino_ratio_benchmark = benchmark['equity'].pct_change().mean() / benchmark['equity'].pct_change()[benchmark['equity'].pct_change() < 0].std() * np.sqrt(365)
    total_return_benchmark = (benchmark['equity'].iloc[-1] / benchmark['equity'].iloc[0]) - 1

    title = f'{strat} (CAGR: {cagr:.2%}, Sharpe Ratio: {sharpe_ratio:.2f}, Sortino Ratio: {sortino_ratio:.2f}, Total Return: {total_return:.2%}) vs. <br>Benchmark (CAGR: {cagr_benchmark:.2%}, Sharpe Ratio: {sharpe_ratio_benchmark:.2f}, Sortino Ratio: {sortino_ratio_benchmark:.2f}, Total Return: {total_return_benchmark:.2%})'
        
    # Plot strategy equity curve and benchmark
    # Plot year-month as xticks using Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=strategy_equity_curve['date'], y=strategy_equity_curve['equity'], mode='lines', name='Strategy Equity Curve', line=dict(color='blue', width=2)))
    fig.add_trace(go.Scatter(x=benchmark['time_period_end'], y=benchmark['equity'], mode='lines', name='Benchmark (50/50 BTC/ETH)', line=dict(color='orange')))
    fig.update_layout(title=title, xaxis_title='Date', yaxis_title='Equity Value (USD)', xaxis=dict(tickformat='%Y-%m', tickangle=90), template='plotly_dark')
    fig.show()

    # Alternative using Matplotlib
    # plt.figure(figsize=(13, 6))
    # plt.title(f'Equity Curve for {strat} ({symbol_id})')
    # plt.xlabel('Date')
    # plt.ylabel('Equity Value (USD)')
    # plt.plot(strategy_equity_curve['date'], strategy_equity_curve['equity'], label='Strategy Equity Curve', color='blue')
    # plt.plot(benchmark.index, benchmark['equity'], label='Benchmark (50/50 BTC/ETH)', color='orange')
    # plt.xticks(pd.date_range(start=strategy_equity_curve['date'].min(), end=strategy_equity_curve['date'].max(), freq='M'), rotation=90)
    
    # plt.grid()
    # plt.tight_layout()
    # plt.legend()
    # plt.show()

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

def plot_bootstrapped_sharpe_sortino_calmar_ratios(monte_carlo_risk_metrics, N):

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

    # Plot histograms of Sharpe Ratio, Sortino Ratio, and Calmar Ratio using Plotly
    fig = make_subplots(rows=1, cols=3, subplot_titles=(f'Distribution of {N:,} Bootstrapped Sharpe Ratios', f'Distribution of {N:,} Bootstrapped Sortino Ratios', f'Distribution of {N:,} Bootstrapped Calmar Ratios'))
    fig.add_trace(go.Histogram(x=monte_carlo_risk_metrics['sharpe_ratio'], nbinsx=100, name='Sharpe Ratio', marker_color='green', histnorm='probability'), row=1, col=1)
    fig.add_trace(go.Histogram(x=monte_carlo_risk_metrics['sortino_ratio'], nbinsx=100, name='Sortino Ratio', marker_color='blue', histnorm='probability'), row=1, col=2)
    fig.add_trace(go.Histogram(x=monte_carlo_risk_metrics['calmar_ratio'], nbinsx=100, name='Calmar Ratio', marker_color='red', histnorm='probability'), row=1, col=3)
    fig.update_traces(marker_line_width=1, marker_line_color='black')
    fig.update_layout(title_text=f'Distribution of {N:,} Bootstrapped Risk Metrics', showlegend=False, height=500, width=1500, template='plotly_dark')
    fig.update_xaxes(title_text='Sharpe Ratio', row=1, col=1)
    fig.update_xaxes(title_text='Sortino Ratio', row=1, col=2)
    fig.update_xaxes(title_text='Calmar Ratio', row=1, col=3)
    fig.update_yaxes(title_text='Probability', row=1, col=1)
    fig.update_yaxes(title_text='Probability', row=1, col=2)
    fig.update_yaxes(title_text='Probability', row=1, col=3)
    fig.show()

    # Alternative using Matplotlib
    # # Histogram of Monte Carlo Sharpe Ratios
    # fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
    # axs[0].set_title(f'Distribution of {N:,} Bootstrapped Sharpe Ratios')
    # axs[0].set_xlabel('Sharpe Ratio')
    # axs[0].grid()
    # sns.histplot(monte_carlo_risk_metrics['sharpe_ratio'], ax=axs[0], stat='probability', color='green')

    # # Histogram of Monte Carlo Sortino Ratios
    # axs[1].set_title(f'Distribution of {N:,} Bootstrapped Sortino Ratios')
    # axs[1].set_xlabel('Sortino Ratio')
    # axs[1].grid()
    # sns.histplot(monte_carlo_risk_metrics['sortino_ratio'], ax=axs[1], stat='probability', color='blue');

    # # Histogram of Monte Carlo Calmar Ratios
    # axs[2].set_title(f'Distribution of {N:,} Bootstrapped Calmar Ratios')
    # axs[2].set_xlabel('Calmar Ratio')
    # axs[2].grid()
    # sns.histplot(monte_carlo_risk_metrics['calmar_ratio'], ax=axs[2], stat='probability', color='red')

    # plt.tight_layout();

def plot_bootstrapped_drawdowns(monte_carlo_risk_metrics, N):
    # Risk of Ruin
    risk_of_ruin = (np.array(monte_carlo_risk_metrics['max_dd']) <= -70).mean()

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
    
    # Plot histograms of Avg. Drawdown and Max Drawdown using Plotly
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f'Distribution of {N:,} Bootstrapped Avg. Drawdowns', f'Distribution of {N:,} Bootstrapped Max Drawdowns'))
    fig.add_trace(go.Histogram(x=monte_carlo_risk_metrics['avg_dd'], nbinsx=100, name='Avg. Drawdown (%)', marker_color='blue', histnorm='probability'), row=1, col=1)
    fig.add_trace(go.Histogram(x=monte_carlo_risk_metrics['max_dd'], nbinsx=100, name='Max Drawdown (%)', marker_color='red', histnorm='probability'), row=1, col=2)
    fig.update_traces(marker_line_width=1, marker_line_color='black')
    fig.update_layout(title_text=f'Distribution of {N:,} Bootstrapped Drawdowns', showlegend=False, height=500, width=1300, template='plotly_dark')
    fig.update_xaxes(title_text='Avg. Drawdown (%)', row=1, col=1)
    fig.update_xaxes(title_text='Max Drawdown (%)', row=1, col=2)
    fig.update_yaxes(title_text='Probability', row=1, col=1)
    fig.update_yaxes(title_text='Probability', row=1, col=2)
    fig.show()

    # Alternative using Matplotlib
    # fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))
    # plt.subplot(121)
    # axs[0].set_title(f'Distribution of {N:,} Bootstrapped Avg. Drawdowns (%)')
    # axs[0].set_xlabel('Avg. Drawdown (%)')
    # axs[0].grid()

    # # Histogram of Avg. Drawdown Pcts.
    # sns.histplot(monte_carlo_risk_metrics['avg_dd'], ax=axs[0], stat='probability')
    # plt.grid()

    # plt.subplot(121)
    # axs[1].set_title(f'Distribution of {N:,} Bootstrapped Max Drawdowns (%)')
    # axs[1].set_xlabel('Max Drawdown (%)')
    # axs[1].grid()

    # # Histogram of Monte Carlo Max Drawdown Pcts.
    # sns.histplot(monte_carlo_risk_metrics['max_dd'], ax=axs[1], stat='probability')
    # plt.grid();

def plot_bootstrapped_alpha_beta(monte_carlo_risk_metrics, N):
    # Alpha and Beta
    alpha = np.array(monte_carlo_risk_metrics['alpha'])
    beta = np.array(monte_carlo_risk_metrics['beta'])

    # Mean Alpha and Beta
    mean_alpha = np.mean(alpha)
    mean_beta = np.mean(beta)

    # 99% Confidence Interval for Alpha and Beta
    alpha_ci_99 = np.percentile(alpha, [0.5, 99.5])
    beta_ci_99 = np.percentile(beta, [0.5, 99.5])

    print(f'Probability of Alpha > 0: {(alpha > 0).mean():.2f}')
    print(f'Probability of Beta ∈ [-0.05, 0.05]: {((beta >= -0.05) & (beta <= 0.05)).mean():.2f}')
    print()
    print(f'Mean Alpha: {mean_alpha:.4f}')
    print(f'Mean Beta: {mean_beta:.4f}')
    print()
    print(f'99% Confidence Interval for Alpha: {alpha_ci_99}')
    print(f'99% Confidence Interval for Beta: {beta_ci_99}')
    
    # Histogram of Alpha and Beta using Plotly
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f'Distribution of {N:,} Bootstrapped Alphas', f'Distribution of {N:,} Bootstrapped Betas'))
    fig.add_trace(go.Histogram(x=alpha, nbinsx=100, name='Alpha', marker_color='purple', histnorm='probability'), row=1, col=1)
    fig.add_trace(go.Histogram(x=beta, nbinsx=100, name='Beta', marker_color='orange', histnorm='probability'), row=1, col=2)
    fig.update_traces(marker_line_width=1, marker_line_color='black')
    fig.update_layout(title_text=f'Distribution of {N:,} Bootstrapped Alpha and Beta', showlegend=False, height=500, width=1300, template='plotly_dark')
    fig.update_xaxes(title_text='Alpha', row=1, col=1)
    fig.update_xaxes(title_text='Beta', row=1, col=2)
    fig.update_yaxes(title_text='Probability', row=1, col=1)
    fig.update_yaxes(title_text='Probability', row=1, col=2)
    fig.show()
    
    # Alternative using Matplotlib
    # fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))
    # plt.subplot(121)
    # axs[0].set_title(f'Distribution of {N:,} Bootstrapped Alphas')
    # axs[0].set_xlabel('Alpha')
    # axs[0].grid()

    # # Histogram of Alphas
    # sns.histplot(alpha, ax=axs[0], stat='probability', color='purple')
    # plt.grid()

    # plt.subplot(121)
    # axs[1].set_title(f'Distribution of {N:,} Bootstrapped Betas')
    # axs[1].set_xlabel('Beta')
    # axs[1].grid()

    # # Histogram of Betas
    # sns.histplot(beta, ax=axs[1], stat='probability', color='orange')
    # plt.grid();