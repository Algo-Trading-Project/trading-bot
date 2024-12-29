# ML-based strategy on a single token
from backtest.strategies.single_asset.ml_strategy import MLStrategy
# ML-based strategy on a portfolio
from backtest.strategies.portfolio.portfolio_ml_strategy import PortfolioMLStrategy

# Backtesters for single token and portfolio strategies
from backtest.backtester import BackTester
from backtest.portfolio_backtester import PortfolioBackTester

import time


if __name__ == '__main__':

    # Execute a walk-forward optimization across all single-token strategies
    # and log the results to DuckDB

    # 1. Single Token Strategies
    # print('Backtesting Single Token Strategies')
    # print('-----------------------------------')
    # print()

    # # Initialize a BackTester instance w/ the intended strategies to backtest and
    # # a performance metric to optimize on
    # b = BackTester(
    #     strategies = [MLStrategy()],
    #     optimization_metric = 'Sortino Ratio',
    #     resample_period = '1d',
    #     use_dollar_bars = False
    # )

    # backtest_start = time.time()
    # b.execute()
    # backtest_end = time.time()
    # mins_elapsed = (backtest_end - backtest_start) / 60

    # print()
    # print(f'Backtests completed in {mins_elapsed:.2f} minutes')
    # print()

    # Execute an out-of-sample backtest across all portfolio strategies
    # and log the results to DuckDB 

    # 2. Portfolio Strategies
    print('Backtesting Portfolio Strategies')
    print('--------------------------------')
    print()

    # Initialize a PortfolioBackTester instance w/ the intended strategies to backtest
    pb = PortfolioBackTester(
        strategies = [PortfolioMLStrategy()],
        resample_period = '1d'
    )

    backtest_start = time.time()
    pb.execute()
    backtest_end = time.time()
    mins_elapsed = (backtest_end - backtest_start) / 60

    print()
    print(f'Backtests completed in {mins_elapsed:.2f} minutes')
    print()
