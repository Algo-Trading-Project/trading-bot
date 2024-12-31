# ML-based strategy on a single token
from backtest.strategies.single_asset.ml_strategy import MLStrategy

# ML-based strategy on a portfolio
from backtest.strategies.portfolio.portfolio_ml_strategy import PortfolioMLStrategy

# Backtesters for single token and portfolio strategies
from backtest.backtester import BackTester

import time


if __name__ == '__main__':
    # Execute a walk-forward optimization across all single-token strategies
    # and log the results to DuckDB

    # Initialize a BackTester instance w/ the intended strategies to backtest and
    # a performance metric to optimize on
    b = BackTester(
        strategies = [PortfolioMLStrategy()],
        optimization_metric = 'Sortino Ratio',
        resample_period = '1d',
        use_dollar_bars = False
    )

    backtest_start = time.time()
    b.execute()
    backtest_end = time.time()
    mins_elapsed = (backtest_end - backtest_start) / 60

    print()
    print(f'Backtests completed in {mins_elapsed:.2f} minutes')
    print()

