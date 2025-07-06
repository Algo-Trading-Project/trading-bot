from backtest.strategies.portfolio.portfolio_ml_strategy import PortfolioMLStrategy
from backtest.strategies.portfolio.ma_crossover_strategy import MACrossoverStrategy

# Backtesters for single token and portfolio strategies
from backtest.backtester import BackTester
import time

if __name__ == '__main__':
    # Execute a walk-forward optimization across input strategies
    # and log the results to DuckDB/Redshift
    # Initialize a BackTester instance w/ the intended strategies to backtest and
    # a performance metric to optimize on
    b = BackTester(
        strategies = [
            PortfolioMLStrategy(optimization_metric = 'Sortino Ratio'),
        ],
        resample_period = '1D',
        start_date = '2018-11-01',
        end_date = '2025-06-01',
    )

    backtest_start = time.time()
    b.execute()
    backtest_end = time.time()
    mins_elapsed = (backtest_end - backtest_start) / 60

    print()
    print(f'Backtests completed in {mins_elapsed:.2f} minutes')
    print()