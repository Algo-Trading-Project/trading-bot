from backtest.strategies.portfolio.portfolio_ml_strategy import PortfolioMLStrategy
from backtest.strategies.portfolio.ma_crossover_strategy import MACrossoverStrategy
from backtest.strategies.portfolio.cross_sectional_alpha_rank_strategy import CrossSectionalAlphaRankStrategy

# Backtesters for single token and portfolio strategies
from backtest.backtester import BackTester
import time

if __name__ == '__main__':
    is_backtest = True

    # Execute a walk-forward optimization across input strategies
    # and log the results to DuckDB/Redshift
    if is_backtest:

        # Initialize a BackTester instance w/ the intended strategies to backtest and
        # a performance metric to optimize on
        b = BackTester(
            strategies = [
                CrossSectionalAlphaRankStrategy(optimization_metric = 'Sortino Ratio'),
            ],
            resample_period = '1d',
            use_dollar_bars = False,
            start_date = '2018-01-01',
            end_date = '2024-12-31'
        )

        backtest_start = time.time()
        b.execute()
        backtest_end = time.time()
        mins_elapsed = (backtest_end - backtest_start) / 60

        print()
        print(f'Backtests completed in {mins_elapsed:.2f} minutes')
        print()
    else:
        pass

