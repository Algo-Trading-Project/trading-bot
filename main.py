from backtest.strategies.ma_crossover import MACrossOver
from backtest.BackTester import BackTester

import time

if __name__ == '__main__':

    # Initialize a BackTester instance w/ the intended strategies to backtest and
    # a performance metric to optimize on

    b = BackTester(
        strategies = [MACrossOver()],
        optimization_metric = 'Sortino Ratio'
    )

    # Execute a walk-forward optimization across all strategies
    # and log the results to Redshift
    
    backtest_start = time.time()
    b.execute()
    backtest_end = time.time()

    print()
    print('Total Time Elapsed: {} mins'.format(round(abs(backtest_end - backtest_start) / 60.0, 2)))    