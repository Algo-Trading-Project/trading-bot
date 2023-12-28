from backtest.strategies.ma_crossover import MACrossOver
from backtest.strategies.bollinger_bands import BollingerBands
from backtest.backtester import BackTester

import time

if __name__ == '__main__': 

    # Initialize a BackTester instance w/ the intended strategies to backtest,
    # a performance metric to optimize on, and a dictionary of backtest hyperparameters

    b = BackTester(
        strategies = [MACrossOver, BollingerBands],
        optimization_metric = 'Sortino Ratio'
    )

    # Execute a walk-forward optimization across all strategies
    # and log the results to Redshift
    
    backtest_start = time.time()
    b.execute()
    backtest_end = time.time()

    print()
    print('Total Time Elapsed: {} mins'.format(round(abs(backtest_end - backtest_start) / 60.0, 2)))    