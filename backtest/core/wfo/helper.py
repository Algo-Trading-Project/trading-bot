import numpy as np
import vectorbt as vbt
import pandas as pd

"""
This file contains helper functions for calculating the dynamic 
stop-loss, take-profit, and position size during the walk-forward optimization 
process.
"""

def calculate_sl(open, high, low, close, volume, method, window):
    if method == 'std':
        # Calculate the rolling standard deviation of the close price
        # over the window
        rolling_std = pd.Series(close).rolling(window).std().values

        # Calculate the stop-loss price as close price minus 1 times
        # the rolling standard deviation
        sl = np.nan_to_num(close - 1 * rolling_std, nan = 0.05)

        # The stop-loss as a percentage of the close price
        sl = sl / close

        return sl

    elif method == 'atr':
        # Calculate the rolling average true range of the close price
        # over the window
        rolling_atr = vbt.ATR.run(high, low, close, window = window).atr

        # Calculate the stop-loss price as close price minus 2 times
        # the rolling average true range
        sl = np.nan_to_num(close - 2 * rolling_atr, nan = 0.05)

        # The stop-loss as a percentage of the close price
        sl = sl / close

        return sl
    else:
        raise ValueError('Invalid stop-loss method')

def calculate_tp(open, high, low, close, volume, method, window):
    if method == 'std':
        # Calculate the rolling standard deviation of the close price
        # over the window
        rolling_std = pd.Series(close).rolling(window).std().values

        # Calculate the take-profit price as close price plus 2 times
        # the rolling standard deviation
        tp = np.nan_to_num(close * (1.0029) + 2 * rolling_std, nan = 0.05)

        # The take-profit as a percentage of the close price
        tp = tp / close

        return tp
    elif method == 'atr':
        # Calculate the rolling average true range of the close price
        # over the window
        rolling_atr = vbt.ATR.run(high, low, close, window = window).atr

        # Calculate the take-profit price as close price plus 2 times
        # the rolling average true range
        tp = np.nan_to_num(close + rolling_atr, nan = 0.05)

        # The take-profit as a percentage of the close price
        tp = tp / close

        return tp
    else:
        raise ValueError('Invalid take-profit method')

def calculate_size(open, high, low, close, volume, backtest_params):
    if backtest_params['size'] == 'atr':
        # Calculate the rolling average true range of the close price
        # over the last week
        rolling_atr = vbt.ATR.run(high, low, close, window = 60 * 24 * 7).atr

        # Normalize the average true range by dividing it by the close price
        rolling_atr = (rolling_atr / close) * 100

        # Calculate the position size as the percentage of the account
        # that should be risked on each trade divided by the average true range
        size = 0.25 / rolling_atr

        return size
    elif backtest_params['size'] == 'std':
        # Calculate the rolling standard deviation of the close price
        # over the last week
        rolling_std = pd.Series(close).rolling(60 * 24 * 7).std().values

        # Normalize the standard deviation by dividing it by the close price
        rolling_std = (rolling_std / close) * 100

        # Calculate the position size as the percentage of the account
        # that should be risked on each trade divided by the standard deviation
        size = 0.25 / rolling_std

        return size
    else:
        raise ValueError('Invalid position sizing method')
