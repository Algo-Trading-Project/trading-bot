import numpy as np
import pandas as pd
import vectorbtpro as vbt

class BaseStrategy:

    # Set backtest parameters
    
    # init_cash - Initial cash
    # fees      - Comission percent
    # sl_stop   - Stop-loss percent
    # sl_trail  - Indicate whether or not want a trailing stop-loss
    # tp_stop   - Take-profit percent
    # size      - Percentage of capital to use for each trade
    # size_type - Indicates the 'size' parameter represents a percent

    backtest_params = {
        'init_cash': 10_000,
        'fees': 0.0029,
        'sl_stop': 0.1,
        'sl_trail': True,
        'tp_stop': 0.1,
        'size': 0.05,
        'size_type': 2 # e.g. 2 if 'size' is 'Fixed Percent' and 0 otherwise
    }

    optimize_dict = {}

    indicator_factory_dict = {
        'class_name':'BaseStrategy',
        'short_name':'base_strategy',
        'input_names':['open', 'high', 'low', 'close', 'volume', 'trades_count'],
        'param_names':[],
        'output_names':['entries', 'exits', 'tp', 'sl', 'size']
    }

    def calculate_sl(self, open, high, low, close, volume, backtest_params, window):
        if type(backtest_params['sl_stop']) == float:
            sl = np.abs(np.full(len(close), backtest_params['sl_stop']))
            return sl

        elif backtest_params['sl_stop'] == 'std':
            # Calculate the rolling standard deviation of the close price
            # over the window
            rolling_std = pd.Series(close).pct_change().rolling(window, min_periods = 1).std().values
            # Calculate the stop-loss price as 2 times the rolling standard deviation of returns
            # below the close price
            sl = 2 * rolling_std
            return sl

        elif backtest_params['sl_stop'] == 'atr':
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

    def calculate_tp(self, open, high, low, close, volume, backtest_params, window):
        if type(backtest_params['tp_stop']) == float:
            tp = np.abs(np.full(len(close), backtest_params['tp_stop']))
            return tp

        elif backtest_params['tp_stop'] == 'std':
            # Calculate the rolling standard deviation of the close price
            # over the window
            rolling_std = pd.Series(close).pct_change().rolling(window, min_periods = 1).std().values

            # Calculate the take-profit price as 2 times the rolling standard deviation of returns
            # above the close price
            tp = 2 * rolling_std
            return tp
        elif backtest_params['tp_stop'] == 'atr':
            # Calculate the rolling average true range of the close price
            # over the window
            rolling_atr = vbt.ATR.run(high, low, close, window = window).atr

            # Calculate the take-profit price as close price plus 2 times
            # the rolling average true range
            tp = np.nan_to_num(close + 2 * rolling_atr, nan = 0.01)

            # The take-profit as a percentage of the close price
            tp = tp / close

            return tp
        else:
            raise ValueError('Invalid take-profit method')

    def calculate_size(self, open, high, low, close, volume, backtest_params, window):
        if type(backtest_params['size']) == float:
            size = np.full(len(close), backtest_params['size'])
            return size

        elif backtest_params['size'] == 'atr':
            # Calculate the rolling average true range of the close price
            # over the last week
            rolling_atr = vbt.ATR.run(high, low, close, window = window).atr

            # Normalize the average true range by dividing it by the close price
            rolling_atr = (rolling_atr / close) * 100

            # The more volatile the asset, the smaller the position size
            # and vice versa
            size = 1 / rolling_atr


            return size
        elif backtest_params['size'] == 'std':
            # Calculate the rolling standard deviation of the close price
            # over the last week
            rolling_std = pd.Series(close).pct_change().rolling(window).std().values * 100

            # Calculate the position size as the percentage of the account
            # that should be risked on each trade divided by the standard deviation
            size = 1 / rolling_std

            return size
        else:
            raise ValueError('Invalid position sizing method')
