from ..core.wfo.helper import calculate_tp, calculate_sl, calculate_size

import numpy as np

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

    def calculate_tp(self, open, high, low, close, volume, backtest_params, window):
        
        if type(backtest_params['tp_stop']) == float:
            tp = np.full(len(close), backtest_params['tp_stop'])
            return tp

        return np.abs(calculate_tp(
            open, 
            high, 
            low, 
            close,
            volume, 
            method = backtest_params['tp_stop'],
            window = window,
            backtest_params = backtest_params
        ) - 1)
    
    def calculate_sl(self, open, high, low, close, volume, backtest_params, window):
        
        if type(backtest_params['sl_stop']) == float:
            sl = np.full(len(close), backtest_params['sl_stop'])
            return sl

        return np.abs(calculate_sl(
            open, 
            high, 
            low, 
            close, 
            volume, 
            method = backtest_params['sl_stop'],
            window = window,
            backtest_params = backtest_params
        ) - 1)

    def calculate_size(self, open, high, low, close, volume, backtest_params):
        
        if type(backtest_params['size']) == float:
            size = np.full(len(close), backtest_params['size'])
            return size

        return calculate_size(
            open, 
            high, 
            low, 
            close, 
            volume, 
            backtest_params
        )
    
        