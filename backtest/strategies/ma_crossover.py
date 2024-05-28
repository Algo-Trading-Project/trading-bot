import vectorbt as vbt
import numpy as np

from .base_strategy import BaseStrategy

class MACrossOver(BaseStrategy):
    
    indicator_factory_dict = {
        'class_name':'MACrossOver',
        'short_name':'ma_crossover',
        'input_names':['open', 'high', 'low', 'close', 'volume'],
        'param_names':['fast_window', 'slow_window'],
        'output_names':['entries', 'exits', 'tp', 'sl', 'size']
    }

    optimize_dict = {
        'fast_window': [15, 30, 60, 60 * 12, 60 * 24],
        'slow_window': [60 * 24, 60 * 24 * 7, 60 * 24 * 30]
    }

    backtest_params = {
        'init_cash': 10_000,
        'fees': 0.00295,
        'sl_stop': np.inf,
        'sl_trail': True,
        'tp_stop': np.inf,
        'size': 'atr',
        'size_type': 2
    }

    def __init__(self):
        pass

    @staticmethod
    def indicator_func(open, high, low, close, volume,
                       fast_window, slow_window): 

        backtest_params = MACrossOver.backtest_params
        
        tp = MACrossOver.calculate_tp(open, high, low, close, volume, backtest_params, window = 60 * 24)
        sl = MACrossOver.calculate_sl(open, high, low, close, volume, backtest_params, window = 60 * 24)
        size = MACrossOver.calculate_size(open, high, low, close, volume, backtest_params)
        
        sma_slow = vbt.MA.run(close, window = slow_window, ewm = True)       
        sma_fast = vbt.MA.run(close, window = fast_window, ewm = True)      
        
        entries = sma_fast.ma_crossed_above(sma_slow).values
        exits = sma_fast.ma_crossed_below(sma_slow).values

        return entries, exits, tp, sl, size