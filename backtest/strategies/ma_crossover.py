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
        'fast_window': [3, 6, 12, 24, 24 * 2],
        'slow_window': [24 * 3, 24 * 4, 24 * 5, 24 * 6, 24 * 7]
    }

    backtest_params = {
        'init_cash': 100_000,
        'fees': 0.00295,
        'sl_stop': np.inf,
        'sl_trail': True,
        'tp_stop': np.inf,
        'size': 0.05,
        'size_type': 2
    }

    def __init__(self):
        pass

    @staticmethod
    def indicator_func(open, high, low, close, volume,
                       fast_window, slow_window): 

        backtest_params = MACrossOver.backtest_params
        
        tp = MACrossOver.calculate_tp(open, high, low, close, volume, backtest_params)
        sl = MACrossOver.calculate_sl(open, high, low, close, volume, backtest_params)
        size = MACrossOver.calculate_size(open, high, low, close, volume, backtest_params)
        
        sma_slow = vbt.MA.run(close, window = slow_window, ewm = True)       
        sma_fast = vbt.MA.run(close, window = fast_window, ewm = True)      
        
        entries = sma_fast.ma_crossed_above(sma_slow).values
        exits = sma_fast.ma_crossed_below(sma_slow).values

        return entries, exits, tp, sl, size