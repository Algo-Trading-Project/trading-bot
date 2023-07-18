import numpy as np
import vectorbt as vbt

class TestStratVBT:
    
    indicator_factory_dict = {
        'class_name':'Test_Strat_VBT',
        'short_name':'test_strat',
        'input_names':['open', 'high', 'low', 'close', 'volume'],
        'param_names':['fast_window', 'slow_window'],
        'output_names':['entries', 'exits']
    }

    optimize_dict = {
        'fast_window': [6, 12, 24],
        'slow_window': [24 * 7, 24 * 14]
    }

    default_dict = {
        'fast_window':24,
        'slow_window':24 * 7
    }

    def indicator_func(open, high, low, close, volume, fast_window, slow_window):        
        sma_slow = vbt.MA.run(close, window = slow_window, ewm = True)       
        sma_fast = vbt.MA.run(close, window = fast_window, ewm = True)      
        
        entries = sma_fast.ma_crossed_above(sma_slow)
        exits = sma_fast.ma_crossed_below(sma_slow)

        return entries, exits
