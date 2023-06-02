import vectorbt as vbt
import numpy as np

class TestStratVBT:
    
    indicator_factory_dict = {
        'class_name':'Test_Strat_VBT',
        'short_name':'test_strat',
        'input_names':['close'],
        'param_names':['fast_window', 'slow_window'],
        'output_names':['entries', 'exits']
    }

    optimize_dict = {
        'fast_window': [3, 6, 12, 24],
        'slow_window': [48, 96, 24 * 7, 24 * 14, 24 * 30]
    }

    default_dict = {
        'fast_window':24,
        'slow_window':24 * 7
    }

    def indicator_func(close, fast_window, slow_window):
        sma_slow = []
        sma_fast = []

        entries = [False]
        exits = [False]

        for i in range(len(close)):
            slow_data_window = close[i - slow_window: i + 1]
            fast_data_window = close[i - fast_window: i + 1]

            if len(fast_data_window) - 1 < fast_window:
                sma_fast.append(np.nan)
            else:
                sma_fast.append(np.mean(fast_data_window))

            if len(slow_data_window) - 1 < slow_window:
                sma_slow.append(np.nan)
            else:
                sma_slow.append(np.mean(slow_data_window))

        for i in range(1, len(close)):
            if sma_fast[i - 1] < sma_slow[i - 1] and sma_fast[i] > sma_slow[i]:
                entries.append(True)
            else:
                entries.append(False)
            
            if sma_fast[i - 1] > sma_slow[i - 1] and sma_fast[i] < sma_slow[i]:
                exits.append(True)
            else:
                exits.append(False)

        return entries, exits