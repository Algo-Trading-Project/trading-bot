import vectorbt as vbt

class TestStratVBT:
    
    indicator_factory_dict = {
        'class_name':'Test_Strat_VBT',
        'short_name':'test_strat',
        'input_names':['close'],
        'param_names':['fast_window', 'slow_window'],
        'output_names':['entries', 'exits']
    }

    optimize_dict = {
        'fast_window': [3, 6, 12],
        'slow_window': [24, 48, 96]
    }

    default_dict = {
        'fast_window':24,
        'slow_window':24 * 7
    }

    def indicator_func(close, fast_window, slow_window):
        fast_ma = vbt.MA.run(close, window = fast_window)
        slow_ma = vbt.MA.run(close, window = slow_window)

        entries = fast_ma.ma_crossed_above(slow_ma)
        exits = fast_ma.ma_crossed_below(slow_ma)

        return entries, exits