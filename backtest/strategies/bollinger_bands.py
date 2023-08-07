import vectorbt as vbt
import numpy as np

class BollingerBands:
    
    indicator_factory_dict = {
        'class_name':'BollingerBands',
        'short_name':'bolingerbands',
        'input_names':['open', 'high', 'low', 'close', 'volume'],
        'param_names':['bb_window', 'bb_std_mult'],
        'output_names':['entries', 'exits']
    }

    optimize_dict = {
        'bb_window': [6, 12, 24, 24 * 7, 24 * 30],
        'bb_std_mult': [1, 1.5, 2, 2.5, 3]
    }

    default_dict = {
        'bb_window': 24,
        'bb_std_mult': 2,
    }

    def indicator_func(open, high, low, close, volume,
                       bb_window, bb_std_mult):  
        
        bb = vbt.BBANDS.run(close, window = bb_window, ewm = True, alpha = bb_std_mult)      

        # Entries
        close_crossed_below_lower_bb = bb.close_crossed_below(bb.lower).values
        entries = close_crossed_below_lower_bb

        # Exits
        exits = bb.close_crossed_above(bb.middle).values

        return entries, exits
