import vectorbt as vbt
import numpy as np

class BollingerBands:
    
    indicator_factory_dict = {
        'class_name':'BollingerBands',
        'short_name':'bolingerbands',
        'input_names':['open', 'high', 'low', 'close', 'volume'],
        'param_names':['bb_window', 'bb_std_mult', 'rsi_window', 'rsi_lower_thresh'],
        'output_names':['entries', 'exits']
    }

    optimize_dict = {
        'bb_window': [6, 12, 24],
        'bb_std_mult': [1, 2, 3],
        'rsi_window': [6, 12, 24],
        'rsi_lower_thresh': [10, 20, 30, 40, 50]
    }

    def indicator_func(open, high, low, close, volume,
                       bb_window, bb_std_mult,
                       rsi_window, rsi_lower_thresh):  
        
        bb = vbt.BBANDS.run(close, window = bb_window, ewm = True, alpha = bb_std_mult)      
        rsi = vbt.RSI.run(close, window = rsi_window, ewm = True)

        # Entries
        close_crossed_below_lower_bb = bb.close_crossed_below(bb.lower).values
        rsi_below_thresh = rsi.rsi_below(rsi_lower_thresh)
        entries = np.logical_and(close_crossed_below_lower_bb, rsi_below_thresh)

        # Exits
        exits = bb.close_crossed_above(bb.middle).values

        return entries, exits
