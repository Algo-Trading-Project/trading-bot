import vectorbt as vbt
import numpy as np
import pandas as pd

from base_strategy import BaseStrategy

class MomentumVol(BaseStrategy):
    indicator_factory_dict = {
        'class_name':'MomentumVol',
        'short_name':'MomentumVol',
        'input_names':['open', 'high', 'low', 'close', 'volume'],
        'param_names':[
            'momentum_period', 'atr_window', 'volatility_ratio_window',
            'volume_window', 'volatility_threshold', 'high_volatility_threshold', 
            'time_exit_hours', 'cooldown_period'
        ],
        'output_names':['entries', 'exits']
    }

    optimize_dict = {
        'momentum_period': [12, 24],
        'atr_window': [12, 24],
        'volatility_ratio_window': [48, 72],
        'volume_window': [6, 12],
        'volatility_threshold': [0.5, 1],
        'high_volatility_threshold': [1.5, 2],
        'time_exit_hours': [48],
        'cooldown_period': [0, 24]
    }

    def indicator_func(open, high, low, close, volume,
                       momentum_period=24, atr_window=14, 
                       volatility_ratio_window=168, volume_window=24, 
                       volatility_threshold=0.5, high_volatility_threshold=1.5, 
                       time_exit_hours=24, cooldown_period=24):
        
        n = len(close)
        entries = np.zeros(n, dtype=bool)
        exits = np.zeros(n, dtype=bool)
        cooldown_counter = 0

        # Calculate momentum and ATR
        momentum = np.zeros(n)
        np.divide(close[1:] - close[:-1], close[:-1], out=momentum[1:])
        atr = vbt.ATR.run(high, low, close, window=atr_window).atr.to_numpy()

        # Calculate volatility ratio
        volatility_ratio = np.zeros(n)
        atr_rolling_mean = np.convolve(atr, np.ones(volatility_ratio_window) / volatility_ratio_window, mode='valid')
        np.divide(atr[volatility_ratio_window - 1:], atr_rolling_mean, out=volatility_ratio[volatility_ratio_window - 1:])

        # Calculate volume average
        volume_avg = np.convolve(volume, np.ones(volume_window) / volume_window, mode='valid')

        for i in range(1, n):
            # Entry logic
            if (momentum[i] > 0) and (momentum[i - 1] < momentum[i]) \
               and (volatility_ratio[i] <= volatility_threshold) \
               and (volume[i] > volume_avg[i - volume_window]) \
               and (cooldown_counter == 0):
                entries[i] = True
                cooldown_counter = cooldown_period
            else:
                cooldown_counter = max(0, cooldown_counter - 1)

            # Exit logic
            if (volatility_ratio[i] > high_volatility_threshold) or (momentum[i] < 0):
                exits[i] = True

        return entries, exits
