import vectorbt as vbt
import numpy as np

from statsmodels.tsa.arima.model import ARIMA

class ARIMAStrat:
    
    indicator_factory_dict = {
        'class_name':'ARIMA',
        'short_name':'arima',
        'input_names':['open', 'high', 'low', 'close', 'volume'],
        'param_names':['p', 'd', 'q'],
        'output_names':['entries', 'exits']
    } 

    optimize_dict = {
        'p': [0,1,2],
        'd': [0,1],
        'q': [0,1,2]
    }

    default_dict = {
        'p': 1,
        'd': 0,
        'q': 1
    }

    def indicator_func(open, high, low, close, volume,
                       p, d, q):
        entries = []
        exits = []
            
        for i in range(len(close)):  
            if i < 24 * 7:
                entries.append(False)
                exits.append(False)
                continue

            start = max(i - 24 * 30 * 2, 0)
            end = i + 1
            
            model = ARIMA(
                close[start: end], 
                order = (p, d, q), 
                enforce_stationarity = False, 
                enforce_invertibility = False,
            )

            fit_model = model.fit()
            pred = fit_model.forecast(steps = 1)

            entry_signal = pred > close[i]
            exit_signal = pred < close[i]

            entries.append(entry_signal)
            exits.append(exit_signal)

        return entries, exits
