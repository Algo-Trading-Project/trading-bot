import vectorbt as vbt
import numpy as np

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
class ARIMAStrat:
    
    indicator_factory_dict = {
        'class_name':'ARIMA',
        'short_name':'arima',
        'input_names':['open', 'high', 'low', 'close', 'volume'],
        'param_names':['p', 'd', 'q'],
        'output_names':['entries', 'exits']
    } 

    optimize_dict = {
        'p': [1],
        'd': [0, 1],
        'q': [1]
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
        
        y_pred = []
        y_true = []

        for i in range(len(close)):  
            if i < 24 * 7:
                entries.append(False)
                exits.append(False)
                continue

            end = i + 1

            model = ARIMA(
                close[:end], 
                order = (p, d, q), 
                enforce_stationarity = False, 
                enforce_invertibility = False,
            )

            fit_model = model.fit()
            pred = fit_model.forecast(steps = 1)[0]

            if end < len(close) - 1:
                y_true.append(close[i + 1])
                y_pred.append(pred)
            else:
                y_true.append(close[i])
                y_pred.append(pred)

            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            print('({}/{}) ARIMA(p = {}, d = {}, q = {}) rmse: {}'.format(i + 1, len(close), p, d, q, rmse), end = '\r', flush = True)

            entry_signal = pred > close[i]
            exit_signal = pred < close[i]

            entries.append(entry_signal)
            exits.append(exit_signal)
            
        return entries, exits
