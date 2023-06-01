import vectorbt as vbt
import numpy as np

from statsmodels.tsa.arima.model import ARIMA

class ARIMAStrat:
    
    indicator_factory_dict = {
        'class_name':'ARIMA',
        'short_name':'arima',
        'input_names':['close'],
        'param_names':['p', 'd', 'q', 'ema_window', 'forecast_window'],
        'output_names':['entries', 'exits']
    }

    optimize_dict = {
        'p': [2],
        'd': [1],
        'q': [2],
        'ema_window': [24],
        'forecast_window': [3]
    }

    default_dict = {
        'p': 1,
        'd': 0,
        'q': 1,
        'ema_window': 24,
        'forecast_window':1
    }

    def indicator_func(close, p, d, q, ema_window, forecast_window):
        entries = []
        exits = []
        
        ema = vbt.MA.run(
            close = close, 
            window = ema_window,
            ewm = True
        ).ma.values
    
        for i in range(len(close)):
            print('(p = {}, d = {}, q = {}, ema = {}, forecast_window = {}) {} / {}'.format(p, d, q, ema_window, forecast_window, i + 1, len(close)))
            
            if i < 24 * 7:
                entries.append(False)
                exits.append(False)
                continue
            
            model = ARIMA(
                close[i - 24 * 7: i + 1], 
                order = (p, d, q), 
                enforce_stationarity = False, 
                enforce_invertibility = False,
            )

            fit_model = model.fit()
            
            pred = fit_model.get_prediction(start = 0, end = forecast_window)
            conf_int = pred.conf_int(alpha = 0.01)

            lower_bound = conf_int[-1][0]
            upper_bound = conf_int[-1][-1]

            entry_signal = (
                (close[i - 1][0] < ema[i - 1][0]) and
                (close[i][0] > ema[i][0]) and
                ((upper_bound - close[i][0]) / close[i][0] >= .005)
            )

            exit_signal = (
                (close[i - 1][0] > ema[i - 1][0]) and
                (close[i][0] < ema[i][0])
            )

            entries.append(entry_signal)
            exits.append(exit_signal)

        return entries, exits

print(.000000000000000000000000001)