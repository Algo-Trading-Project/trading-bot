import vectorbt as vbt
import numpy as np

from statsmodels.tsa.arima.model import ARIMA

class ARIMAStrat:
    
    indicator_factory_dict = {
        'class_name':'ARIMA',
        'short_name':'arima',
        'input_names':['close'],
        'param_names':['p', 'd', 'q', 'ema1_window', 'ema2_window', 'forecast_window'],
        'output_names':['entries', 'exits']
    } 

    optimize_dict = {
        'p': [0,1,2],
        'd': [0,1],
        'q': [0,1,2],
        'ema1_window': [3,7,9,12,15],
        'ema2_window':[21,26,33,50,100],
        'forecast_window': [1,2,3,4,5]
    }

    default_dict = {
        'p': 1,
        'd': 0,
        'q': 1,
        'ema1_window': 9,
        'ema2_window': 26,
        'forecast_window':1
    }

    def indicator_func(close, p, d, q, ema1_window, ema2_window, forecast_window):
        entries = []
        exits = []
        
        ema1 = vbt.MA.run(
            close = close, 
            window = ema1_window,
            ewm = True
        ).ma.values

        ema2 = vbt.MA.run(
            close = close, 
            window = ema2_window,
            ewm = True
        ).ma.values
    
        for i in range(len(close)):            
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
                ((close[i][0] < ema2[i][0]) or (ema1[i][0] > ema2[i][0])) and
                (((upper_bound - close[i][0]) / close[i][0] >= .005) or (upper_bound - ema2[i][0]) > 0)
            )

            exit_signal = (
                ((close[i][0] > ema2[i][0]) or (ema1[i][0] < ema2[i][0])) and 
                ((close[i][0] - lower_bound) / close[i][0] >= .005) or (upper_bound - ema2[i-1][0]) > 0 
            )

            entries.append(entry_signal)
            exits.append(exit_signal)

        return entries, exits
