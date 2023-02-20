from strategies.BaseStrategy import BaseStrategy
import requests as r
import json
import redshift_connector
import pandas as pd
import numpy as np
import arch

# import warnings
# warnings.filterwarnings("ignore")

from statsmodels.tsa.arima.model import ARIMA
from math import ceil

class ARIMAStrategy(BaseStrategy):
    
    def __init__(self, strat_time_frame_minutes, sl, symbol_id):
        super().__init__(strat_time_frame_minutes = strat_time_frame_minutes, sl = sl, symbol_id = symbol_id)
        
        self.preds = np.array([])
        self.actual = np.array([])

    def process_tick(self, tick_data):
        trading_signal = super().process_tick(tick_data)

        if trading_signal == 'Sell' or trading_signal == 'None':
            return trading_signal
        
        elif trading_signal == 'Generate':
            if len(self.data) >= 10:
                best_config = self.ARIMA_gridsearch()
                order = best_config['config']
                
                model = ARIMA(self.data['price_open'].astype(float).values, order = order)
                model_fit = model.fit()
                forecast = model_fit.forecast()[0]

                

                print('curr price: {}'.format(self.data.iloc[-1]['price_open']))
                print('forecast: {}'.format(forecast))

                self.preds = np.append(self.preds, forecast)
                self.actual = np.append(self.actual, tick_data['price_open'])

    def ARIMA_gridsearch(self):
        ########################## HELPER ###################################################
        def evaluate_ARIMA_model(order):
            training_data_end_index = ceil(len(self.data) * 0.8)
            
            train = self.data.iloc[:training_data_end_index]['price_open'].astype(float).values
            test = self.data.iloc[training_data_end_index:]['price_open'].astype(float).values
            predictions = np.array([])
            
            for curr_index in range(len(test)):
                model = ARIMA(train, order = order)
                model_fit = model.fit()
                yhat = model_fit.forecast()[0]
                
                predictions = np.append(predictions, yhat)
                train = np.append(train, test[curr_index])

            residuals = test - predictions
            
            test_rmse = np.sqrt(np.mean(residuals**2))

            return {'config': order, 'rmse': float(test_rmse)}
                    
        ########################## HELPER ###################################################

        best_rmse, best_config = float('inf'), None

        for p in [1, 2, 3, 4, 5]:
            for d in [0, 1, 2]:
                for q in [1, 2, 3, 4, 5]:
                    try:
                        performance_metrics = evaluate_ARIMA_model(order = (p, d, q))

                        if performance_metrics['rmse'] < best_rmse:
                            best_rmse = performance_metrics['rmse']
                            best_config = performance_metrics
                    except Exception as e:
                        continue

        return best_config




