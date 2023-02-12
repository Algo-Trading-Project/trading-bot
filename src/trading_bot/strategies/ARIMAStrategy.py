from strategies.BaseStrategy import BaseStrategy
import requests as r
import json
import redshift_connector
import pandas as pd
import numpy as np

from statsmodels.tsa.arima.model import ARIMA
from math import ceil

class ARIMAStrategy(BaseStrategy):
    
    def __init__(self, strat_time_frame_minutes, sl, symbol_id):
        super().__init__(strat_time_frame_minutes = strat_time_frame_minutes, sl = sl, symbol_id = symbol_id)
        
        # self.historical_data = self.__fetch_price_data()

    def __fetch_price_data(self):
        with redshift_connector.connect(
            host = 'project-poseidon.cpsnf8brapsd.us-west-2.redshift.amazonaws.com',
            database = 'administrator',
            user = 'administrator',
            password = 'Free2play2'
        ) as conn:
            with conn.cursor() as cursor:
                # Query to fetch open price data for an ETH pair & exchange of interest
                # within a specified date range
                exchange, _, base, quote = self.symbol_id.split('_')
                query = """
                SELECT
                    time_period_start,
                    time_period_end,
                    price_close
                FROM token_price.eth.stg_price_data_1h
                WHERE
                    asset_id_base = '{}' AND
                    asset_id_quote = '{}' AND
                    exchange_id = '{}'
                ORDER BY time_period_start ASC
                """.format(base, quote, exchange)

                # Execute query on Redshift and return result
                cursor.execute(query)
                tuples = cursor.fetchall()

                df = pd.DataFrame(tuples, ['time_period_start', 'time_period_end', 'price_close'])

                # TODO: convert date in coinapi API request to ISO format before sending 
                latest_ds_for_pair_redshift = df.iloc[-1]['time_period_end']

                # TODO: get new coinapi API key
                api_request_url = 'https://rest.coinapi.io/v1/ohlcv/{}/history?period_id={}&time_start={}&limit={}'.format(self.symbol_id, '1HRS', latest_ds_for_pair_redshift, 100000)
                headers = {'X-CoinAPI-Key':'coin_api_api_key'}
        
                response = r.get(
                    url = api_request_url,
                    headers = headers,
                )

                response_json = response.json()

                # TODO: merge historical price data with recent price data into one Dataframe and return

    def process_tick(self, tick_data):
        trading_signal = super().process_tick(tick_data)

        print('trading signal super class: {}'.format(trading_signal))

        if trading_signal == 'Sell' or trading_signal == None:
            return trading_signal
        
        elif trading_signal == 'Generate':
            if len(self.data) >= 10:
                best_config = self.ARIMA_gridsearch()
                order = best_config['config']
                
                model = ARIMA(self.data['price_open'].values, order = order)
                model_fit = model.fit()
                forecast = model_fit.forecast()[0]

                print('tick data: ', tick_data)
                print('forecast next price: ', forecast)

    def ARIMA_gridsearch(self):
        ########################## HELPER ###################################################
        def evaluate_ARIMA_model(order):
            training_data_end_index = ceil(len(self.data) * 0.8)
            
            train = self.data.iloc[:training_data_end_index]['price_open'].values
            test = self.data.iloc[training_data_end_index:]['price_open'].values

            date_range_test = self.data.iloc[training_data_end_index:]['time_period_start'].astype(str)

            predictions = np.array([])
            
            for curr_index in range(len(test)):
                model = ARIMA(train, order = order)
                model_fit = model.fit()

                yhat = model_fit.forecast()[0]

                predictions = np.append(predictions, yhat)
                train = np.append(train, test[curr_index])

            residuals = test - predictions
            
            test_rmse = np.sqrt(np.mean(residuals**2))
            avg_residuals = np.average(residuals)
            std_residuals = np.std(residuals)
            
            OOS_accuracies = residuals / test * 100
            avg_OOS_accuracy = np.average(OOS_accuracies)
            std_OOS_accuracy = np.std(OOS_accuracies)

            return {'config': order, 'rmse': float(test_rmse), 'avg_residual': float(avg_residuals), 
                    'std_residual':float(std_residuals), 'avg_OOS_accuracy': float(avg_OOS_accuracy), 'std_OOS_accuracy': float(std_OOS_accuracy),
                    'date_range_test': list(date_range_test), 'test_predictions': list(predictions), 'test_data': list(test), 'residuals': list(residuals)}
                    
        ########################## HELPER ###################################################

        best_rmse, best_config = float('inf'), None

        for p in [1]:
            for d in [0, 1, 2]:
                for q in [0, 1, 2]:
                    try:
                        performance_metrics = evaluate_ARIMA_model(order = (p, d, q))

                        if performance_metrics['rmse'] < best_rmse:
                            best_rmse = performance_metrics['rmse']
                            best_config = performance_metrics
                    except:
                        continue

        return best_config




