#################################
# ADDING PACKAGE TO PYTHON PATH #
#################################
import sys
import os
sys.path.append(os.getcwd())

#################################
#             MISC              #
#################################
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', UserWarning)

import time
import redshift_connector
import pandas as pd
import numpy as np
import json

###############################################
#      TRADING STRATEGIES / BACKTESTING       #
###############################################
from core.Backtest import Backtest

from strategies.TestStratVBT import TestStratVBT
from strategies.ARIMA import ARIMAStrat

class BackTester:

    def __init__(self, 
                 strategies,
                 optimization_metric = 'Sortino Ratio',
                 backtest_params = {'init_cash':10_000, 'fees':0.01}
                 ):

        self.optimization_metric = optimization_metric
        self.backtest_params = backtest_params
        self.strategies = strategies

    def serialize_json_data(obj):
        if isinstance(obj, pd.Timedelta):
            return obj.total_seconds()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
             
    def __fetch_OHLCV_df_from_redshift(self, asset_id_base, asset_id_quote, exchange_id):
        # Connect to Redshift cluster containing price data
        with redshift_connector.connect(
            host = 'project-poseidon.cpsnf8brapsd.us-west-2.redshift.amazonaws.com',
            database = 'token_price',
            user = 'administrator',
            password = 'Free2play2'
        ) as conn:
            with conn.cursor() as cursor:
                # Query to fetch OHLCV data for a token & exchange of interest
                title = asset_id_base + '-' + asset_id_quote
                query = """
                SELECT 
                    time_period_end,
                    price_open,
                    price_high,
                    price_low,
                    price_close,
                    volume_traded
                FROM token_price.coinapi.price_data_1h
                WHERE 
                    asset_id_base = '{}' AND
                    asset_id_quote = '{}' AND
                    exchange_id = '{}'
                ORDER BY time_period_start ASC
                """.format(asset_id_base, asset_id_quote, 
                            exchange_id)

                # Execute query on Redshift and return result
                cursor.execute(query)
                tuples = cursor.fetchall()
                
                # Return queried data as a DataFrame
                cols = ['Date', 'price_open', 'price_high', 'price_low', 'price_close', 'volume_traded']
                df = pd.DataFrame(tuples, columns = cols).set_index('Date')
                df = df.astype(float)

                return df

    def backtest(self, base, quote, exchange, strat, training_data, testing_data):
        is_backtest = Backtest(
            strategy = strat,
            price_data = training_data,
            optimization_metric = self.optimization_metric,
            backtest_params = self.backtest_params
        )   

        optimal_params, is_backtest_results = is_backtest.optimize() 

        # Evaluate optimized trading strategy on unseen data

        oos_backtest = Backtest(
            strategy = strat,
            price_data = testing_data,
            optimization_metric = self.optimization_metric,
            backtest_params = self.backtest_params
        )   

        oos_backtest_results = oos_backtest.backtest(optimal_params)

        # Delete unwanted metrics from in-sample and oos
        # backtest results before uploading to Redshift
        for del_col in ['Start', 'End', 'Period', 'Total Fees Paid', 
                        'Total Closed Trades', 'Total Open Trades', 'Open Trade PnL']:
            try:
                del is_backtest_results[del_col]
            except:
                pass

            try:
                del oos_backtest_results[del_col]
            except:
                pass

        # Upload backtest
        with redshift_connector.connect(
            host = 'project-poseidon.cpsnf8brapsd.us-west-2.redshift.amazonaws.com',
            database = 'trading_bot',
            user = 'administrator',
            password = 'Free2play2'
        ) as conn:
            with conn.cursor() as cursor:
                # Query to insert backtest results into Redshift 
                query = """
                INSERT INTO trading_bot.eth.backtest_results VALUES 
                (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                str_is_backtest_results = json.dumps(is_backtest_results, default = BackTester.serialize_json_data)
                str_is_backtest_results = str_is_backtest_results.replace('NaN', 'null').replace('Infinity', 'null')

                str_oos_backtest_results = json.dumps(oos_backtest_results, default = BackTester.serialize_json_data)
                str_oos_backtest_results = str_oos_backtest_results.replace('NaN', 'null').replace('Infinity', 'null')

                # Execute query on Redshift
                params = (strat.indicator_factory_dict['class_name'], base, quote, exchange,
                          training_data.index[0], training_data.index[-1], 
                          str_is_backtest_results, 
                          testing_data.index[0], testing_data.index[-1], 
                          str_oos_backtest_results,
                          json.dumps(optimal_params, default = BackTester.serialize_json_data), self.optimization_metric,
                          json.dumps(self.backtest_params, default = BackTester.serialize_json_data))
                
                cursor.execute(query, params)
                cursor.close()

            conn.commit()
    
    def walk_forward_optimization(self, base, quote, exchange, strat,
                                  in_sample_size, out_of_sample_size):
        
        # Walk-forward analysis
        start = 0

        # Fetch revelant price data for backtest
        backtest_data = self.__fetch_OHLCV_df_from_redshift(
            asset_id_base = base,
            asset_id_quote = quote,
            exchange_id = exchange
        )

        while start + in_sample_size + out_of_sample_size <= len(backtest_data):
            print()
            print('Progress: {} / {} days...'.format(int(start / 24), int(len(backtest_data) / 24)))
            print()

            if len(backtest_data.iloc[start:]) - (in_sample_size + out_of_sample_size) < out_of_sample_size:
                in_sample_data = backtest_data.iloc[start:start + in_sample_size]
                out_of_sample_data = backtest_data.iloc[start + in_sample_size:]
            else:
                in_sample_data = backtest_data.iloc[start:start + in_sample_size]
                out_of_sample_data = backtest_data.iloc[start + in_sample_size:start + in_sample_size + out_of_sample_size]
                        
            self.backtest(
                base = base,
                quote = quote,
                exchange = exchange,
                strat = strat,
                training_data = in_sample_data,
                testing_data = out_of_sample_data
            )
            
            start += out_of_sample_size
                                    
    def execute(self):
        # Run a walk-forward optimization on each pair/strategy combination

        with redshift_connector.connect(
            host = 'project-poseidon.cpsnf8brapsd.us-west-2.redshift.amazonaws.com',
            database = 'token_price',
            user = 'administrator',
            password = 'Free2play2'
        ) as conn:
            with conn.cursor() as cursor:
                # Get all unique pairs in Redshift w/ atleast 1 year's worth of
                # hourly price data
                query = """
                WITH num_days_data AS (
                    SELECT 
                        asset_id_base,
                        asset_id_quote,
                        exchange_id,
                        COUNT(*) / 24.0 AS num_days_data
                    FROM token_price.coinapi.price_data_1h
                    GROUP BY asset_id_base, asset_id_quote, exchange_id
                ), 
                ordered_pairs_by_data_size AS (
                    SELECT
                        *,
                        ROW_NUMBER() OVER (PARTITION BY asset_id_base, asset_id_quote 
                                           ORDER BY num_days_data DESC) AS pos
                    FROM num_days_data
                )

                SELECT 
                    asset_id_base,
                    asset_id_quote,
                    exchange_id
                FROM ordered_pairs_by_data_size
                WHERE 
                    pos = 1 AND
                    num_days_data >= 365
                ORDER BY asset_id_base, asset_id_quote, exchange_id
                """

                # Execute query on Redshift and return result
                cursor.execute(query)
                tuples = cursor.fetchall()
                
                # Turn queried data into a DataFrame
                df = pd.DataFrame(tuples, columns = ['asset_id_base', 'asset_id_quote', 'exchange_id'])
                
        for i in range(len(df)):
            row = df.iloc[i]
            base, quote, exchange = row['asset_id_base'], row['asset_id_quote'], row['exchange_id']

            for strat in self.strategies:
                print()
                print('Backtesting the {} strategy on {} ({} / {})'.format(
                   strat.indicator_factory_dict['class_name'],
                    base + '_' + 'USD' + '_' + exchange,
                    i + 1, len(df)
                ))

                self.walk_forward_optimization(
                    base = base,
                    quote = quote, 
                    exchange = exchange,
                    strat = strat,
                    in_sample_size = 24 * 30 * 4,
                    out_of_sample_size = 24 * 30 * 4 
                )

if __name__ == '__main__': 
    backtest_params = {'init_cash': 10_000, 'fees': 0.01, 'size':1}

    b = BackTester(
        strategies = [TestStratVBT],
        optimization_metric = 'Sortino Ratio',
        backtest_params = backtest_params
    )

    backtest_start = time.time()
    b.execute()
    backtest_end = time.time()

    print()
    print('Total Time Elapsed: {} mins'.format(round(abs(backtest_end - backtest_start) / 60.0, 2)))