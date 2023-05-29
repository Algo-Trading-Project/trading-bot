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
import multiprocessing as mp
import json
import vectorbt as vbt

#################################
#      TRADING STRATEGIES       #
#################################
from strategies.Strategy import Strategy
from strategies.indicator_funcs import indicator_func

class BackTester:

    def __init__(self, 
                 indicator_factory_dict,
                 indicator_func_defaults_dict,
                 optimization_metric = 'Total Return',
                 optimize_dict = {},
                 backtest_params = {'starting_cash':10_000, 'commission':0.01}
                 ):

        self.optimization_metric = optimization_metric
        self.optimize_dict = optimize_dict
        self.backtest_params = backtest_params

        self.indicator_factory_dict = indicator_factory_dict
        self.indicator_func_defaults_dict = indicator_func_defaults_dict   

    def serialize_json_data(obj):
        if isinstance(obj, pd.Timedelta):
            return obj.total_seconds()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        
        raise TypeError("Type not serializable")
     
    def __fetch_OHLCV_df_from_redshift(self, asset_id_base, asset_id_quote, exchange_id):
        # Connect to Redshift cluster containing price data
        with redshift_connector.connect(
            host = 'project-poseidon.cpsnf8brapsd.us-west-2.redshift.amazonaws.com',
            database = 'administrator',
            user = 'administrator',
            password = 'Free2play2'
        ) as conn:
            with conn.cursor() as cursor:
                # Query to fetch OHLCV data for an ETH pair & exchange of interest
                # within a specified date range
                if asset_id_base == 'ETH' and asset_id_quote == 'USD':
                    title = asset_id_base + '-' + asset_id_quote
                    query = """
                    SELECT 
                        time_period_end,
                        price_close AS "{}"
                    FROM token_price.eth.stg_price_data_1h
                    WHERE 
                        asset_id_base = '{}' AND
                        asset_id_quote = '{}' AND
                        exchange_id = '{}'
                    ORDER BY time_period_start ASC
                    """.format(title, asset_id_base, asset_id_quote, 
                               exchange_id)
                else:
                    title = asset_id_base + '-' + 'USD'
                    query = """
                    WITH requested_pair AS (
                        SELECT 
                            time_period_end,
                            price_close
                        FROM token_price.eth.stg_price_data_1h
                        WHERE
                            asset_id_base = '{}' AND
                            asset_id_quote = '{}' AND
                            exchange_id = '{}'
                    ), 
                    eth_usd_bitfinex AS (
                        SELECT
                            time_period_end,
                            price_close
                        FROM token_price.eth.stg_price_data_1h
                        WHERE
                            asset_id_base = 'ETH' AND
                            asset_id_quote = 'USD' AND
                            exchange_id = 'BITFINEX'
                    )

                    SELECT
                        e.time_period_end,
                        r.price_close / (1 / e.price_close) AS "{}"
                    FROM eth_usd_bitfinex e INNER JOIN requested_pair r
                        ON e.time_period_end = r.time_period_end
                    ORDER BY e.time_period_end
                    """.format(asset_id_base, asset_id_quote, exchange_id, title)

                # Execute query on Redshift and return result
                cursor.execute(query)
                tuples = cursor.fetchall()
                
                # Return queried data as a DataFrame
                df = pd.DataFrame(tuples, columns = ['Date', title]).set_index('Date')
                df = df.astype(float)

                return df

    def backtest(self, base, quote, exchange, strat, training_data, testing_data):
        in_sample_backtest = Strategy(
            price_data = training_data,
            indicator_factory_params = self.indicator_factory_dict.get(strat),
            indicator_func = strat,
            indicator_func_defaults = self.indicator_func_defaults_dict.get(strat),
            optimize_dict = self.optimize_dict.get(strat),
            optimization_metric = self.optimization_metric
        )   

        optimal_params, is_backtest_results = in_sample_backtest.optimize() 

        # Evaluate optimized trading strategy on unseen data

        out_of_sample_backtest = Strategy(
            price_data = testing_data,
            indicator_factory_params = self.indicator_factory_dict.get(strat),
            indicator_func = strat,
            indicator_func_defaults = self.indicator_func_defaults_dict.get(strat)
        )   

        oos_backtest_results = out_of_sample_backtest.backtest(optimal_params)

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
                (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                
                # Execute query on Redshift
                params = (self.indicator_factory_dict.get(strat)['class_name'], base, quote, exchange,
                          training_data.index[0], training_data.index[-1], 
                          json.dumps(is_backtest_results, default = BackTester.serialize_json_data), 
                          testing_data.index[0], testing_data.index[-1], 
                          json.dumps(oos_backtest_results, default = BackTester.serialize_json_data),
                          json.dumps(optimal_params, default = BackTester.serialize_json_data), self.optimization_metric)
                
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
                    FROM token_price.eth.stg_price_data_1h
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
                """

                # Execute query on Redshift and return result
                cursor.execute(query)
                tuples = cursor.fetchall()
                
                # Turn queried data into a DataFrame
                df = pd.DataFrame(tuples, columns = ['asset_id_base', 'asset_id_quote', 'exchange_id'])

        for i in range(len(df)):
            row = df.iloc[i]
            base, quote, exchange = row['asset_id_base'], row['asset_id_quote'], row['exchange_id']

            for indicator_func in self.optimize_dict.keys():
                print()
                print('Backtesting the {} strategy on {}'.format(
                    self.indicator_factory_dict.get(indicator_func)['class_name'],
                    base + '_' + 'USD' + '_' + exchange
                ))

                self.walk_forward_optimization(
                    base = base,
                    quote = quote, 
                    exchange = exchange,
                    strat = indicator_func,
                    in_sample_size = 24 * 30 * 4,
                    out_of_sample_size = 24 * 30 * 2 
                )

if __name__ == '__main__': 

    # This section contains one dictionary per strategy.  Each dictionary
    # includes all of the hyperparameters for a given strategy as keys and
    # a list of hyperparameter values to backtest as the values

    optimize_args_test_strat_vbt = {
        'fast_window':[3,6,12],
        'slow_window':[24,48,96]
    }

    # This section contains one dictionary.  The keys are the custom indicator functions
    # or strategies that will be backtested and the values are dictionaries with the following
    # keys:
    #       class_name : str - Name of the strategy
    #       short_name : str - Shorter name of the strategy
    #       input_names : list - List of inputs to the strategy
    #       param_names : list - List of hyperparameters of the strategy
    #       output_names : list - List of outputs of the strategy

    indicator_factory_dict = {
        indicator_func: {
         'class_name':'Test_Strat_VBT',
         'short_name':'test_strat',
         'input_names':['close'],
         'param_names':['fast_window', 'slow_window'],
         'output_names':['entries', 'exits']
         }
    }

    # This section contains one dictionary.  The keys are the custom indicator functions
    # or strategies that will be backtested and the values are dictionaries containing
    # default values for every hyperparameter of a given strategy

    indicator_func_defaults_dict = {
        indicator_func: {
            'fast_window':24,
            'slow_window':24 * 7
        }
    }

    b = BackTester(
        indicator_factory_dict = indicator_factory_dict,
        indicator_func_defaults_dict = indicator_func_defaults_dict,
        optimization_metric = 'Calmar Ratio',
        optimize_dict = {
            indicator_func: optimize_args_test_strat_vbt
        }
    )

    backtest_start = time.time()
    b.execute()
    backtest_end = time.time()

    print()
    print('time elapsed since backtest: {} mins'.format(round(abs(backtest_end - backtest_start) / 60.0, 2)))