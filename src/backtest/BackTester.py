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
from math import ceil

#################################
#      TRADING STRATEGIES       #
#################################
from backtesting import Backtest
from src.backtest.strategies.ARIMA import ARIMAStrategy

class BackTester:

    def __init__(self, 
                 symbol_ids,
                 start_date,
                 end_date,
                 strategies,
                 optimize_dict = {},
                 backtest_params = {'starting_cash':10_000, 'commission':0.01},
                 ):

        self.symbol_ids = symbol_ids

        self.start_date = start_date
        self.end_date = end_date

        self.strategies = strategies
        self.optimize_dict = optimize_dict
        self.backtest_params = backtest_params

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
                    query = """
                    SELECT 
                        time_period_end,
                        price_open,
                        price_high,
                        price_low,
                        price_close,
                        volume_traded
                    FROM token_price.eth.stg_price_data_1h
                    WHERE 
                        asset_id_base = '{}' AND
                        asset_id_quote = '{}' AND
                        exchange_id = '{}' AND
                        time_period_start >= '{}' AND
                        time_period_end <= '{}'
                    ORDER BY time_period_start ASC
                    """.format(asset_id_base, asset_id_quote, exchange_id, self.start_date, self.end_date)
                else:
                    query = """
                    WITH requested_pair AS (
                        SELECT 
                            time_period_end,
                            price_open,
                            price_high,
                            price_low,
                            price_close,
                            volume_traded
                        FROM token_price.eth.stg_price_data_1h
                        WHERE
                            asset_id_base = '{}' AND
                            asset_id_quote = '{}' AND
                            exchange_id = '{}'
                        ORDER BY time_period_start
                    ), 
                    eth_usd_coinbase AS (
                        SELECT
                            time_period_start,
                            price_open,
                            price_close
                        FROM token_price.eth.stg_price_data_1h
                        WHERE
                            asset_id_base = 'ETH' AND
                            asset_id_quote = 'USD' AND
                            exchange_id = 'COINBASE' AND
                            time_period_start >= '{}' AND
                            time_period_end <= '{}'
                        ORDER BY time_period_start
                    )

                    SELECT
                        e.time_period_start,
                        r.price_open / (1 / e.price_open) AS price_open,
                        r.price_close / (1 / e.price_close) AS price_high,
                        r.price_close / (1 / e.price_close) AS price_low,
                        r.price_close / (1 / e.price_close) AS price_close,
                        r.volume_traded
                    FROM eth_usd_coinbase e INNER JOIN requested_pair r
                        ON e.time_period_start = r.time_period_start
                    ORDER BY e.time_period_start
                    """.format(asset_id_base, asset_id_quote, exchange_id, self.start_date, self.end_date)

                # Execute query on Redshift and return result
                cursor.execute(query)
                tuples = cursor.fetchall()
                
                # Return queried data as a DataFrame
                df = pd.DataFrame(tuples, columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']).set_index('Time')
                df = df.astype(float)

                return df

    def backtest(self, base, quote, exchange, strat, training_data, testing_data):
        in_sample_backtest = Backtest(
            data = training_data, 
            strategy = strat, 
            cash = self.backtest_params['starting_cash'], 
            commission = self.backtest_params['commission'],
            trade_on_close = True
        )

        in_sample_backtest_results = in_sample_backtest.optimize(
            **self.optimize_dict.get(strat),
            maximize = 'Sortino Ratio',
            )
        
        # Evaluate optimized trading strategy on unseen data
        optimal_strat_params_str = str(in_sample_backtest_results['_strategy'])
        
        optimal_strat_params_list = optimal_strat_params_str.replace('Strategy', '').strip(strat.name()).strip('()').split(',')
        optimal_strat_params = {}

        for param in optimal_strat_params_list:
            key, val = param.split('=')
            val = float(val)

            optimal_strat_params[key] = val
        
        strat.update_hyperparameters(optimal_strat_params)

        out_of_sample_backtest = Backtest(
            data = testing_data, 
            strategy = strat, 
            cash = self.backtest_params['starting_cash'], 
            commission = self.backtest_params['commission'],
            trade_on_close = True
        )

        out_of_sample_backtest_results = out_of_sample_backtest.run()

        # Delete unwanted metrics from in-sample and oos
        # backtest results before uploading to Redshift
        for del_col in ['Start', 'End', 'Duration', 'Max. Drawdown Duration',
                        'Avg. Drawdown Duration', 'Max. Trade Duration',
                        'Avg. Trade Duration', '_strategy', '_equity_curve', '_trades']:
            try:
                del in_sample_backtest_results[del_col]
            except:
                pass

            try:
                del out_of_sample_backtest_results[del_col]
            except:
                pass

        is_res_str = json.dumps(dict(in_sample_backtest_results))
        oos_res_str = json.dumps(dict(out_of_sample_backtest_results))

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
                (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                
                # Execute query on Redshift
                params = (base, quote, exchange, strat.name(), training_data.index[0],
                          training_data.index[-1], is_res_str, testing_data.index[0],
                          testing_data.index[-1], oos_res_str, json.dumps(optimal_strat_params))
                
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
            
            if len(backtest_data.iloc[start:]) - (in_sample_size + out_of_sample_size) < out_of_sample_size:
                in_sample_data = backtest_data.iloc[start:start + in_sample_size].copy()
                out_of_sample_data = backtest_data.iloc[start + in_sample_size:].copy()
            else:
                in_sample_data = backtest_data.iloc[start:start + in_sample_size].copy()
                out_of_sample_data = backtest_data.iloc[start + in_sample_size:start + in_sample_size + out_of_sample_size].copy()
                        
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
        for symbol_id in self.symbol_ids:
            base, quote, exchange = symbol_id.split('_')
            
            print()
            print('Now backtesting {}'.format(symbol_id))
            print()
            
            for strat in self.strategies:
                self.walk_forward_optimization(
                    base = base,
                    quote = quote, 
                    exchange = exchange,
                    strat = strat,
                    in_sample_size = 24 * 30 * 3,
                    out_of_sample_size = 24 * 30 
                )

if __name__ == '__main__': 
    # mp.set_start_method('fork')

    symbol_ids = [
        'ETH_USD_COINBASE'
    ]

    optimize_args_arima = {
        'p':[1],
        'd':[0],
        'q':[1],
        'ema_window':[6]
    }

    b = BackTester(
        start_date = '2022/07/01',
        end_date = '2022/09/01',
        symbol_ids = symbol_ids,
        strategies = [ARIMAStrategy],
        optimize_dict = {ARIMAStrategy: optimize_args_arima}
    )

    backtest_start = time.time()

    b.execute()

    backtest_end = time.time()

    print()
    print('time elapsed since backtest: {} mins'.format(round(abs(backtest_end - backtest_start) / 60.0, 2)))