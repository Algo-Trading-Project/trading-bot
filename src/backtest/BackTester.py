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
from math import ceil

#################################
#      TRADING STRATEGIES       #
#################################
from backtesting import Backtest
from src.backtest.strategies.IchimokuCloud import IchimokuCloud
from src.backtest.strategies.ARIMA import ARIMAStrategy
from src.backtest.strategies.MovingAverage import MovingAverage

class BackTester:

    def __init__(self, 
                 symbol_ids,
                 start_date,
                 end_date,
                 strategies,
                 optimize = False,
                 optimize_dict = {},
                 backtest_params = {'starting_cash':10_000, 'commission':0.0025},
                 pct_training_data = 0.7):

        self.symbol_ids = symbol_ids

        self.start_date = start_date
        self.end_date = end_date

        self.strategies = strategies
        self.optimize = optimize
        self.optimize_dict = optimize_dict
        self.backtest_params = backtest_params

        self.pct_training_data = pct_training_data

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
                query = """
                SELECT 
                    time_period_start,
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

                # Execute query on Redshift and return result
                cursor.execute(query)
                tuples = cursor.fetchall()
                
                # Return queried data as a DataFrame
                df = pd.DataFrame(tuples, columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']).set_index('Time')
                df = df.astype(float)

                training_data_end_index = ceil(len(df) * self.pct_training_data)
                training_data, testing_data = df.iloc[:training_data_end_index], df.iloc[training_data_end_index:]

                return training_data, testing_data

    def evaluate_strategies(self):
        backtest_result_cols = [
            'asset_id_base', 'asset_id_quote', 'exchange_id', 'strategy_name', '_strategy',
            'Start', 'End', 'Duration', 'Equity Final [$]', 
            'Return [%]', 'Buy & Hold Return [%]', 'Sharpe Ratio', 'Sortino Ratio', 
            'Calmar Ratio', 'Max. Drawdown [%]', 'Avg. Drawdown [%]', 'Max. Drawdown Duration',
            'Avg. Drawdown Duration', '# Trades', 'Win Rate [%]', 'Avg. Trade [%]', 
            'Best Trade [%]', 'Worst Trade [%]', 'Max. Trade Duration', 'Avg. Trade Duration' 
        ]

        backtest_results = pd.DataFrame(columns = backtest_result_cols)

        for symbol_id in self.symbol_ids:
            split_symbol_id = symbol_id.split('_')
            asset_id_base, asset_id_quote, exchange_id = split_symbol_id[0], split_symbol_id[1], split_symbol_id[2]
            
            training_data, testing_data = self.__fetch_OHLCV_df_from_redshift(asset_id_base, asset_id_quote, exchange_id)

            print()
            print('training data range: {} - {}'.format(training_data.index[0], training_data.index[-1]))
            print('testing data range: {} - {}'.format(testing_data.index[0], testing_data.index[-1]))
            print()

            for strat in self.strategies:
                print()
                print('Now backtesting the {}/{} pair on {} using the {} strategy'.format(asset_id_base, asset_id_quote, exchange_id, strat.name()))
                print()

                in_sample_backtest = Backtest(
                    data = training_data, 
                    strategy = strat, 
                    cash = self.backtest_params['starting_cash'], 
                    commission = self.backtest_params['commission'],
                    trade_on_close = True
                )

                if self.optimize and isinstance(self.optimize_dict.get(strat), dict):
                    stats_for_strat = in_sample_backtest.optimize(
                        **self.optimize_dict.get(strat),
                        maximize = 'Equity Final [$]',
                    )
                else:
                    stats_for_strat = in_sample_backtest.run()
                
                # Evaluate optimized trading strategy on unseen data
                optimal_strat_params_str = str(stats_for_strat['_strategy'])
                
                optimal_strat_params_list = optimal_strat_params_str.strip(strat.name()).strip('()').split(',')
                optimal_strat_params = {}

                for param in optimal_strat_params_list:
                    key, val = param.split('=')
                    val = int(val)

                    optimal_strat_params[key] = val
                
                strat.update_hyperparameters(optimal_strat_params)

                out_of_sample_backtest = Backtest(
                    data = testing_data, 
                    strategy = strat, 
                    cash = self.backtest_params['starting_cash'], 
                    commission = self.backtest_params['commission'],
                    trade_on_close = True
                )

                stats_for_strat = out_of_sample_backtest.run()

                # Visualize results of trading strategy on unseen data
                out_of_sample_backtest.plot(
                    results = stats_for_strat, 
                    open_browser = False,
                    filename = 'src/backtest/{}_{}.html'.format(symbol_id, strat.name())
                )

                stats_for_strat['asset_id_base'] = asset_id_base
                stats_for_strat['asset_id_quote'] = asset_id_quote
                stats_for_strat['exchange_id'] = exchange_id
                stats_for_strat['strategy_name'] = optimal_strat_params_str
                stats_for_strat_df = stats_for_strat[backtest_result_cols].to_frame().T
                backtest_results = backtest_results.append(stats_for_strat_df, ignore_index = True)

                backtest_results.to_csv('src/backtest/backtest_results.csv')
        
        return backtest_results

if __name__ == '__main__': 
    # mp.set_start_method('fork')

    optimize_args = {
        'ema_window_small':[3, 6, 12, 24],
        'ema_window_large':range(24 * 7, 24 * 14 + 1, 24),
        'lookback_window_trend': [24*7, 24*14, 24*21, 24*28],
        'trend_threshold': [10, 20, 30, 40, 50],
    }

    symbol_ids = [
        'ETH_USD_COINBASE', #'LINK_ETH_BINANCE', 'BNB_ETH_BINANCE',
        #'ADA_ETH_BINANCE', #'DOGE_ETH_YOBIT', 'EOS_ETH_BINANCE',
    #     'XRP_ETH_BINANCE', 'THETA_ETH_GATEIO', 'XLM_ETH_BINANCE',
    #     'XMR_ETH_BINANCE', 'ZEC_ETH_BINANCE', 'BCH_ETH_HITBTC',
    #     'IOTA_ETH_BINANCE'
     ]

    b = BackTester(
        start_date = '2021/06/25',
        end_date = '2022/11/11',
        symbol_ids = symbol_ids,
        strategies = [MovingAverage],
        optimize = True,
        optimize_dict = {MovingAverage: optimize_args},
        pct_training_data = 0.7,

    )

    backtest_start = time.time()
    strat_results = b.evaluate_strategies()
    backtest_end = time.time()

    print()
    print('time elapsed since backtest: {} mins'.format(round(abs(backtest_end - backtest_start) / 60.0, 2)))