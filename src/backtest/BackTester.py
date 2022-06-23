#################################
# ADDING PACKAGE TO PYTHON PATH #
#################################
from math import ceil
import sys
import os

sys.path.append(os.getcwd())

#################################
#             MISC              #
#################################
import time
import redshift_connector
import pandas as pd
import numpy as np
import multiprocessing as mp

#################################
#      TRADING STRATEGIES       #
#################################
from backtesting import Backtest
from src.strategies.IchimokuCloud import IchimokuCloud

class BackTester:

    def __init__(self, 
                 symbol_ids,
                 start_date,
                 end_date,
                 strategies,
                 optimize = False,
                 optimize_dict = None,
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
                        maximize = 'Return [%]'
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
                stats_for_strat['strategy_name'] = strat.name()

                stats_for_strat_df = stats_for_strat[backtest_result_cols].to_frame().T
                backtest_results = backtest_results.append(stats_for_strat_df, ignore_index = True)

                backtest_results.to_csv('src/backtest/backtest_results.csv')
        
        return backtest_results

if __name__ == '__main__': 
    mp.set_start_method('fork')

    optimize_args_ichimoku_cloud = {
        'tenkan_window':np.arange(5, 5 * 3 + 1, 5).tolist(),
        'kijun_window':np.arange(10, 10 * 3 + 1, 10).tolist(),
        'senkou_b_window':np.arange(24, 24 * 3 + 1, 24).tolist(),
        'senkou_ab_period':np.arange(1, 6).tolist(),
        'en_bar':np.arange(4, 9).tolist(),
        'ex_bar':np.arange(4, 8).tolist(),
        'constraint':lambda x: (x.tenkan_window < x.kijun_window) and (x.tenkan_window < x.senkou_b_window) and (x.kijun_window < x.senkou_b_window)
    }

    symbol_ids = [
        'ETH_USD_COINBASE', 'LINK_ETH_BINANCE', 'BNB_ETH_BINANCE',
        'ADA_ETH_BINANCE', 'DOGE_ETH_YOBIT', 'EOS_ETH_BINANCE',
        'XRP_ETH_BINANCE', 'THETA_ETH_GATEIO', 'XLM_ETH_BINANCE',
        'XMR_ETH_BINANCE', 'ZEC_ETH_BINANCE', 'BCH_ETH_HITBTC',
        'IOTA_ETH_BINANCE'
    ]

    b = BackTester(
        symbol_ids = symbol_ids,
        start_date = '2018/06/19',
        end_date = '2021/06/19',
        strategies = [IchimokuCloud],
        optimize = True,
        optimize_dict = {
            IchimokuCloud:optimize_args_ichimoku_cloud
        },
        pct_training_data = 0.5
    )

    backtest_start = time.time()
    strat_results = b.evaluate_strategies()
    backtest_end = time.time()

    print()
    print('time elapsed since backtest: {} mins'.format(round(abs(backtest_end - backtest_start) / 60.0, 2)))