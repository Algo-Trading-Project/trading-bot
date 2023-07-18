#################################
# ADDING PACKAGE TO PYTHON PATH #
#################################
import sys
import os
sys.path.append(os.getcwd())

#################################
#         MISC IMPORTS          #
#################################
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', UserWarning)
warnings.simplefilter('ignore', RuntimeWarning)

import time
import redshift_connector
import pandas as pd
pd.options.mode.chained_assignment = None

import numpy as np
import json
import itertools
import statsmodels.api as sm

from datetime import timedelta
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import adfuller
###############################################
#      TRADING STRATEGIES / BACKTESTING       #
###############################################
from core.PairsTradingBackTest import PairsTradingBacktest
from core.performance_metrics import calculate_performance_metrics

class PairsTradingBackTester:

    def __init__(self, 
                 optimize_dict,
                 optimization_metric = 'Sortino Ratio'
                 ):
        
        self.optimize_dict = optimize_dict
        self.optimization_metric = optimization_metric

    def serialize_json_data(obj):
        if isinstance(obj, pd.Timedelta):
            return round(obj.total_seconds() / float(60 * 60 * 24), 2)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        
    def numpy_combinations(x):
        idx = np.stack(np.triu_indices(len(x), k=1), axis=-1)
        return x[idx]
    
    ############################################ HELPER FUNCTIONS ############################################

    def __get_data(self, symbol_id_1, symbol_id_2):
        base_1, quote_1, exchange_id_1 = symbol_id_1.split('_')
        base_2, quote_2, exchange_id_2 = symbol_id_2.split('_')
                
        with redshift_connector.connect(
            host = 'project-poseidon.cpsnf8brapsd.us-west-2.redshift.amazonaws.com',
            database = 'token_price',
            user = 'administrator',
            password = 'Free2play2'
        ) as conn:
            with conn.cursor() as cursor:
                query = """
                WITH symbol_id_1 AS (
                    SELECT 
                        time_period_end,
                        price_close
                    FROM token_price.coinapi.price_data_1h
                    WHERE
                        asset_id_base = '{}' AND
                        asset_id_quote = '{}' AND
                        exchange_id = '{}'
                ), symbol_id_2 AS (
                    SELECT 
                        time_period_end,
                        price_close
                    FROM token_price.coinapi.price_data_1h
                    WHERE
                        asset_id_base = '{}' AND
                        asset_id_quote = '{}' AND
                        exchange_id = '{}'
                )
                
                SELECT
                    s1.time_period_end,
                    s1.price_close AS "{}",
                    s2.price_close AS "{}"
                FROM symbol_id_1 s1 INNER JOIN symbol_id_2 s2
                        ON s1.time_period_end = s2.time_period_end
                ORDER BY s1.time_period_end 
                """.format(
                    base_1, quote_1, exchange_id_1,
                    base_2, quote_2, exchange_id_2,
                    symbol_id_1, symbol_id_2
                )
        
                cursor.execute(query)
                tuples = cursor.fetchall()
        
                # Return queried data as a DataFrame
                df = pd.DataFrame(tuples, columns = ['time_period_end', symbol_id_1, symbol_id_2]).set_index('time_period_end')
                df = df.astype(float)

                return df
            
    def __is_cointegrated(self, symbol_id_1, symbol_id_2, p_val_thresh = 0.05, correlation_thresh = 0.9):                    
        data = self.__get_data(
            symbol_id_1 = symbol_id_1,
            symbol_id_2 = symbol_id_2
        ).iloc[-24 * 365 * 2:]

        correlation = np.corrcoef(data[symbol_id_1].values, data[symbol_id_2].values)[0][1]

        if correlation < correlation_thresh:
            return False, {}
                    
        X = data[symbol_id_1]
        X = add_constant(X)
        
        Y = data[symbol_id_2]
        Y = add_constant(Y)
                
        ols1 = sm.OLS(Y[symbol_id_2], X).fit()
        ols2 = sm.OLS(X[symbol_id_1], Y).fit()
        
        best_ols = min([ols1, ols2], key = lambda x: adfuller(x.resid)[1])
        best_p_val = adfuller(best_ols.resid)[1]
        
        if best_ols == ols1 and best_p_val <= p_val_thresh:
            coint_dict = {
                'X':symbol_id_1,
                'Y':symbol_id_2
            }
            return True, coint_dict
            
        elif best_ols == ols2 and best_p_val <= p_val_thresh:
            coint_dict = {
                'X':symbol_id_2,
                'Y':symbol_id_1
            }
            return True, coint_dict

        else:
            return False, {}
        
    ########################################## HELPER FUNCTIONS END ##########################################

    def backtest(self, symbol_id_1, symbol_id_2, 
                 price_data, starting_equity, is_start_i, 
                 is_end_i, oos_start_i, oos_end_i):
        
        is_backtest = PairsTradingBacktest(
            price_data = price_data,
            symbol_id_1 = symbol_id_1,
            symbol_id_2 = symbol_id_2,
            start_i = is_start_i,
            end_i = is_end_i
        )

        is_backtest.backtest_params['initial_capital'] = starting_equity

        optimal_params = is_backtest.optimize_parameters(
            optimize_dict = self.optimize_dict,
            performance_metric = self.optimization_metric,
            minimize = False
        )

        # Evaluate optimized trading strategy on unseen data
        oos_backtest = PairsTradingBacktest(
            price_data = price_data,
            symbol_id_1 = symbol_id_1,
            symbol_id_2 = symbol_id_2,
            start_i = oos_start_i,
            end_i = oos_end_i
        )
       
        oos_backtest.backtest_params['initial_capital'] = starting_equity

        oos_backtest.z_window = optimal_params['z_window']
        oos_backtest.hedge_ratio_window = optimal_params['hedge_ratio_window']
        oos_backtest.z_score_upper_thresh = optimal_params['z_score_upper_thresh']
        oos_backtest.z_score_lower_thresh = optimal_params['z_score_lower_thresh']
        # oos_backtest.rolling_cointegration_window = optimal_params['rolling_cointegration_window']
        oos_backtest.max_holding_time = optimal_params['max_holding_time']

        oos_backtest.backtest()
        
        return oos_backtest.trades, oos_backtest.equity
    
    def walk_forward_optimization(self, symbol_id_1, symbol_id_2,
                                  in_sample_size, out_of_sample_size):
        
        # Walk-forward analysis
        start = 0

        # Fetch revelant price data for backtest
        backtest_data = self.__get_data(
            symbol_id_1 = symbol_id_1,
            symbol_id_2 = symbol_id_2
        ).iloc[-24 * 365 * 2:]

        equity_curves = []
        trades = []
        price_data = []
        # capital_curves = []

        starting_curr_capital = 10_000

        while start + in_sample_size + out_of_sample_size <= len(backtest_data):
            print()
            print('Backtesting on days: {} - {} / {}'.format(int(start / 24), int((start + in_sample_size + out_of_sample_size) / 24), int(len(backtest_data) / 24)))
            print('Starting curr capital: {}'.format(starting_curr_capital))

            if len(backtest_data.iloc[start:]) - (in_sample_size + out_of_sample_size) < out_of_sample_size:
                # in_sample_data = backtest_data.iloc[start:start + in_sample_size]
                # out_of_sample_data = backtest_data.iloc[start + in_sample_size:]
                is_start_i = start
                is_end_i = start + in_sample_size

                oos_start_i = start + in_sample_size
                oos_end_i = len(backtest_data)
            else:
                # in_sample_data = backtest_data.iloc[start:start + in_sample_size]
                # out_of_sample_data = backtest_data.iloc[start + in_sample_size:start + in_sample_size + out_of_sample_size]
                is_start_i = start
                is_end_i = start + in_sample_size

                oos_start_i = start + in_sample_size
                oos_end_i = start + in_sample_size + out_of_sample_size

            oos_trades, oos_equity_curve = self.backtest(
                symbol_id_1 = symbol_id_1,
                symbol_id_2 = symbol_id_2,
                price_data = backtest_data,
                starting_equity = starting_curr_capital,
                is_start_i = is_start_i,
                is_end_i = is_end_i,
                oos_start_i = oos_start_i,
                oos_end_i = oos_end_i
            )

            tr = 1 + ((oos_equity_curve['equity'].iloc[-1] - oos_equity_curve['equity'].iloc[0]) / oos_equity_curve['equity'].iloc[0])

            print('num trades: {}, avg trade: {}'.format(len(oos_trades), round(oos_trades['pnl_pct'].mean(), 5)))
            print('total return: {}'.format(tr))

            starting_curr_capital = oos_equity_curve['equity'].iloc[-1]

            oos_trades['entry_date'] = oos_trades['entry_date'].astype(str)
            oos_trades['exit_date'] = oos_trades['exit_date'].astype(str)
            oos_trades_dict = oos_trades.to_dict(orient = 'records')

            insert_str = ''

            for trade in oos_trades_dict:
                pnl = float(str(trade['pnl'])[:38])
                pnl_pct = float(str(trade['pnl_pct'])[:38])

                insert_str += """('{}', '{}', '{}', '{}', {}, {}, {}, {}, '{}'), """.format(
                    trade['entry_date'], 
                    trade['exit_date'], 
                    symbol_id_1, 
                    symbol_id_2,
                    pnl, 
                    pnl_pct, 
                    trade[symbol_id_1], 
                    trade[symbol_id_2], 
                    str(int(trade['is_long']))
                )

            insert_str = insert_str[:-2]

            insert_str_2 = ''
            
            for i in range(len(oos_equity_curve)):
                date = oos_equity_curve.index[i]
                equity = oos_equity_curve['equity'].iloc[i]

                insert_str_2 += """('{}', '{}', '{}', {}), """.format(symbol_id_1, symbol_id_2, date, equity)

            insert_str_2 = insert_str_2[:-2]

            with redshift_connector.connect(
                host = 'project-poseidon.cpsnf8brapsd.us-west-2.redshift.amazonaws.com',
                database = 'trading_bot',
                user = 'administrator',
                password = 'Free2play2'
            ) as conn:
                with conn.cursor() as cursor:
                    # Query to insert backtest results into Redshift 
                    if len(oos_trades) > 0:
                        query = """
                        INSERT INTO eth.pair_trading_backtest_trades VALUES {}
                        """.format(insert_str)
                        
                        # Execute query on Redshift
                        cursor.execute(query)

                    query = """
                    INSERT INTO eth.pair_trading_backtest_equity_curves VALUES {}
                    """.format(insert_str_2)

                    # Execute query on Redshift
                    cursor.execute(query)

                    cursor.close()

                conn.commit()

            trades.append(oos_trades)
            price_data.append(backtest_data.iloc[oos_start_i:oos_end_i])
            equity_curves.append(oos_equity_curve)
            
            start += out_of_sample_size
        
        trades = pd.concat(trades, ignore_index = True)
        price_data = pd.concat(price_data).sort_index()
        equity_curves = pd.concat(equity_curves).sort_index()

        return equity_curves, trades, price_data
                                                        
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
                top_100_tokens_by_volume_query = """
                WITH last_month AS (
                    SELECT  
                        asset_id_base,
                        asset_id_quote,
                        exchange_id,
                        MAX(time_period_start) - INTERVAL '30 DAYS' AS start_date
                    FROM token_price.coinapi.price_data_1h
                    GROUP BY         
                        asset_id_base,
                        asset_id_quote,
                        exchange_id
                )

                SELECT 
                    o.asset_id_base,
                    o.asset_id_quote,
                    o.exchange_id
                FROM token_price.coinapi.price_data_1h o INNER JOIN last_month l
                    ON o.asset_id_base = l.asset_id_base AND
                        o.asset_id_quote = l.asset_id_quote AND
                        o.exchange_id = l.exchange_id 
                WHERE
                    time_period_start >= start_date
                GROUP BY o.asset_id_base, o.asset_id_quote, o.exchange_id
                ORDER BY AVG(volume_traded / (1 / price_close)) * 24 DESC
                LIMIT 50
                """
                cursor.execute(top_100_tokens_by_volume_query)
                tuples = cursor.fetchall()
                
                token_df = pd.DataFrame(tuples, columns = ['asset_id_base', 'asset_id_quote', 'exchange_id'])
                token_df['symbol_id'] = token_df['asset_id_base'] + '_' + token_df['asset_id_quote'] + '_' + token_df['exchange_id']
               
                combs = PairsTradingBackTester.numpy_combinations(token_df['symbol_id'].to_numpy())

                history = {}

                for i in range(len(combs)):
                    p1, p2 = combs[i][0], combs[i][1]

                    if history.get(p1.split('_')[0] + '_' + p2.split('_')[0]):
                        continue
                    
                    if p1.split('_')[0] == p2.split('_')[0]:
                        continue
                        
                    is_cointegrated, x_y_dict = self.__is_cointegrated(
                        symbol_id_1 = p1,
                        symbol_id_2 = p2,
                        p_val_thresh = 0.05,
                        correlation_thresh = 0.8
                    )

                    print('({}/{}) ({}, {}) is cointegrated: {}'.format(i + 1, len(combs), p1, p2, is_cointegrated))
                    print()

                    if is_cointegrated:
                        history[p1.split('_')[0] + '_' + p2.split('_')[0]] = True

                        print()
                        print('({} / {}) Now backtesting pairs {} / {}'.format(i + 1, len(combs), x_y_dict['X'], x_y_dict['Y']))
                        print()

                        oos_equity_curves, oos_trades, oos_price_data = self.walk_forward_optimization(
                            symbol_id_1 = x_y_dict['X'],
                            symbol_id_2 = x_y_dict['Y'],
                            in_sample_size = 24 * 30 * 4,
                            out_of_sample_size = 24 * 30 * 2,  
                        )

                        performance_metrics = calculate_performance_metrics(
                            oos_equity_curves, 
                            oos_trades, 
                            oos_price_data
                        ).to_dict(orient = 'records')[0]
                        
                        performance_metrics = json.dumps(performance_metrics, default = PairsTradingBackTester.serialize_json_data)
                        performance_metrics = performance_metrics.replace('NaN', 'null').replace('Infinity', 'null')

                        with redshift_connector.connect(
                            host = 'project-poseidon.cpsnf8brapsd.us-west-2.redshift.amazonaws.com',
                            database = 'trading_bot',
                            user = 'administrator',
                            password = 'Free2play2'
                        ) as conn:
                            with conn.cursor() as cursor:
                                # Query to insert backtest results into Redshift 
                                query = """
                                INSERT INTO eth.pairs_trading_backtest_results VALUES
                                (%s, %s, %s, %s, %s, %s)
                                """

                                # Execute query on Redshift
                                params = (x_y_dict['X'], x_y_dict['Y'], oos_price_data.index[0],
                                          oos_price_data.index[-1], performance_metrics,  self.optimization_metric)
                                
                                cursor.execute(query, params)
                                cursor.close()

                            conn.commit()
if __name__ == '__main__': 
    optimize_dict = {
        'z_window':[24, 24 * 7, 24 * 14, 24 * 28],
        'hedge_ratio_window':[24, 24 * 7, 24 * 14, 24 * 28],
        'z_thresh_upper':[1, 2, 3],
        'z_thresh_lower':[-1, -2, -3],
        # 'rolling_cointegration_window':[24 * 7, 24 * 30, 24 * 60],
        'max_holding_time': [24, 24 * 7, 24 * 28, float('inf')]
    }

    b = PairsTradingBackTester(
        optimize_dict = optimize_dict,
        optimization_metric = 'sharpe_ratio'
    )

    print()
    print('Starting')
    print()

    backtest_start = time.time()
    b.execute()
    backtest_end = time.time()

    print()
    print('Total Time Elapsed: {} mins'.format(round(abs(backtest_end - backtest_start) / 60.0, 2)))