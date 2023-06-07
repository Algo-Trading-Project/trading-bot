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

class PairsTradingBackTester:

    def __init__(self, 
                 optimize_dict,
                 optimization_metric = 'Sortino Ratio'
                 ):
        
        self.optimize_dict = optimize_dict
        self.optimization_metric = optimization_metric

    def serialize_json_data(obj):
        if isinstance(obj, pd.Timedelta):
            return obj.total_seconds()
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
                    FROM token_price.eth.stg_price_data_1h
                    WHERE
                        asset_id_base = '{}' AND
                        asset_id_quote = '{}' AND
                        exchange_id = '{}'
                ), symbol_id_2 AS (
                    SELECT 
                        time_period_end,
                        price_close
                    FROM token_price.eth.stg_price_data_1h
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
            
    def __is_cointegrated(self, symbol_id_1, symbol_id_2, p_val_thresh = 0.05, correlation_thresh = 0.8):                    
        data = self.__get_data(
            symbol_id_1 = symbol_id_1,
            symbol_id_2 = symbol_id_2
        )  

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
        
    def __calculate_performance_metrics(self, oos_equity_curve, oos_trades, oos_price_data):

        ########################### HELPER FUNCTIONS ####################################
        def exposure_time(duration, trades):
            exposure = timedelta(days = 0, hours = 0)

            for i in range(len(trades)):
                entry_date = pd.to_datetime(trades.at[i, 'entry_date'])
                exit_date = pd.to_datetime(trades.at[i, 'exit_date'])
                trade_duration = exit_date - entry_date
                exposure += trade_duration
            
            return round(exposure / duration * 100, 2)

        def buy_and_hold_return(data):
            token1_start_value = data.at[data.index[0], data.columns[0]]
            token1_end_value = data.at[data.index[-1], data.columns[0]]
            buy_and_hold_return_token_1 = round((token1_end_value - token1_start_value) / token1_start_value * 100, 2)
            
            token2_start_value = data.at[data.index[0], data.columns[1]]
            token2_end_value = data.at[data.index[-1], data.columns[1]]
            buy_and_hold_return_token_2 = round((token2_end_value - token2_start_value) / token2_start_value * 100, 2)

            return max([buy_and_hold_return_token_1, buy_and_hold_return_token_2])
        
        def sharpe_ratio(equity):
            returns = equity.equity.pct_change()
            mean_returns = returns.mean()
            std_returns = returns.std()
            
            try:
                return np.sqrt(8760) * mean_returns / std_returns 
            except:
                return np.nan
            
        def sortino_ratio(equity):
            returns = equity.pct_change()
            negative_returns = returns[returns['equity'] < 0]
            
            mean_returns = returns.mean()
            std_negative_returns = negative_returns.std()
            
            try:
                return np.sqrt(8760) * mean_returns / std_negative_returns 
            except:
                return np.nan
            
        def calmar_ratio(equity):
            num_years = len(equity) / 8760
            cum_ret_final = (1 + equity.equity.pct_change()).prod().squeeze()
            annual_returns = cum_ret_final ** (1 / num_years) - 1
            
            try:
                return annual_returns / abs(max_drawdown(equity) / 100)
            except:
                return np.nan

        def cagr_over_avg_drawdown(equity):
            num_years = len(equity) / 8760
            cum_ret_final = (1 + equity.equity.pct_change()).prod().squeeze()
            annual_returns = cum_ret_final ** (1 / num_years) - 1
            
            try:
                return annual_returns / abs(avg_drawdown(equity) / 100)
            except:
                return np.nan

        def profit_factor(trades):
            if len(trades) == 0:
                return np.nan
            
            gross_profit = trades[trades['pnl'] > 0]['pnl'].sum()
            gross_loss = abs(trades[trades['pnl'] < 0]['pnl'].sum())
            
            return gross_profit / gross_loss
        
        def max_drawdown(equity):
            rolling_max_equity = equity.cummax()
            drawdown = (equity / rolling_max_equity) - 1
            max_dd = drawdown.min()
            return round(max_dd * 100, 2)
        
        def avg_drawdown(equity):
            rolling_max_equity = equity.cummax()
            drawdown = (equity / rolling_max_equity) - 1
            avg_dd = drawdown.mean()
            return round(avg_dd * 100, 2)
        
        def max_drawdown_duration(equity):
            dates = pd.Series([pd.to_datetime(equity.index[0])])
            
            diff = equity.cummax().diff().fillna(0)
            diff = diff[diff['equity'] != 0]

            for date in diff.index:
                date = pd.to_datetime(date)
                dates = pd.concat([dates, pd.Series(date)], ignore_index = True)

            return dates.diff().max()
        
        def avg_drawdown_duration(equity):
            dates = pd.Series([pd.to_datetime(equity.index[0])])
            
            diff = equity.cummax().diff().fillna(0)
            diff = diff[diff['equity'] != 0]

            for date in diff.index:
                date = pd.to_datetime(date)
                dates = pd.concat([dates, pd.Series(date)], ignore_index = True)

            return dates.diff().mean()

        def win_rate(trades):
            if len(trades) == 0:
                return np.nan
            
            num_winning_trades = len(trades[trades['pnl_pct'] > 0])
            num_trades_total = len(trades)

            return round(num_winning_trades / num_trades_total * 100, 2)
        
        def best_trade(trades):
            if len(trades) == 0:
                return np.nan
            
            return round(trades['pnl_pct'].max() * 100, 2)
            
        def worst_trade(trades):
            if len(trades) == 0:
                return np.nan
            
            return round(trades['pnl_pct'].min() * 100, 2)
        
        def avg_trade(trades):
            if len(trades) == 0:
                return np.nan
            
            return round(trades['pnl_pct'].mean() * 100, 2)
        
        def max_trade_duration(trades):
            if len(trades) == 0:
                return np.nan
            
            return (pd.to_datetime(trades['exit_date']) - pd.to_datetime(trades['entry_date'])).max()            
            
        def avg_trade_duration(trades):
            if len(trades) == 0:
                return np.nan
            
            return (pd.to_datetime(trades['exit_date']) - pd.to_datetime(trades['entry_date'])).mean()            
        #################################### HELPER FUNCTIONS END ####################################
                
        start = oos_price_data.index[0]
        end = oos_price_data.index[-1]
        duration = pd.to_datetime(end) - pd.to_datetime(start)

        return_pct = (oos_equity_curve['equity'].iloc[-1] - oos_equity_curve['equity'].iloc[0]) / oos_equity_curve['equity'].iloc[0]
        
        metrics_dict = {
            'duration':duration, 
            'exposure_time': exposure_time(duration, oos_trades),
            'equity_final':oos_equity_curve.dropna().iloc[-1]['equity'],
            'equity_peak':oos_equity_curve['equity'].max(),
            'return_pct':round(return_pct * 100, 2),
            'buy_and_hold_return_pct':buy_and_hold_return(oos_price_data),
            'sharpe_ratio':sharpe_ratio(oos_equity_curve), 
            'sortino_ratio':sortino_ratio(oos_equity_curve),
            'calmar_ratio':calmar_ratio(oos_equity_curve),
            'cagr_over_avg_drawdown':cagr_over_avg_drawdown(oos_equity_curve),
            'profit_factor':profit_factor(oos_trades),
            'max_dd_pct':max_drawdown(oos_equity_curve),
            'avg_dd_pct':avg_drawdown(oos_equity_curve),
            'max_dd_duration':max_drawdown_duration(oos_equity_curve), 
            'avg_dd_duration':avg_drawdown_duration(oos_equity_curve),
            'num_trades':len(oos_trades), 
            'win_rate_pct':win_rate(oos_trades), 
            'best_trade_pct':best_trade(oos_trades),
            'worst_trade_pct':worst_trade(oos_trades), 
            'avg_trade_pct':avg_trade(oos_trades),
            'max_trade_duration':max_trade_duration(oos_trades),
            'avg_trade_duration':avg_trade_duration(oos_trades)
        }

        return pd.DataFrame(metrics_dict).reset_index()

    ########################################## HELPER FUNCTIONS END ##########################################

    def backtest(self, symbol_id_1, symbol_id_2, training_data, testing_data):
        is_backtest = PairsTradingBacktest(
            price_data = training_data,
            symbol_id_1 = symbol_id_1,
            symbol_id_2 = symbol_id_2
        )

        optimal_params, is_backtest_results = is_backtest.optimize_parameters(
            optimize_dict = self.optimize_dict,
            performance_metric = self.optimization_metric,
            minimize = False
        )

        # Evaluate optimized trading strategy on unseen data

        oos_backtest = PairsTradingBacktest(
            price_data = testing_data,
            symbol_id_1 = symbol_id_1,
            symbol_id_2 = symbol_id_2
        )

        oos_backtest.z_window = optimal_params['z_window']
        oos_backtest.hedge_ratio_window = optimal_params['hedge_ratio_window']
        oos_backtest.z_score_upper_thresh = optimal_params['z_score_upper_thresh']
        oos_backtest.z_score_lower_thresh = optimal_params['z_score_lower_thresh']

        oos_backtest_results = oos_backtest.backtest()

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
                INSERT INTO trading_bot.eth.pairs_trading_backtest_results VALUES 
                (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                str_is_backtest_results = json.dumps(is_backtest_results, default = PairsTradingBackTester.serialize_json_data)
                str_is_backtest_results = str_is_backtest_results.replace('NaN', 'null').replace('Infinity', 'null')

                str_oos_backtest_results = json.dumps(oos_backtest_results, default = PairsTradingBackTester.serialize_json_data)
                str_oos_backtest_results = str_oos_backtest_results.replace('NaN', 'null').replace('Infinity', 'null')

                # Execute query on Redshift
                params = (symbol_id_1, symbol_id_2, training_data.index[0], training_data.index[-1], 
                          str_is_backtest_results, testing_data.index[0], testing_data.index[-1],
                          str_oos_backtest_results, json.dumps(optimal_params, default = PairsTradingBackTester.serialize_json_data),
                          self.optimization_metric, json.dumps(oos_backtest.backtest_params, default = PairsTradingBackTester.serialize_json_data)
                          )
                
                cursor.execute(query, params)
                cursor.close()

            conn.commit()
        
        return oos_backtest.equity, oos_backtest.trades
    
    def walk_forward_optimization(self, symbol_id_1, symbol_id_2,
                                  in_sample_size, out_of_sample_size):
        
        # Walk-forward analysis
        start = 0

        # Fetch revelant price data for backtest
        backtest_data = self.__get_data(
            symbol_id_1 = symbol_id_1,
            symbol_id_2 = symbol_id_2
        )

        equity_curves = []
        trades = []
        price_data = []

        while start + in_sample_size + out_of_sample_size <= len(backtest_data):
            print()
            print('Backtesting on days: {} - {} / {}'.format(int(start / 24), int((start + in_sample_size + out_of_sample_size) / 24), int(len(backtest_data) / 24)))
            print()

            if len(backtest_data.iloc[start:]) - (in_sample_size + out_of_sample_size) < out_of_sample_size:
                in_sample_data = backtest_data.iloc[start:start + in_sample_size]
                out_of_sample_data = backtest_data.iloc[start + in_sample_size:]
            else:
                in_sample_data = backtest_data.iloc[start:start + in_sample_size]
                out_of_sample_data = backtest_data.iloc[start + in_sample_size:start + in_sample_size + out_of_sample_size]
                        
            oos_equity_curve, oos_trades = self.backtest(
                symbol_id_1 = symbol_id_1,
                symbol_id_2 = symbol_id_2,
                training_data = in_sample_data,
                testing_data = out_of_sample_data
            )
            
            equity_curves.append(oos_equity_curve)
            trades.append(oos_trades)
            price_data.append(out_of_sample_data)
            
            start += out_of_sample_size

        equity_curves = pd.concat(equity_curves).sort_index()
        trades = pd.concat(trades, ignore_index = True).sort_index()
        price_data = pd.concat(price_data).sort_index()

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
                    FROM token_price.eth.stg_price_data_1h o
                    GROUP BY         
                        asset_id_base,
                        asset_id_quote,
                        exchange_id
                )

                SELECT 
                    o.asset_id_base,
                    o.asset_id_quote,
                    o.exchange_id
                FROM token_price.eth.stg_price_data_1h o INNER JOIN last_month l
                    ON o.asset_id_base = l.asset_id_base AND
                        o.asset_id_quote = l.asset_id_quote AND
                        o.exchange_id = l.exchange_id
                WHERE
                    o.asset_id_base != 'ETH' AND
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

                for i in range(len(combs)):
                    symbol_id_1, symbol_id_2 = combs[i][0], combs[i][1]
                        
                    is_cointegrated, x_y_dict = self.__is_cointegrated(
                        symbol_id_1 = symbol_id_1,
                        symbol_id_2 = symbol_id_2,
                        p_val_thresh = 0.05,
                        correlation_thresh = 0.8
                    )

                    print('({}/{}) ({}, {}) is cointegrated: {}'.format(i + 1, len(combs), symbol_id_1, symbol_id_2, is_cointegrated))
                    print()

                    if is_cointegrated:
                        print()
                        print('({} / {}) Now backtesting pairs {} / {}'.format(i + 1, len(combs), symbol_id_1, symbol_id_2))
                        print()

                        oos_equity_curve, oos_trades, oos_price_data = self.walk_forward_optimization(
                            symbol_id_1 = x_y_dict['X'],
                            symbol_id_2 = x_y_dict['Y'],
                            in_sample_size = 24 * 30 * 6,
                            out_of_sample_size = 24 * 30 * 3 
                        )

                        performance_metrics = self.__calculate_performance_metrics(
                            oos_equity_curve, 
                            oos_trades, 
                            oos_price_data
                        ).to_dict(orient = 'records')[0]
                        
                        performance_metrics = json.dumps(performance_metrics, default = PairsTradingBackTester.serialize_json_data)
                        performance_metrics = performance_metrics.replace('NaN', 'null')

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
                                (%s, %s, %s, %s, %s, %s, %s, %s)
                                """

                                # Execute query on Redshift
                                params = (symbol_id_1, symbol_id_2, oos_price_data.index[0],
                                          oos_price_data.index[-1], performance_metrics, self.optimization_metric)
                                
                                cursor.execute(query, params)
                                cursor.close()

                            conn.commit()
if __name__ == '__main__': 
    optimize_dict = {
        'z_window':[24, 24*7, 24 * 30, 24 * 60],
        'hedge_ratio_window':[24, 24*7, 24 * 30, 24 * 60],
        'z_thresh_upper':[1, 1.5, 2],
        'z_thresh_lower':[-1, -1.5, -2]
    }

    b = PairsTradingBackTester(
        optimize_dict = optimize_dict,
        optimization_metric = 'sortino_ratio'
    )

    print('Starting')
    print()

    backtest_start = time.time()
    b.execute()
    backtest_end = time.time()

    print()
    print('Total Time Elapsed: {} mins'.format(round(abs(backtest_end - backtest_start) / 60.0, 2)))