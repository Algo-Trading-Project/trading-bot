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
from core.performance_metrics import calculate_performance_metrics

class BackTester:

    def __init__(self, 
                 strategies,
                 optimization_metric = 'Sortino Ratio',
                 backtest_params = {'init_cash':10_000, 'fees':0.005}
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

    def backtest(self, base, quote, exchange, strat, training_data, testing_data, starting_equity):
        print('starting equity: ', starting_equity)
        backtest_params = self.backtest_params.copy()
        backtest_params['init_cash'] = starting_equity

        is_backtest = Backtest(
            strategy = strat,
            price_data = training_data,
            optimization_metric = self.optimization_metric,
            backtest_params = backtest_params
        )   

        optimal_params = is_backtest.optimize() 

        # Evaluate optimized trading strategy on unseen data
        oos_backtest = Backtest(
            strategy = strat,
            price_data = testing_data,
            optimization_metric = self.optimization_metric,
            backtest_params = backtest_params
        )   

        oos_trades, oos_equity_curve = oos_backtest.backtest(optimal_params)
        
        return oos_trades, oos_equity_curve
    
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
        
        equity_curves = []
        trades = []
        price_data = []

        starting_equity = 10_000

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
                        
            oos_trades, oos_equity_curve = self.backtest(
                base = base,
                quote = quote,
                exchange = exchange,
                strat = strat,
                training_data = in_sample_data,
                testing_data = out_of_sample_data,
                starting_equity = starting_equity
            )
            
            tr = 1 + ((oos_equity_curve['equity'].iloc[-1] - oos_equity_curve['equity'].iloc[0]) / oos_equity_curve['equity'].iloc[0])
            
            print('num trades: {}, avg trade: {}'.format(len(oos_trades), round(oos_trades['pnl_pct'].mean(), 5)))
            print('total return: {}'.format(tr))

            starting_equity = oos_equity_curve.iloc[-1]['equity']
            
            oos_trades['entry_date'] = oos_trades['entry_date'].astype(str)
            oos_trades['exit_date'] = oos_trades['exit_date'].astype(str)
            oos_trades_dict = oos_trades.to_dict(orient = 'records')
            
            insert_str = ''

            for trade in oos_trades_dict:
                pnl = float(str(trade['pnl'])[:38])
                pnl_pct = float(str(trade['pnl_pct'])[:38])

                insert_str += """('{}', '{}', '{}', '{}', '{}', '{}', '{}'), """.format(
                    base + '_' + quote + '_' + exchange, 
                    strat.indicator_factory_dict['class_name'],
                    trade['entry_date'],
                    trade['exit_date'],
                    pnl,
                    pnl_pct,
                    trade['is_long']
                )

            insert_str = insert_str[:-2]

            insert_str_2 = ''
            
            for i in range(len(oos_equity_curve)):
                date = oos_equity_curve.index[i]
                equity = oos_equity_curve['equity'].iloc[i]

                insert_str_2 += """('{}', '{}', '{}', {}), """.format(
                    base + '_' + quote + '_' + exchange,
                    strat.indicator_factory_dict['class_name'],
                    date,
                    equity
                )

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
                        INSERT INTO eth.backtest_trades VALUES {}
                        """.format(insert_str)
                        
                        # Execute query on Redshift
                        cursor.execute(query)

                    query = """
                    INSERT INTO eth.backtest_equity_curves VALUES {}
                    """.format(insert_str_2)

                    # Execute query on Redshift
                    cursor.execute(query)

                    cursor.close()

                conn.commit()

            equity_curves.append(oos_equity_curve)
            trades.append(oos_trades)
            price_data.append(out_of_sample_data)
            
            start += out_of_sample_size

        equity_curves = pd.concat(equity_curves).sort_index()
        trades = pd.concat(trades, ignore_index = True)
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

                oos_equity_curves, oos_trades, oos_price_data = self.walk_forward_optimization(
                    base = base,
                    quote = quote, 
                    exchange = exchange,
                    strat = strat,
                    in_sample_size = 24 * 30 * 6,
                    out_of_sample_size = 24 * 30 * 3 
                )

                performance_metrics = calculate_performance_metrics(
                    oos_equity_curves, 
                    oos_trades, 
                    oos_price_data
                ).to_dict(orient = 'records')[0]
                
                performance_metrics = json.dumps(performance_metrics, default = BackTester.serialize_json_data)
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
                        INSERT INTO eth.backtest_results VALUES
                        (%s, %s, %s, %s, %s, %s)
                        """

                        # Execute query on Redshift
                        params = (
                            strat.indicator_factory_dict['class_name'],
                            base + '_' + quote + '_' + exchange,
                            oos_price_data.index[0],
                            oos_price_data.index[-1],
                            performance_metrics,  
                            self.optimization_metric
                        )
                        
                        cursor.execute(query, params)
                        cursor.close()

                    conn.commit()

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