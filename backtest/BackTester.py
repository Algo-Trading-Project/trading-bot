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
import itertools

###############################################
#      TRADING STRATEGIES / BACKTESTING       #
###############################################
from core.walk_forward_optimization import WalkForwardOptimization
from core.performance_metrics import calculate_performance_metrics
from core.performance_metrics import compute_deflated_sharpe_ratio

from strategies.base import Strategy
from strategies.ma_crossover import MACrossOver
from strategies.arima import ARIMAStrat
from strategies.bollinger_bands import BollingerBands

class BackTester:

    def __init__(self, 
                 strategies: list,
                 optimization_metric: str = 'Sharpe Ratio',
                 backtest_params: dict = {'init_cash':10_000, 'fees':0.005}
                 ):
        """
        Coordinates the walk-forward optimization of a set of strategies over a set of CoinAPI tokens, 
        utilizing the vectorbt package for efficient parameter optimization. The results of the walk-forward
        optimizations are logged to Redshift for further dashboarding/analysis.

        Parameters:
        -----------
        strategies : list
            List of Strategy classes from backtest/strategies to backtest.

        optimization_metric : str, default = 'Sharpe Ratio'
            Performance metric to maximize/minimize during the in-sample parameter optimizations. 
            The full list of metrics available to use can be found in the file 
            backtest/core/walk_forward_optimization.py in the metric_map dictionary in the __init__ method.

        backtest_params : dict, default = {'init_cash':10_000, 'fees':0.005}
            Dictionary containing miscellaneous parameters to configure the
            backtest.

        """

        self.optimization_metric = optimization_metric
        self.backtest_params = backtest_params
        self.strategies = strategies

    def __serialize_json_data(obj: pd.Timedelta | int | float) -> int | float:
        """
        Converts obj into a form that can be JSON serialized.

        Parameters:
        -----------
        obj : pd.Timedelta or int or float
            A value of a JSON dictionary that needs to be converted
            into a serializable format.

        Returns:
        --------
        int or float:
            Result of converting obj into a JSON serializable format.

        """

        if isinstance(obj, pd.Timedelta):
            return obj.total_seconds()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
             
    def __fetch_OHLCV_df_from_redshift(self, base: str, quote: str, exchange: str) -> pd.DataFrame:
        """
        Queries OHLCV data for {exchange}_SPOT_{base}_{quote} CoinAPI pair stored in
        Project Poseidon's Redshift cluster. Returns the queried data as a DataFrame 
        indexed by timestamp.

        Parameters:
        -----------
        base : str 
            CoinAPI asset_id_base of the token being backtested.

        quote : str
            CoinAPI asset_id_quote of the token being backtested.

        exchange : str
            CoinAPI exchange_id of the token being backtested.

        Returns:
        --------
        DataFrame:
            DataFrame of queried OHLCV data, indexed by timestamp.

        """
        # Connect to Redshift cluster containing price data
        with redshift_connector.connect(
            host = 'project-poseidon.cpsnf8brapsd.us-west-2.redshift.amazonaws.com',
            database = 'token_price',
            user = 'administrator',
            password = 'Free2play2'
        ) as conn:
            with conn.cursor() as cursor:
                # Query to fetch OHLCV data for a token & exchange of interest
                title = base + '-' + quote
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
                """.format(base, quote, exchange)

                # Execute query on Redshift and return result
                cursor.execute(query)
                tuples = cursor.fetchall()
                
                # Return queried data as a DataFrame
                cols = ['Date', 'price_open', 'price_high', 'price_low', 'price_close', 'volume_traded']
                df = pd.DataFrame(tuples, columns = cols).set_index('Date')
                df = df.astype(float)

                return df
            
    def __upsert_into_redshift_table(self, 
                                     table: str, 
                                     insert_str: str, 
                                     cursor: redshift_connector.Cursor, 
                                     conn: redshift_connector.Connection,
                                     start_date,
                                     end_date) -> None:

        begin_transaction = """
        BEGIN TRANSACTION;
        """

        create_staging_table = """
        CREATE TABLE IF NOT EXISTS trading_bot.eth.stg_{table} (LIKE trading_bot.eth.{table})
        """.format(table = table)

        insert_into_staging_table = """
        INSERT INTO trading_bot.eth.stg_{table} VALUES {insert_str}
        """.format(table = table, insert_str = insert_str)

        if table == 'backtest_equity_curves':
            delete_from_target_table = """
            DELETE FROM trading_bot.eth.{table} 
            USING trading_bot.eth.stg_{table}
            WHERE
                {table}.symbol_id = stg_{table}.symbol_id AND
                {table}.strat = stg_{table}.strat AND
                {table}.date = stg_{table}.date
            """.format(table = table)
        elif table == 'backtest_trades':
            delete_from_target_table = """
            DELETE FROM trading_bot.eth.{table} 
            USING trading_bot.eth.stg_{table}
            WHERE
                {table}.symbol_id = stg_{table}.symbol_id AND
                {table}.strat = stg_{table}.strat AND
                {table}.entry_date >= '{start_date}' AND
                {table}.exit_date <= '{end_date}'
            """.format(table = table, start_date = start_date, end_date = end_date)
        else:
            delete_from_target_table = """
            DELETE FROM trading_bot.eth.{table} 
            USING trading_bot.eth.stg_{table}
            WHERE
                {table}.symbol_id = stg_{table}.symbol_id AND
                {table}.strategy = stg_{table}.strategy
            """.format(table = table)

        insert_into_target_table = """
        INSERT INTO trading_bot.eth.{table} 
        SELECT * FROM trading_bot.eth.stg_{table};
        """.format(table = table)

        drop_staging_table = """
        DROP TABLE trading_bot.eth.stg_{table}
        """.format(table = table)

        end_transaction = """
        END TRANSACTION;
        """

        queries = [
            begin_transaction,

            create_staging_table, 

            insert_into_staging_table,

            delete_from_target_table, 

            insert_into_target_table, 

            drop_staging_table,

            end_transaction
        ]

        for query in queries:
            try:
                cursor.execute(query)
            except redshift_connector.InterfaceError as err:
                with redshift_connector.connect(
                    host = 'project-poseidon.cpsnf8brapsd.us-west-2.redshift.amazonaws.com',
                    database = 'trading_bot',
                    user = 'administrator',
                    password = 'Free2play2'
                ) as conn2:
                    with conn2.cursor() as cursor2:
                        
                        for query in queries:
                            cursor2.execute(query)

                    conn2.commit()
                    return
                    
        conn.commit()

    def walk_forward_optimization(self, 
                                  strat, 
                                  backtest_data: pd.DataFrame, 
                                  is_start_i: int, 
                                  is_end_i: int,
                                  oos_start_i: int, 
                                  oos_end_i: int, 
                                  starting_equity: float) -> (pd.DataFrame, pd.DataFrame, float):
        
        """
        Optimizes the parameters of a Strategy (strat) on the in-sample data and performs 
        a backtest on the out-of-sample data w/ the optimized parameters. Returns the 
        out-of-sample trades, out-of-sample equity curve, and the deflated sharpe ratio (DSR) 
        so that it can be logged to Redshift for further dashboarding/analysis.

        Parameters:
        -----------
        strat: Strategy class in backtest/strategies
            Trading strategy to be backtested.

        backtest_data: DataFrame
            OHLCV CoinAPI token price data being backtested on.

        is_start_i: int
            Start index of the in-sample optimization period in backtest_data.

        is_end_i: int
            End index of the in-sample optimization period in backtest_data.

        oos_start_i: int
            Start index of the out-of-sample test period in backtest_data.

        oos_end_i: int
            End index of the out-of-sample test period in backtest_data.

        starting_equity: float
            Amount of capital to initiate the backtest with.

        Returns:
        --------
        DataFrame:
            DataFrame containing the trades from the out-of-sample backtest.

        DataFrame:
            DataFrame containing the equity curve from the out-of-sample backtest.

        float:
            Deflated sharpe ratio (DSR) for the in-sample optimization period.

        """
                
        backtest_params = self.backtest_params.copy()
        backtest_params['init_cash'] = starting_equity

        wfo = WalkForwardOptimization(
            strategy = strat,
            backtest_data = backtest_data,
            is_start_i = is_start_i,
            is_end_i = is_end_i,
            oos_start_i = oos_start_i,
            oos_end_i = oos_end_i,
            optimization_metric = self.optimization_metric,
            backtest_params = backtest_params
        )

        optimal_params, portfolio = wfo.optimize()
        oos_trades, oos_equity_curve = wfo.walk_forward(optimal_params)

        annualization_factor = portfolio.returns_acc.ann_factor

        # Drop potential non-finite sharpe ratio values so we can calculate
        # Deflated Sharpe Ratio without issues
        sharpes = portfolio.returns_acc.sharpe_ratio()
        sharpes.replace([np.inf, -np.inf], np.nan, inplace = True)
        sharpes.dropna(inplace = True)
        
        estimated_sharpe = sharpes.max() / np.sqrt(annualization_factor)
        sharpe_variance = sharpes.var() / annualization_factor
        nb_trials = len(sharpes)
        backtest_horizon = is_end_i - is_start_i + 1
        skew = portfolio.loc[sharpes.idxmax()].returns().skew()
        kurtosis = portfolio.loc[sharpes.idxmax()].returns().kurt()

        deflated_sharpe_ratio = compute_deflated_sharpe_ratio(
            estimated_sharpe = estimated_sharpe,
            sharpe_variance = sharpe_variance,
            nb_trials = nb_trials,
            backtest_horizon = backtest_horizon,
            skew = skew, 
            kurtosis = kurtosis
        )

        return oos_trades, oos_equity_curve, deflated_sharpe_ratio
    
    def execute_wfo(self, 
                    base: str,
                    quote: str, 
                    exchange: str, 
                    strat: Strategy,
                    in_sample_size: int, 
                    out_of_sample_size: int) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, list):

        """
        Executes a walk-forward optimization on an arbitrary trading strategy and a
        {exchange}_SPOT_{base}_{quote} CoinAPI pair. Aggregates and returns the out-of-sample equity curves,
        trades, and token prices to calculate the overall out-of-sample performance metrics.

        Parameters:
        -----------
        base : str 
            CoinAPI asset_id_base of the pair being backtested.

        quote : str
            CoinAPI asset_id_quote of the pair being backtested.

        exchange : str
            CoinAPI exchange_id of the pair being backtested.

        strat: Strategy class in backtest/strategies
            Trading strategy to be backtested.

        in_sample_size : int
            Length of the in-sample optimization period in the walk-forward
            optimization.

        out_of_sample_size : int
            Length of the out-of-sample test period in the walk-forward
            optimization.

        Returns:
        --------
        DataFrame:
            DataFrame containing the combined equity curves from all of the
            out-of-sample periods, sorted by date.

        DataFrame:
            DataFrame containing the combined trades from all of the
            out-of-sample periods.

        DataFrame:
            DataFrame containing the combined token price data from all of the
            out-of-sample periods, sorted by date.

        list:
            List of Deflated Sharpe Ratios (DSR) across all in-sample
            optimization periods.

        """

        # Walk-forward analysis
        start = 0

        # Fetch revelant price data for backtest
        backtest_data = self.__fetch_OHLCV_df_from_redshift(
            base = base,
            quote = quote,
            exchange = exchange
        )
        
        equity_curves = []
        trades = []
        price_data = []
        deflated_sharpe_ratios = []

        starting_equity = 10_000

        while start + in_sample_size + out_of_sample_size <= len(backtest_data):
            print()
            print('Progress: {} / {} days...'.format(int(start / 24), int(len(backtest_data) / 24)))
            print()
            print('*** Starting Equity: ', starting_equity)
            print()

            if len(backtest_data.iloc[start:]) - (in_sample_size + out_of_sample_size) < out_of_sample_size:
                is_start_i = start
                is_end_i = start + in_sample_size

                oos_start_i = start + in_sample_size
                oos_end_i = len(backtest_data)
            else:
                is_start_i = start
                is_end_i = start + in_sample_size

                oos_start_i = start + in_sample_size
                oos_end_i = start + in_sample_size + out_of_sample_size

            oos_trades, oos_equity_curve, deflated_sharpe_ratio = self.walk_forward_optimization(
                strat = strat,
                backtest_data = backtest_data,
                is_start_i = is_start_i,
                is_end_i = is_end_i,
                oos_start_i = oos_start_i,
                oos_end_i = oos_end_i,
                starting_equity = starting_equity
            )

            tr = 1 + ((oos_equity_curve['equity'].iloc[-1] - oos_equity_curve['equity'].iloc[0]) / oos_equity_curve['equity'].iloc[0])
            
            print('*** Num. Trades: {}'.format(len(oos_trades)))
            print()
            print('*** Avg. Trade: {}'.format(round(oos_trades['pnl_pct'].mean(), 4)))
            print()
            print('*** Total Return: {}'.format(tr))
            print()

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
                    # UPSERT backtest trades into Redshift 
                    if len(oos_trades) > 0:
                        self.__upsert_into_redshift_table(
                            table = 'backtest_trades', 
                            insert_str = insert_str, 
                            cursor = cursor, 
                            conn = conn,
                            start_date = oos_equity_curve.index[0],
                            end_date = oos_equity_curve.index[-1]
                        )

                    # UPSERT backtest equity curve into Redshift
                    self.__upsert_into_redshift_table(
                        table = 'backtest_equity_curves',
                        insert_str = insert_str_2,
                        cursor = cursor,
                        conn = conn,
                        start_date = oos_equity_curve.index[0],
                        end_date = oos_equity_curve.index[-1]
                    )

            equity_curves.append(oos_equity_curve)
            trades.append(oos_trades)
            price_data.append(backtest_data.iloc[oos_start_i:oos_end_i])
            deflated_sharpe_ratios.append(deflated_sharpe_ratio)
            
            start += out_of_sample_size

        equity_curves = pd.concat(equity_curves).sort_index()
        trades = pd.concat(trades, ignore_index = True)
        price_data = pd.concat(price_data).sort_index()

        return equity_curves, trades, price_data, deflated_sharpe_ratios
                                    
    def execute(self) -> None:
        """
        Runs a walk-forward optimization on each (token, Strategy) combination, where token
        is a CoinAPI token and Strategy is a trading strategy in backtest/strategies. Performance 
        metrics are then calculated on the combined out-of-sample equity curve, trade, and 
        price data output from the walk-forward optimization and logged to Redshift for further 
        dashboarding/analysis.

        Parameters:
        -----------
        None

        Returns:
        --------
        None
        """

        with redshift_connector.connect(
            host = 'project-poseidon.cpsnf8brapsd.us-west-2.redshift.amazonaws.com',
            database = 'token_price',
            user = 'administrator',
            password = 'Free2play2'
        ) as conn:
            with conn.cursor() as cursor:
                # Get top 50 unique tokens in Redshift by average daily volume in
                # the past 30 days since last candle
                query = """
                WITH num_days_data AS (
                    SELECT 
                        asset_id_base,
                        asset_id_quote,
                        exchange_id,
                        COUNT(*) / 24.0 AS num_days
                    FROM coinapi.price_data_1h
                    WHERE 
                        asset_id_base NOT LIKE '%USD%'
                    GROUP BY 
                        asset_id_base,
                        asset_id_quote,
                        exchange_id
                ),
                token_history_len_rank AS (
                    SELECT
                        asset_id_base,
                        asset_id_quote,
                        exchange_id,
                        ROW_NUMBER() OVER (
                            PARTITION BY asset_id_base
                            ORDER BY num_days DESC
                        ) AS rank
                    FROM num_days_data
                ),
                tokens_w_longest_history AS (
                    SELECT
                        asset_id_base,
                        asset_id_quote,
                        exchange_id
                    FROM token_history_len_rank
                    WHERE 
                        rank = 1
                ),
                last_month AS (
                    SELECT  
                        asset_id_base,
                        asset_id_quote,
                        exchange_id,
                        MAX(time_period_start) - INTERVAL '30 DAYS' AS start_date
                    FROM token_price.coinapi.price_data_1h
                    WHERE 
                        asset_id_base || '_' || asset_id_quote || '_' || exchange_id IN (
                            SELECT 
                                asset_id_base || '_' || asset_id_quote || '_' || exchange_id
                            FROM tokens_w_longest_history
                        )
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
                    ON  o.asset_id_base = l.asset_id_base AND
                        o.asset_id_quote = l.asset_id_quote AND
                        o.exchange_id = l.exchange_id 
                WHERE
                    time_period_start >= start_date
                GROUP BY o.asset_id_base, o.asset_id_quote, o.exchange_id
                ORDER BY AVG(volume_traded / (1 / price_close)) * 24 DESC
                LIMIT 50
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

                oos_equity_curves, oos_trades, oos_price_data, deflated_sharpe_ratios = self.execute_wfo(
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

                deflated_sharpe_ratios = [round(dsr, 4) for dsr in deflated_sharpe_ratios]
                deflated_sharpe_ratios = json.dumps(deflated_sharpe_ratios)

                performance_metrics['deflated_sharpe_ratios'] = deflated_sharpe_ratios
                performance_metrics = json.dumps(performance_metrics, default = BackTester.__serialize_json_data)
                performance_metrics = performance_metrics.replace('NaN', 'null').replace('Infinity', 'null')

                insert_str = """
                ('{}', '{}', '{}', '{}', '{}', '{}')
                """.format(
                    strat.indicator_factory_dict['class_name'],
                    base + '_' + quote + '_' + exchange,
                    oos_price_data.index[0],
                    oos_price_data.index[-1],
                    performance_metrics,  
                    self.optimization_metric
                )

                self.__upsert_into_redshift_table(
                    table = 'backtest_results',
                    insert_str = insert_str,
                    cursor = cursor,
                    conn = conn,
                    start_date = oos_equity_curves.index[0],
                    end_date = oos_equity_curves.index[-1]
                )

if __name__ == '__main__': 

    # Set backtest parameters

    backtest_params = {
        'init_cash': 10_000, 
        'fees': 0.00295, 
        'sl_stop': 0.2,
        'sl_trail': True,
        'size': 0.05,
        'size_type':2
    }

    # Initialize a BackTester instance w/ the intended strategies to optimize,
    # an optimization metric to find the best combination of strategy parameters,
    # and a dictionary of backtest hyperparameters

    b = BackTester(
        strategies = [BollingerBands, MACrossOver],
        optimization_metric = 'Sharpe Ratio',
        backtest_params = backtest_params
    )

    # Execute a walk-forward optimization across all strategies
    # and log the results to Redshift

    backtest_start = time.time()
    b.execute()
    backtest_end = time.time()

    print()
    print('Total Time Elapsed: {} mins'.format(round(abs(backtest_end - backtest_start) / 60.0, 2)))