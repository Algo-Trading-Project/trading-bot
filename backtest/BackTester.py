#################################
# ADDING PACKAGE TO PYTHON PATH #
#################################
import sys
import os
# sys.path.append(os.getcwd())

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

from typing import Union

###############################################
#      TRADING STRATEGIES / BACKTESTING       #
###############################################
from .core.wfo.walk_forward_optimization import WalkForwardOptimization
from .core.simulation.pbo import pbo
from .core.performance.performance_metrics import *
from .core.performance.performance_metrics_pbo import sortino

from .strategies.ma_crossover import MACrossOver
from .strategies.bollinger_bands import BollingerBands
from .strategies.linear_regression import LogisticRegressionStrategy

class BackTester:

    # List of tokens to backtest
    TOKENS_TO_BACKTEST = [
        'BTC_USD_COINBASE', 'ETH_USD_COINBASE', 'ADA_USDT_BINANCE',
        'ALGO_USD_COINBASE', 'ATOM_USDT_BINANCE', 'BCH_USD_COINBASE',
        'BNB_USDC_BINANCE', 'DOGE_USDT_BINANCE', 'ETC_USD_COINBASE',
        'FET_USDT_BINANCE', 'FTM_USDT_BINANCE', 'HOT_USDT_BINANCE',
        'IOTA_USDT_BINANCE', 'LINK_USD_COINBASE', 'LTC_USD_COINBASE',
        'MATIC_USDT_BINANCE',
    ]

    def __init__(
        self,  
        strategies: list,                  
        optimization_metric: str = 'Sharpe Ratio',                 
        backtest_params: dict = {'init_cash':100_000, 'fees':0.005}
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

        backtest_params : dict, default = {'init_cash':100_000, 'fees':0.005}
            Dictionary containing miscellaneous parameters to configure the
            backtest.

        """

        self.optimization_metric = optimization_metric
        self.backtest_params = backtest_params
        self.strategies = strategies

    def __serialize_json_data(obj: Union[pd.Timedelta, int, float]) -> Union[int, float]:
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
        else:
            raise TypeError('Object of type {} is not JSON serializable'.format(type(obj)))
             
    def __fetch_OHLCV_df_from_redshift(
        self, 
        base: str, 
        quote: str, 
        exchange: str
    ) -> pd.DataFrame:
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

                # Fill in any gaps in data with last seen value
                df = df.asfreq(freq = 'H', method = 'ffill')

                return df
            
    def __upsert_into_redshift_table(
        self, 
        table: str, 
        insert_str: str, 
        cursor: redshift_connector.Cursor,
        conn: redshift_connector.Connection
    ) -> None:
        
        """
        Performs an upsert on the specified Redshift table in the trading_bot database.

        Parameters
        ----------
        table : str
            Redshift table to upsert into

        insert_str : str
            Rows to upsert into Redshift represented as a string

        cursor : Cursor
            redshift_connector Cursor

        conn : Connection
            redshift_connector Connection

        Returns
        -------
        None
        """

        begin_transaction = """
        BEGIN TRANSACTION;
        """

        create_staging_table = """
        CREATE TABLE IF NOT EXISTS trading_bot.eth.stg_{table} (LIKE trading_bot.eth.{table})
        """.format(table = table)

        insert_into_staging_table = """
        INSERT INTO trading_bot.eth.stg_{table} VALUES {insert_str}
        """.format(table = table, insert_str = insert_str)

        delete_from_target_table = """
        DELETE FROM trading_bot.eth.{table} 
        USING trading_bot.eth.stg_{table}
        WHERE
            {table}.symbol_id = stg_{table}.symbol_id AND
            {table}.strat = stg_{table}.strat
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

    def __walk_forward_optimization(
        self, 
        strat, 
        backtest_data: pd.DataFrame,
        is_start_i: int, 
        is_end_i: int, 
        oos_start_i: int, 
        oos_end_i: int, 
        starting_equity: float
    ) -> (pd.DataFrame, pd.DataFrame):
        
        """
        Optimizes the parameters of a Strategy (strat) on the in-sample data and performs 
        a backtest on the out-of-sample data w/ the optimized parameters. Returns the 
        out-of-sample trades and out-of-sample equity curve so that it can be logged to 
        Redshift for further dashboarding/analysis.

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

        optimal_params, in_sample_portfolio = wfo.optimize()
        oos_trades, oos_equity_curve = wfo.walk_forward(optimal_params)

        return oos_trades, oos_equity_curve

    def __execute_wfo(
        self, 
        base: str,
        quote: str,   
        exchange: str,
        strat,       
        in_sample_size: int, 
        out_of_sample_size: int
    ) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):

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
        oos_returns = []

        starting_equity = self.backtest_params['init_cash']

        while start + in_sample_size + out_of_sample_size <= len(backtest_data):
            
            # print('\rProgress: {} / {} days...'.format(int((start + in_sample_size) / 24), int(len(backtest_data) / 24), end = '', flush = True))

            # print('*** Starting Equity: ', starting_equity)
            # print()

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

            oos_trades, oos_equity_curve = self.__walk_forward_optimization(
                strat = strat,
                backtest_data = backtest_data,
                is_start_i = is_start_i,
                is_end_i = is_end_i,
                oos_start_i = oos_start_i,
                oos_end_i = oos_end_i,
                starting_equity = starting_equity
            )

            tr = round(1 + ((oos_equity_curve['equity'].iloc[-1] - oos_equity_curve['equity'].iloc[0]) / oos_equity_curve['equity'].iloc[0]), 3)
            
            starting_equity = oos_equity_curve.iloc[-1]['equity']
            
            equity_curves.append(oos_equity_curve)
            trades.append(oos_trades)
            price_data.append(backtest_data.iloc[oos_start_i:oos_end_i])
            oos_returns.append(oos_equity_curve['equity'].pct_change().fillna(0))
            
            start += out_of_sample_size

        if len(equity_curves) == 0 or len(trades) == 0 or len(price_data) == 0:
            return None, None, None

        equity_curves = pd.concat(equity_curves).sort_index()
        trades = pd.concat(trades, ignore_index = True)
        price_data = pd.concat(price_data).sort_index()
        
        # Combine the OOS returns into a matrix
        oos_returns_matrix = pd.concat(oos_returns, axis = 1).fillna(0).values

        return equity_curves, trades, price_data, oos_returns_matrix
                                    
    def execute(self):
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
            database = 'trading_bot',
            user = 'administrator',
            password = 'Free2play2'
        ) as conn:
            with conn.cursor() as cursor:

                for strat in self.strategies:
                    for token in self.TOKENS_TO_BACKTEST:
                        base, quote, exchange = token.split('_')

                        print()
                        print('Backtesting the {} strategy on {}'.format(
                        strat.indicator_factory_dict['class_name'],
                        base + '_' + 'USD' + '_' + exchange
                        ))

                        # Execute walk-forward optimization
                        oos_equity_curves, oos_trades, oos_price_data, oos_returns_matrix = self.__execute_wfo(
                            base = base,
                            quote = quote, 
                            exchange = exchange,
                            strat = strat,
                            in_sample_size = 24 * 30 * 2,
                            out_of_sample_size = 24 * 30 * 2
                        )

                        # Perform PBO analysis for each (Strategy, Token) combination
                        pbo_results = pbo(
                            M=oos_returns_matrix,
                            S=6,
                            metric_func=sortino,  # Define or import your metric function
                            threshold=0,
                            n_jobs=1,
                            verbose=True,
                            plot=False  # No plot here, as it will be done on the dashboard
                        )._asdict()

                        # If the walk-forward optimization failed, skip to the next token
                        if (oos_equity_curves is None) or (oos_trades is None) or (oos_price_data is None):
                            continue

                        oos_trades['entry_date'] = oos_trades['entry_date'].astype(str)
                        oos_trades['exit_date'] = oos_trades['exit_date'].astype(str)
                        oos_trades_dict = oos_trades.to_dict(orient = 'records')
                        
                        insert_str_backtest_trades = ''

                        for trade in oos_trades_dict:
                            pnl = float(str(trade['pnl'])[:38])
                            pnl_pct = float(str(trade['pnl_pct'])[:38])

                            insert_str_backtest_trades += """('{}', '{}', '{}', '{}', '{}', '{}', '{}'), """.format(
                                base + '_' + quote + '_' + exchange, 
                                strat.indicator_factory_dict['class_name'],
                                trade['entry_date'],
                                trade['exit_date'],
                                pnl,
                                pnl_pct,
                                trade['is_long']
                            )

                        insert_str_backtest_trades = insert_str_backtest_trades[:-2]

                        insert_str_backtest_equity_curve = ''
                        
                        for i in range(len(oos_equity_curves)):
                            date = oos_equity_curves.index[i]
                            equity = oos_equity_curves['equity'].iloc[i]

                            insert_str_backtest_equity_curve += """('{}', '{}', '{}', {}), """.format(
                                base + '_' + quote + '_' + exchange,
                                strat.indicator_factory_dict['class_name'],
                                date,
                                equity
                            )

                        insert_str_backtest_equity_curve = insert_str_backtest_equity_curve[:-2]

                        performance_metrics = calculate_performance_metrics(
                            oos_equity_curves, 
                            oos_trades, 
                            oos_price_data
                        ).to_dict(orient = 'records')[0]
                        
                        # Serialize PBO results and add them to performance metrics
                        # performance_metrics['pbo_results'] = json.dumps(pbo_results._asdict(), default=BackTester.__serialize_json_data)

                        performance_metrics = json.dumps(performance_metrics, default = BackTester.__serialize_json_data)
                        performance_metrics = performance_metrics.replace('NaN', 'null').replace('Infinity', 'null')

                        insert_str_backtest_result = """
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
                            insert_str = insert_str_backtest_result,
                            cursor = cursor,
                            conn = conn
                        )
                        self.__upsert_into_redshift_table(
                            table = 'backtest_trades',
                            insert_str = insert_str_backtest_trades,
                            cursor = cursor,
                            conn = conn
                        )
                        self.__upsert_into_redshift_table(
                            table = 'backtest_equity_curves',
                            insert_str = insert_str_backtest_equity_curve,
                            cursor = cursor,
                            conn = conn
                        )

        return pbo_results

if __name__ == '__main__': 

    # Set backtest parameters
    
    # init_cash - Initial cash
    # fees      - Comission percent
    # sl_stop   - Stop-loss percent
    # sl_trail  - Indicate whether or not want a trailing stop-loss
    # tp_stop   - Take-profit percent
    # size      - Percentage of capital to use for each trade
    # size_type - Indicates the 'size' parameter represents a percent

    backtest_params = {
        'init_cash': 100_000,
        'fees': 0.00295,
        'sl_stop': 0.1,
        'tp_stop': 0.1,
        'sl_trail': True,
        'size': 0.05,
        'size_type': 2 # e.g. 2 if 'size' is 'Fixed Percent' and 0 otherwise
    }

    # Initialize a BackTester instance w/ the intended strategies to backtest,
    # a performance metric to optimize on, and a dictionary of backtest hyperparameters

    b = BackTester(
        strategies = [MACrossOver, BollingerBands],
        optimization_metric = 'Sortino Ratio',
        backtest_params = backtest_params
    )

    # Execute a walk-forward optimization across all strategies
    # and log the results to Redshift
    
    backtest_start = time.time()
    pbo_results = b.execute()
    backtest_end = time.time()

    print()
    print('Total Time Elapsed: {} mins'.format(round(abs(backtest_end - backtest_start) / 60.0, 2)))