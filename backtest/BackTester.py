#################################
# ADDING PACKAGE TO PYTHON PATH #
#################################
import sys
import os

#################################
#             MISC              #
#################################
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', UserWarning)
warnings.simplefilter('ignore', RuntimeWarning)

import pandas as pd
import numpy as np
import json
import duckdb

###############################################
#      TRADING STRATEGIES / BACKTESTING       #
###############################################
from .core.wfo.walk_forward_optimization import WalkForwardOptimization
from .core.performance.performance_metrics import *

class BackTester:

    # List of tokens to backtest
    TOKENS_TO_BACKTEST = [
        'BTC_USD_COINBASE', 'ETH_USD_COINBASE'
    ]

    def __init__(
        self,  
        strategies: list,                  
        optimization_metric: str = 'Sortino Ratio'                
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
        """

        self.optimization_metric = optimization_metric
        self.strategies = strategies

        self.conn = duckdb.connect(
            database = '/Users/louisspencer/Desktop/Trading-Bot-Data-Pipelines/data/database.db',
            read_only = False
        )

    def __serialize_json_data(obj):
        """
        Converts obj into a form that can be JSON serialized.

        Parameters:
        -----------
        obj : pd.Timedelta or int or float
            A value of a JSON dictionary that needs to be converted
            into a serializable format.

        Returns:
        --------
            Result of converting obj into a JSON serializable format.
        """

        if isinstance(obj, pd.Timedelta):
            return obj.total_seconds()

        elif isinstance(obj, np.integer):
            return int(obj)

        elif isinstance(obj, np.floating):
            return float(obj)

        elif isinstance(obj, dict):
            return {k: BackTester.__serialize_json_data(v) for k, v in obj.items()}

        elif isinstance(obj, list):
            return obj

        elif isinstance(obj, np.ndarray):
            return obj.tolist()

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
        # Query to fetch OHLCV data for a token & exchange of interest
        query = """
        SELECT 
            time_period_end,
            price_open,
            price_high,
            price_low,
            price_close,
            volume_traded
        FROM market_data.price_data_1m
        WHERE 
            asset_id_base = '{}' AND
            asset_id_quote = '{}' AND
            exchange_id = '{}'
        ORDER BY time_period_end ASC
        """.format(base, quote, exchange)

        # Execute query on DuckDB and return result as a DataFrame
        df = self.conn.sql(query).df().set_index('time_period_end').astype(float)
        
        # Fill in any gaps in data with last seen value
        df = df.asfreq(freq = 'min', method = 'ffill')

        return df
            
    def __upsert_into_redshift_table(
        self,
        symbol_id: str,
        strat: str,
        table: str, 
        insert_str: str
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
        
        if table == 'backtest_results':
            upsert_query = f"""
            INSERT OR REPLACE INTO backtest.{table} VALUES {insert_str} 
            """
            self.conn.sql(upsert_query)

        elif table == 'backtest_equity_curves':
            upsert_query = f"""
            INSERT OR REPLACE INTO backtest.{table} VALUES {insert_str} 
            """
            self.conn.sql(upsert_query)
        
        elif table == 'backtest_trades':
            delete_query = f"""
            DELETE FROM backtest.{table}
            WHERE 
                symbol_id = '{symbol_id}' AND 
                strat = '{strat}'
            """
            insert_query = f"""
            INSERT INTO backtest.{table} VALUES {insert_str}
            """

            self.conn.sql(delete_query)
            self.conn.sql(insert_query)

        self.conn.commit()

    def __walk_forward_optimization(
        self, 
        strat, 
        backtest_data: pd.DataFrame,
        is_start_i: int, 
        is_end_i: int, 
        oos_start_i: int, 
        oos_end_i: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        
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

        Returns:
        --------
        DataFrame:
            DataFrame containing the trades from the out-of-sample backtest.

        DataFrame:
            DataFrame containing the equity curve from the out-of-sample backtest.
        """
                
        wfo = WalkForwardOptimization(
            strategy = strat,
            backtest_data = backtest_data,
            is_start_i = is_start_i,
            is_end_i = is_end_i,
            oos_start_i = oos_start_i,
            oos_end_i = oos_end_i,
            optimization_metric = self.optimization_metric
        )

        optimal_params, optimal_is_portfolio = wfo.optimize()
        oos_trades, oos_equity_curve = wfo.walk_forward(optimal_params)

        return oos_trades, oos_equity_curve, optimal_is_portfolio

    def __execute_wfo(
        self, 
        base: str,
        quote: str,   
        exchange: str,
        strat,       
        in_sample_size: int, 
        out_of_sample_size: int
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

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

        is_sharpes = []
        oos_sharpes = []

        starting_equity = strat.backtest_params['init_cash']

        while start + in_sample_size + out_of_sample_size <= len(backtest_data):
            
            print('Progress: {} / {} days...'.format(int((start + in_sample_size) / (24 * 60)), int(len(backtest_data) / (24 * 60))))
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

            oos_trades, oos_equity_curve, optimal_is_portfolio = self.__walk_forward_optimization(
                strat = strat,
                backtest_data = backtest_data,
                is_start_i = is_start_i,
                is_end_i = is_end_i,
                oos_start_i = oos_start_i,
                oos_end_i = oos_end_i
            )

            is_sharpe = optimal_is_portfolio.returns_acc.sharpe_ratio()
            oos_sharpe = sharpe_ratio(oos_equity_curve['equity'])

            is_sharpes.append(is_sharpe)
            oos_sharpes.append(oos_sharpe)

            tr = round(1 + ((oos_equity_curve['equity'].iloc[-1] - oos_equity_curve['equity'].iloc[0]) / oos_equity_curve['equity'].iloc[0]), 3)
            
            print('*** Num. Trades: {}'.format(len(oos_trades)))
            print()
            print('*** Avg. Trade: {}'.format(round(oos_trades['pnl_pct'].mean(), 3)))
            print()
            print('*** Total Return: {}'.format(tr))
            print()
            print()
            
            starting_equity = oos_equity_curve.iloc[-1]['equity']
            strat.backtest_params['init_cash'] = starting_equity
            
            equity_curves.append(oos_equity_curve)
            trades.append(oos_trades)
            price_data.append(backtest_data.iloc[oos_start_i:oos_end_i])
            
            start += out_of_sample_size

        if len(equity_curves) == 0 or len(trades) == 0 or len(price_data) == 0:
            return None, None, None

        equity_curves = pd.concat(equity_curves).sort_index()
        trades = pd.concat(trades, ignore_index = True)
        price_data = pd.concat(price_data).sort_index()
        
        return equity_curves, trades, price_data, is_sharpes, oos_sharpes
                                    
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
            
        for strat in self.strategies:
            for token in self.TOKENS_TO_BACKTEST:
                base, quote, exchange = token.split('_')

                print()
                print('Backtesting the {} strategy on {}'.format(
                strat.indicator_factory_dict['class_name'],
                base + '_' + quote + '_' + exchange
                ))
                print()

                # Execute walk-forward optimization
                oos_equity_curves, oos_trades, oos_price_data, is_sharpes, oos_sharpes = self.__execute_wfo(
                    base = base,
                    quote = quote, 
                    exchange = exchange,
                    strat = strat,
                    in_sample_size = 60 * 24 * 365,
                    out_of_sample_size = 60 * 24 * 90
                )

                # Reset starting equity for next token
                strat.backtest_params['init_cash'] = 10_000

                # If the walk-forward optimization failed, skip to the next token
                if (oos_equity_curves is None) or (oos_trades is None) or (oos_price_data is None):
                    continue

                oos_trades['entry_date'] = pd.to_datetime(oos_trades['entry_date'])
                oos_trades['exit_date'] = pd.to_datetime(oos_trades['exit_date'])
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

                performance_metrics['is_sharpe'] = is_sharpes
                performance_metrics['oos_sharpe'] = oos_sharpes
                
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
                    symbol_id = base + '_' + quote + '_' + exchange,
                    strat = strat.indicator_factory_dict['class_name'],
                    table = 'backtest_results',
                    insert_str = insert_str_backtest_result
                )
                self.__upsert_into_redshift_table(
                    symbol_id = base + '_' + quote + '_' + exchange,
                    strat = strat.indicator_factory_dict['class_name'],
                    table = 'backtest_trades',
                    insert_str = insert_str_backtest_trades
                )
                self.__upsert_into_redshift_table(
                    symbol_id = base + '_' + quote + '_' + exchange,
                    strat = strat.indicator_factory_dict['class_name'],
                    table = 'backtest_equity_curves',
                    insert_str = insert_str_backtest_equity_curve
                )
