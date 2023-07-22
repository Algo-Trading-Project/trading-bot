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

from strategies.MACrossOver import MACrossOver
from strategies.ARIMA import ARIMAStrat

class BackTester:

    def __init__(self, 
                 strategies,
                 optimization_metric = 'Sharpe Ratio',
                 backtest_params = {'init_cash':10_000, 'fees':0.005}
                 ):
        """
        High-level orchestration engine for performing walk-forward optimization on a set of
        strategies over a set of tokens, utilizing the vectorbt package for efficient
        parameter optimization & metric calculation.  The results of the walk-forward optimizations
        are logged to Redshift for further dashboarding/analysis.

        Parameters:
        -----------
        strategies : list
            List of Strategy classes from src/backtest/strategies to backtest.

        optimization_metric : str, default = 'Sharpe Ratio'
            Performance metric to maximize/minimize during the paramater optimization of the 
            in-sample periods. The full list of metrics available to use can be found in the file 
            src/backtest/core/walk_forward_optimization.py in the metric_map dictionary in the __init__ method.

        backtest_params : dict, default = {'init_cash':10_000, 'fees':0.005}
            Dictionary containing miscellaneous parameters to configure the
            backtest.

        """

        self.optimization_metric = optimization_metric
        self.backtest_params = backtest_params
        self.strategies = strategies

    def __serialize_json_data(obj):
        """
        Converts obj into a form that can be JSON serialized.

        Parameters:
        -----------
        obj : Any
            A value of a JSON dictionary that needs to be converted
            into a serializable format.

        Returns:
        --------
        (float or int or None):
            Result of converting obj into a JSON serializable format.

        """
        if isinstance(obj, pd.Timedelta):
            return obj.total_seconds()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
             
    def __fetch_OHLCV_df_from_redshift(self, base, quote, exchange):
        """
        Queries OHLCV data for {exchange}_SPOT_{base}_{quote} CoinAPI pair stored in
        Project Poseidon's Redshift cluster. Returns the queried data as a DataFrame 
        indexed by timestamp.

        Parameters:
        -----------
        base : str 
            CoinAPI asset_id_base of the pair being backtested.

        quote : str
            CoinAPI asset_id_quote of the pair being backtested.

        exchange : str
            CoinAPI exchange_id of the pair being backtested.

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

    def walk_forward_optimization(self, strat, backtest_data, is_start_i, is_end_i,
                 oos_start_i, oos_end_i, starting_equity):
        """
        Optimizes the parameters of a Strategy (strat) on the in-sample data and performs 
        a backtest on the out-of-sample data w/ the optimized parameters. Returns the 
        out-of-sample trades, out-of-sample equity curve, and the deflated sharpe ratio (DSR) 
        so that it can be logged to Redshift for further dashboarding/analysis.

        Parameters:
        -----------
        strat: Strategy class in src/backtest/strategies
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
            DataFrame containing the trades from the walk-forward backtest.

        DataFrame:
            DataFrame containing the equity curve from the walk-forward backtest.

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
        
        estimated_sharpe = portfolio.returns_acc.sharpe_ratio().max() / np.sqrt(annualization_factor)
        sharpe_variance = portfolio.returns_acc.sharpe_ratio().var() / annualization_factor
        nb_trials = len(list(itertools.product(*strat.optimize_dict.values())))
        backtest_horizon = is_end_i - is_start_i + 1
        skew = portfolio.loc[portfolio.returns_acc.sharpe_ratio().idxmax()].returns().skew()
        kurtosis = portfolio.loc[portfolio.returns_acc.sharpe_ratio().idxmax()].returns().kurt()

        deflated_sharpe_ratio = compute_deflated_sharpe_ratio(
            estimated_sharpe = estimated_sharpe,
            sharpe_variance = sharpe_variance,
            nb_trials = nb_trials,
            backtest_horizon = backtest_horizon,
            skew = skew, 
            kurtosis = kurtosis
        )

        return oos_trades, oos_equity_curve, deflated_sharpe_ratio
    
    def orchestrate_wfo(self, base, quote, exchange, strat,
                        in_sample_size, out_of_sample_size):
        """
        Orchestrates the walk-forward optimization of an arbitrary trading strategy (strat) 
        on the {exchange}_SPOT_{base}_{quote} CoinAPI pair. Aggregates and returns the equity curves,
        trades, and token prices across all out-of-sample backtests to calculate the overall out-of-sample
        performance metrics.

        Parameters:
        -----------
        base : str 
            CoinAPI asset_id_base of the pair being backtested.

        quote : str
            CoinAPI asset_id_quote of the pair being backtested.

        exchange : str
            CoinAPI exchange_id of the pair being backtested.

        strat: Strategy class in src/backtest/strategies
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

        float:
            The average deflated Sharpe ratio (DSR) across all in-sample
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
            print('Starting equity: ', starting_equity)
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
            price_data.append(backtest_data.iloc[oos_start_i:oos_end_i])
            deflated_sharpe_ratios.append(deflated_sharpe_ratio)
            
            start += out_of_sample_size

        equity_curves = pd.concat(equity_curves).sort_index()
        trades = pd.concat(trades, ignore_index = True)
        price_data = pd.concat(price_data).sort_index()

        return equity_curves, trades, price_data, deflated_sharpe_ratios
                                    
    def execute(self):
        """
        Runs a walk-forward optimization on each (token, Strategy) combination, where token
        is OHLCV data for a CoinAPI token stored in Redshift and Strategy is a trading strategy
        in src/backtest/strategies. Performance metrics are then calculated on the combined
        out-of-sample equity curve, trade, and price data output from the walk-forward optimization
        and logged to Redshift for further dashboarding/analysis

        Parameters:
        -----------
        None

        Returns:
        ________
        None
        """

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
                SELECT 
                DISTINCT
                    asset_id_base,
                    asset_id_quote,
                    exchange_id
                FROM token_price.coinapi.price_data_1h
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

                oos_equity_curves, oos_trades, oos_price_data, deflated_sharpe_ratios = self.orchestrate_wfo(
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

    # Set initial capital of backtest to $10,000 and set the trading
    # fees to the average 'taker' fee across many well-known centralized
    # exchanges, which is 0.295 percent (or 0.00295)

    backtest_params = {'init_cash': 10_000, 'fees': 0.00295}

    # Initialize a BackTester instance w/ the intended strategies to optimize,
    # an optimization metric to find the best combination of strategy parameters,
    # and a dictionary of backtest hyperparameters

    b = BackTester(
        strategies = [MACrossOver],
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