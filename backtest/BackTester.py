#################################
#             MISC              #
#################################
import warnings

import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', UserWarning)
warnings.simplefilter('ignore', RuntimeWarning)

import duckdb
import pickle
import json
import calendar

from utils.db_utils import QUERY
from backtest.strategies.single_asset.base_strategy import BaseStrategy
from backtest.strategies.portfolio.base_strategy import BasePortfolioStrategy

###############################################
#                BACKTESTING                  #
###############################################
from backtest.core.walk_forward_optimization import WalkForwardOptimization
from backtest.core.performance_metrics import *

"""
This file contains the BackTester class, which is responsible for coordinating the walk-forward 
optimization of a set of strategies over a set of tokens. The results of the walk-forward optimizations 
are logged to our databases for further dashboarding/analysis. The BackTester class is instantiated with a list of strategies, 
an optimization metric to maximize/minimize during the in-sample parameter optimizations, a resample period to 
resample the OHLCV data to a desired timeframe (e.g. 1min, 30min, 1hr) before backtesting, and a boolean flag 
indicating whether to use dollar bars instead of time bars for the backtest.
"""

class BackTester:

    def __init__(
        self,  
        strategies: list,                  
        resample_period: str = '1min',
        start_date: str = '2021-04-01',
        end_date: str = '2022-12-31'          
    ):
        """
        Coordinates the walk-forward optimization of a set of strategies over a set of tokens, 
        utilizing the vectorbt package for efficient parameter optimization. The results of the walk-forward
        optimizations are logged for further dashboarding/analysis.

        Parameters:
        -----------
        strategies : list
            List of Strategy classes from backtest/strategies to backtest.

        resample_period : str, default = '1min'
            Period to resample the OHLCV data to before backtesting.

        use_dollar_bars : bool, default = False
            Whether to use dollar bars instead of time bars for the backtest.

        start_date : str, default = '2021-04-01'
            Start date of the backtest.

        end_date : str, default = '2022-12-31'
             End date of the backtest.
        """
        print()
        print('Initializing BackTester...')
        print()

        # Load the price data for the universe
        price_data = pd.read_parquet('/Users/louisspencer/Desktop/Trading-Bot/data/ml_features.parquet')[['open_spot', 'high_spot', 'low_spot', 'close_spot', 'open_futures', 'high_futures', 'low_futures', 'close_futures', 'time_period_end', 'symbol_id']]
        price_data.columns = ['open', 'high', 'low', 'close', 'open_futures', 'high_futures', 'low_futures', 'close_futures', 'time_period_open', 'symbol_id']
        price_data['time_period_open'] = pd.to_datetime(price_data['time_period_open']) - pd.Timedelta(days=1) 
        price_data = price_data[price_data['symbol_id'].str.contains('USDT')]

        # Pivot the data to get the asset universe close, high, and low prices
        self.universe = (
            price_data
            .pivot_table(
                index = 'time_period_open',
                columns = 'symbol_id',
                values = ['open', 'high', 'low', 'close', 'open_futures', 'high_futures', 'low_futures', 'close_futures'],
                dropna = False
            )
        )
        self.universe = self.universe.sort_index()

        self.strategies = strategies
        self.resample_period = resample_period
        self.use_dollar_bars = None
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)

        self.conn = duckdb.connect(
            database = '~/LocalData/database.db',
            read_only = False
        )

    def __serialize_json_data(obj):
        if isinstance(obj, pd.Timedelta):
            return obj.total_seconds()

        elif isinstance(obj, np.integer):
            return int(obj)

        elif isinstance(obj, np.floating):
            return float(obj)

        elif isinstance(obj, dict):
            return {k: BackTester.__serialize_json_data for k, v in obj.items()}

        elif isinstance(obj, list):
            return obj

        elif isinstance(obj, np.ndarray):
            return obj.tolist()

        else:
            raise TypeError('Object of type {} is not JSON serializable'.format(type(obj)))
             
    def __fetch_OHLCV_df(
        self, 
        base: str, 
        quote: str, 
        exchange: str,
        strat,
        use_dollar_bars: bool = False
    ) -> pd.DataFrame:
        if strat.indicator_factory_dict['class_name'] == 'MLStrategy':
            query = f"""
            SELECT 
                origin_time,
                s.asset_id_base || '_' || s.asset_id_quote || '_' || s.exchange_id AS symbol_id,
                s.open AS open,
                f.open AS open_futures,
                s.high AS high,
                f.high AS high_futures,
                s.low AS low,
                f.low AS low_futures,
                s.close AS close,
                f.close AS close_futures,
            FROM market_data.ohlcv_1m s LEFT JOIN market_data.ohlcv_1m_futures f ON
                s.origin_time = f.origin_time AND
                s.asset_id_base = f.asset_id_base AND
                s.asset_id_quote = f.asset_id_quote AND
                s.exchange_id = f.exchange_id
            """
        else:
            query = """
            SELECT 
                origin_time,
                open,
                high,
                low,
                close,
                volume,
                trades,
                asset_id_base || '_' || asset_id_quote || '_' || exchange_id AS symbol_id
            FROM market_data.ohlcv_1m
            WHERE 
                asset_id_base = '{}' AND
                asset_id_quote = '{}' AND
                exchange_id = '{}'
            ORDER BY origin_time ASC
            """.format(base, quote, exchange)

        # Execute the query on DuckDB and return result as a DataFrame
        df = self.conn.sql(query).df().set_index('origin_time').asfreq('1min', method = 'ffill')

        # Interpolate symbol_id with the most common value
        df['symbol_id'] = df['symbol_id'].fillna(df['symbol_id'].mode().iloc[0])
        # df.index = pd.to_datetime(df.index, utc = True)

        if self.resample_period != '1min':
            # Resample the DataFrame to the specified resample_period
            df = df.resample(self.resample_period, label = 'right', closed = 'left').agg({
                'open': 'first',
                'open_futures': 'first',
                'close': 'last',
                'close_futures': 'last',
                'high': 'max',
                'high_futures': 'max',
                'low': 'min',
                'low_futures': 'min',
                # 'volume': 'sum',
                # 'trades': 'sum',
                'symbol_id': 'first'
            }).ffill(method = 'ffill')
        
        return df
            
    def __upsert_into_table(
        self,
        symbol_id: str,
        strat: BaseStrategy or BasePortfolioStrategy,
        table: str, 
        insert_str: str,
        df: pd.DataFrame = None
    ) -> None:
        strat_name = strat.indicator_factory_dict['class_name']
        
        if table == 'backtest_results':
            upsert_query = f"""
            INSERT OR REPLACE INTO backtest.{table} VALUES {insert_str} 
            """
            self.conn.sql(upsert_query)

        elif table == 'backtest_equity_curves':
            # Ensure no duplicate entries are inserted by df
            df = df.drop_duplicates(subset = ['date'])

            # Register the DataFrame as a table in DuckDB
            self.conn.register('backtest_equity_curves', df)

            # Upsert the DataFrame into DuckDB
            # using the backtest_equity_curves table
            delete_query = f"""
            DELETE FROM backtest.{table}
            WHERE 
                symbol_id = '{symbol_id}' AND 
                strat = '{strat_name}'
            """
            upsert_query = f"""
            INSERT INTO backtest.{table} (symbol_id, strat, date, equity)
            SELECT symbol_id, strat, date, equity
            FROM backtest_equity_curves
            """

            self.conn.sql(delete_query)
            self.conn.sql(upsert_query)

            # Unregister the DataFrame as a table in DuckDB
            self.conn.unregister('backtest_equity_curves')
        
        elif table == 'backtest_trades':
            if issubclass(type(strat), BasePortfolioStrategy):
                delete_query = f"""
                DELETE FROM backtest.{table}
                WHERE 
                    strat = '{strat_name}'
                """
                insert_query = f"""
                INSERT INTO backtest.{table} (strat, entry_date, exit_date, symbol_id, size, entry_fees, exit_fees, pnl, pnl_pct, is_long) VALUES {insert_str}
                """

            elif issubclass(type(strat), BaseStrategy):
                delete_query = f"""
                DELETE FROM backtest.{table}
                WHERE 
                    symbol_id = '{symbol_id}' AND 
                    strat = '{strat_name}'
                """
                insert_query = f"""
                INSERT INTO backtest.{table} (strat, entry_date, exit_date, symbol_id, size, entry_fees, exit_fees, pnl, pnl_pct, is_long) VALUES {insert_str}
                """

            else:
                raise ValueError(f'Invalid strategy class: {strat_name}')

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

        wfo = WalkForwardOptimization(
            strategy = strat,
            backtest_data = backtest_data,
            is_start_i = is_start_i,
            is_end_i = is_end_i,
            oos_start_i = oos_start_i,
            oos_end_i = oos_end_i,
            resample_period = self.resample_period
        )

        optimal_params, optimal_is_portfolio = wfo.optimize()
        oos_trades, oos_equity_curve, oos_port = wfo.walk_forward(optimal_params)

        return oos_trades, oos_equity_curve, optimal_is_portfolio, oos_port, optimal_params

    def __execute_wfo(
        self, 
        base: str = None,
        quote: str = None,  
        exchange: str = None,
        strat = None,     
        in_sample_size: int = 30,
        out_of_sample_size: int = 30
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        start = 0

        # If the strategy is a subclass of BaseStrategy, then the backtest_data is the token's OHLCV data
        if issubclass(type(strat), BaseStrategy):
            backtest_data = self.__fetch_OHLCV_df(
                base = base,
                quote = quote,
                exchange = exchange,
                strat = strat,
                use_dollar_bars = self.use_dollar_bars
            )
            backtest_data.index = pd.to_datetime(backtest_data.index)
            symbol_id = base + '_' + quote + '_' + exchange

        # If the strategy is a subclass of BasePortfolioStrategy, then the backtest_data is the token universe
        elif issubclass(type(strat), BasePortfolioStrategy):
            backtest_data = self.universe
            symbol_id = 'PORTFOLIO_UNIVERSE'

        else:
            raise ValueError(f'Invalid strategy class: {type(strat)}')

        # Only use data between the start and end dates
        backtest_data = backtest_data.loc[(backtest_data.index >= self.start_date) & (backtest_data.index <= self.end_date)]

        if backtest_data.empty:
            return None, None, None, None
        
        equity_curves = []
        trades = []
        price_data = []
        performance_metrics = []

        starting_equity = strat.backtest_params['init_cash']

        if len(backtest_data) <= in_sample_size + out_of_sample_size:
            print('Not enough data to perform walk-forward optimization for {} on {}'.format(
                strat.indicator_factory_dict['class_name'],
                symbol_id
            ))
            print()
            
            in_sample_size = len(backtest_data) // 2
            out_of_sample_size = len(backtest_data) // 2

        while start + in_sample_size + out_of_sample_size <= len(backtest_data):
            self.resample_period = self.resample_period.lower()
            if self.resample_period == '1min':
                day_normalizer = 60 * 24
            elif self.resample_period == '5min':
                day_normalizer = 12 * 24
            elif self.resample_period == '30min':
                day_normalizer = 2 * 24
            elif self.resample_period == '1h':
                day_normalizer = 24
            elif self.resample_period == '4h':
                day_normalizer = 6
            elif self.resample_period == '1d':
                day_normalizer = 1
            else:
                raise ValueError(f'Invalid resample period: {self.resample_period}')
            
            print('Progress: {} / {} days...'.format(int(start / day_normalizer), int(len(backtest_data) / day_normalizer)))
            print()

            if issubclass(type(strat), BasePortfolioStrategy):
                is_start_i = start
                # Get date associated with is_start_i
                is_start_date = backtest_data.index[is_start_i]
                # Get end of month date
                is_end_date = pd.Timestamp(month = is_start_date.month, year = is_start_date.year, day = calendar.monthrange(is_start_date.year, is_start_date.month)[1])
                
                # Get integer index of is_end_date
                is_end_i = backtest_data.index.get_loc(is_end_date)
                oos_start_i = is_end_i + 1
                # Get date associated with oos_start_i
                oos_start_date = backtest_data.index[oos_start_i]
                
                # Get end of month date
                oos_end_date = pd.Timestamp(month = oos_start_date.month, year = oos_start_date.year, day = calendar.monthrange(oos_start_date.year, oos_start_date.month)[1])

                # Get integer index of oos_end_date
                oos_end_i = backtest_data.index.get_loc(oos_end_date)

                # Print the start and end dates of the in-sample and out-of-sample periods
                # Format the dates to be human-readable
                is_start_date = is_start_date.strftime('%Y-%m-%d')
                is_end_date = is_end_date.strftime('%Y-%m-%d')
                oos_start_date = oos_start_date.strftime('%Y-%m-%d')
                oos_end_date = oos_end_date.strftime('%Y-%m-%d')

                start = oos_start_i
            else:
                is_start_i = start
                is_start_date = backtest_data.index[is_start_i].strftime('%Y-%m-%d')
                is_end_i = start + in_sample_size
                is_end_date = backtest_data.index[is_end_i].strftime('%Y-%m-%d')

                oos_start_i = is_end_i
                oos_start_date = backtest_data.index[oos_start_i].strftime('%Y-%m-%d')
                oos_end_i = start + in_sample_size + out_of_sample_size
                oos_end_date = backtest_data.index[oos_end_i].strftime('%Y-%m-%d')

                start += out_of_sample_size

            print(f'*** In-Sample Period: {is_start_date} - {is_end_date}')
            print()
            print(f'*** Out-of-Sample Period: {oos_start_date} - {oos_end_date}')
            print()
            print('*** Starting Equity: ', starting_equity)
            print()

            oos_trades, oos_equity_curve, optimal_is_portfolio, oos_port, optimal_params = self.__walk_forward_optimization(
                strat = strat,
                backtest_data = backtest_data,
                is_start_i = is_start_i,
                is_end_i = is_end_i,
                oos_start_i = oos_start_i,
                oos_end_i = oos_end_i
            )
            # Filter the oos trade to only include trades that occurred during the out-of-sample period
            oos_trades = oos_trades[
                (oos_trades['entry_date'] >= oos_start_date) &
                (oos_trades['exit_date'] <= oos_end_date)
            ].sort_values('entry_date')
            
            long_trades = oos_trades[oos_trades['is_long'] == True]
            short_trades = oos_trades[oos_trades['is_long'] == False]
            print('Long Trades:')
            print(long_trades)
            # for trade in long_trades.itertuples():
            #     print(f"Entry: {trade.entry_date}, Exit: {trade.exit_date}, PnL: {trade.pnl}, Symbol: {trade.symbol_id}, Pnl %: {trade.pnl_pct}")
            print()
            print('Short Trades:')
            print(short_trades)
            print()

            tr = (oos_trades['pnl'].sum() / starting_equity) + 1

            print()
            print('*** Num. Trades: {}'.format(len(oos_trades)))
            print()
            print('*** Total PnL: {}'.format(oos_trades['pnl'].sum()))
            print()
            print('*** Total Return: {}'.format(tr))
            print()
            print()
            
            # starting_equity = oos_equity_curve.iloc[-1]['equity']
            starting_equity *= tr
            strat.backtest_params['init_cash'] = starting_equity
            
            equity_curves.append(oos_equity_curve)
            trades.append(oos_trades)

            # Serialize the optimal in-sample portfolio and out-of-sample portfolio to a BLOB using pickle
            # to store in the backtest_results table
            performance_metrics_dict = {
                'is_start_date': is_start_date,
                'is_end_date': is_end_date,
                'oos_start_date': oos_start_date,
                'oos_end_date': oos_end_date,
                'optimal_is_portfolio': None,
                'oos_port': None,
                'optimal_params': optimal_params
            }
            performance_metrics.append(performance_metrics_dict)

            if issubclass(type(strat), BaseStrategy):
                price_data.append(backtest_data.iloc[oos_start_i:oos_end_i])
            elif issubclass(type(strat), BasePortfolioStrategy):
                pass
            else:
                raise ValueError(f'Invalid strategy class: {strat}')

        if len(equity_curves) == 0 or len(trades) == 0:
            print('Walk-forward optimization failed for {} on {}'.format(
                strat.indicator_factory_dict['class_name'],
                base + '_' + quote + '_' + exchange
            ))
            print('Equity Curves: ')
            print(equity_curves)

            print('Trades: ')
            print(trades)

            return None, None, None, None

        equity_curves = pd.concat(equity_curves).sort_index()
        trades = pd.concat(trades, ignore_index = True)

        if issubclass(type(strat), BaseStrategy):
            price_data = pd.concat(price_data).sort_index()            
        
        return equity_curves, trades, price_data, performance_metrics

    def __upload_backtest_results(
        self, 
        strat,
        oos_equity_curves, 
        oos_trades, 
        oos_price_data, 
        performance_metrics,
        base: str = None,
        quote: str = None,
        exchange: str = None
    ):
        oos_trades_dict = oos_trades.to_dict(orient = 'records')
        insert_str_backtest_trades = ''
        strategy = strat.indicator_factory_dict['class_name']

        if issubclass(type(strat), BasePortfolioStrategy):
            symbol_id = 'PORTFOLIO_UNIVERSE'
        elif issubclass(type(strat), BaseStrategy):
            symbol_id = base + '_' + quote + '_' + exchange
        else:
            raise ValueError(f'Invalid strategy class: {strat}')

        for trade in oos_trades_dict:
            token_traded = trade['symbol_id']
            entry_date = trade['entry_date']
            exit_date = trade['exit_date']
            size = trade['size']
            entry_fees = trade['entry_fees']
            exit_fees = trade['exit_fees']
            pnl = trade['pnl']
            pnl_pct = trade['pnl_pct']
            is_long = trade['is_long']

            insert_str_backtest_trades += f"""('{strategy}', '{entry_date}', '{exit_date}', '{token_traded}', {size}, {entry_fees}, {exit_fees}, {pnl}, {pnl_pct}, {is_long}), """

        insert_str_backtest_trades = insert_str_backtest_trades[:-2]
        backtest_equity_curve_df = []
        
        for i in range(len(oos_equity_curves)):
            date = oos_equity_curves.index[i]
            equity = oos_equity_curves['equity'].iloc[i]

            row_dict = {
                'symbol_id': symbol_id,
                'strat': strat.indicator_factory_dict['class_name'],
                'date': date,
                'equity': equity
            }

            backtest_equity_curve_df.append(row_dict)

        backtest_equity_curve_df = pd.DataFrame(backtest_equity_curve_df)

        performance_metrics = calculate_performance_metrics(
            oos_equity_curves, 
            oos_trades, 
            oos_price_data
        ).to_dict(orient = 'records')[0]

        performance_metrics = json.dumps(performance_metrics, default = BackTester.__serialize_json_data)
        performance_metrics = performance_metrics.replace('NaN', 'null').replace('Infinity', 'null')

        insert_str_backtest_result = """
        ('{}', '{}', '{}', '{}', '{}', '{}')
        """.format(
            strategy,
            symbol_id,
            oos_equity_curves.index[0],
            oos_equity_curves.index[-1],
            performance_metrics,  
            strat.optimization_metric
        )

        self.__upsert_into_table(
            symbol_id = symbol_id,
            strat = strat,
            table = 'backtest_results',
            insert_str = insert_str_backtest_result
        )

        if len(oos_trades) > 0:
            self.__upsert_into_table(
                symbol_id = symbol_id,
                strat = strat,
                table = 'backtest_trades',
                insert_str = insert_str_backtest_trades
            )
            
        self.__upsert_into_table(
            symbol_id = symbol_id,
            strat = strat,
            table = 'backtest_equity_curves',
            insert_str = '',
            df = backtest_equity_curve_df
        )

    def __backtest_strategy(self, strat, tokens):
        if issubclass(type(strat), BaseStrategy):
            for token in tokens['symbol_id']:
                base, quote, exchange = token.split('_')
                print()
                print('Backtesting the {} strategy on {}'.format(strat.indicator_factory_dict['class_name'], token))
                print()

                # Execute walk-forward optimization of the strategy on the token
                oos_equity_curves, oos_trades, oos_price_data, performance_metrics = self.__execute_wfo(
                    base = base,
                    quote = quote,
                    exchange = exchange,
                    strat = strat,
                    in_sample_size = 30,
                    out_of_sample_size = 30
                )

                # Reset starting equity for next token
                strat.backtest_params['init_cash'] = 10_000

                # If the walk-forward optimization failed, skip to the next token
                if (oos_equity_curves is None) or (oos_trades is None) or (oos_price_data is None):
                    continue

                # Upload backtest results to DuckDB
                self.__upload_backtest_results(
                    strat = strat,
                    oos_equity_curves = oos_equity_curves,
                    oos_trades = oos_trades,
                    oos_price_data = oos_price_data,
                    performance_metrics = performance_metrics,
                    base = base,
                    quote = quote,
                    exchange = exchange
                )

        elif issubclass(type(strat), BasePortfolioStrategy):
            print()
            print('Backtesting the {} strategy on the token universe'.format(strat.indicator_factory_dict['class_name']))
            print()

            # Execute walk-forward optimization of the strategy on the universe
            oos_equity_curves, oos_trades, oos_price_data, performance_metrics = self.__execute_wfo(
                strat = strat,
                in_sample_size = 30,
                out_of_sample_size = 30
            )

            # If the walk-forward optimization failed, raise an error
            if (oos_equity_curves is None) or (oos_trades is None):
                raise ValueError(f'Walk-forward optimization failed for {strat.indicator_factory_dict["class_name"]}')

            # Upload backtest results to DuckDB
            self.__upload_backtest_results(
                strat = strat,
                oos_equity_curves = oos_equity_curves,
                oos_trades = oos_trades,
                oos_price_data = oos_price_data,
                performance_metrics = performance_metrics
            )

        else:
            raise ValueError(f'Invalid strategy class: {strat}')

    def execute(self):
        """
        Runs a walk-forward optimization on each (Strategy, token) combination, where token
        is a crypto token and Strategy is a trading strategy in backtest/strategies. Performance 
        metrics are then calculated on the combined out-of-sample equity curve, trade, and 
        price data output from the walk-forward optimization and logged for further 
        dashboarding/analysis.

        Parameters:
        -----------
        None

        Returns:
        --------
        None
        """
        # Fetch all tokens to backtest
        tokens = QUERY(
            """
            SELECT DISTINCT asset_id_base, asset_id_quote, exchange_id
            FROM market_data.ml_dataset
            ORDER BY asset_id_base, asset_id_quote, exchange_id
            """
        )
        tokens['symbol_id'] = tokens['asset_id_base'] + '_' + tokens['asset_id_quote'] + '_' + tokens['exchange_id']

        for strat in self.strategies:
            self.__backtest_strategy(strat, tokens)