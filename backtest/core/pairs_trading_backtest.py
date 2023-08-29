import pandas as pd
import numpy as np
import itertools

from numba_funcs import *
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from performance_metrics import calculate_performance_metrics

class PairsTradingBacktest:
    # BACKTESTING PARAMETERS

    # Window size to calculate the rolling z score
    # of the price spread
    z_window = 48

    # Window size to calculate the rollling hedge ratios 
    hedge_ratio_window = 24 * 7

    # Value of price spread rolling z score to initiate 
    # an entry/exit when crossed above
    z_score_upper_thresh = 2

    # Value of price spread rolling z score to initiate 
    # an entry/exit when crossed below
    z_score_lower_thresh = -2

    # Max amount of time in hours a position can be held
    max_holding_time = 48

    def __init__(self, 
                 price_data,
                 symbol_id_1,
                 symbol_id_2,
                 start_i,
                 end_i,
                 backtest_params = {
                    'initial_capital':10_000, 
                    'pct_capital_per_trade': 0.05,
                    'comission': 0.00295,
                    'sl': float('inf'),
                    'tp':float('inf'),
                    'max_slippage':0.00
                    }
                ):
        """
        price_data - DataFrame w/ two columns of price data named 
                     symbol_id_1, symbol_id_2) and indexed by timestamp

        symbol_id_1 - Token we're shorting in the backtest

        symbol_id_2 - Token we're longing in the backtest

        backtest_params - Dictionary of hyperparameters for the current backtest
        """

        self.symbol_id_1 = symbol_id_1
        self.symbol_id_2 = symbol_id_2

        self.start_i = start_i
        self.end_i = end_i

        self.start_date = pd.to_datetime(price_data.index[0])
        self.end_date = pd.to_datetime(price_data.index[-1])

        self.backtest_params = backtest_params
        self.curr_capital_arr = None

        # Fetch data required for backtest
        self.data = price_data

        self.backtest_data = self.data.iloc[self.start_i:self.end_i]
        
        # Backtest initialized w/ no position
        self.position = 0

        # Number of units long in our current position
        self.curr_position_long_units = 0

        # Number of units short in our current position
        self.curr_position_short_units = 0

        # Trades executed throughout the backtest
        self.trades = pd.DataFrame(columns = ['entry_date', 'exit_date', 'entry_i', 'exit_i', self.symbol_id_2, self.symbol_id_1, 'pnl', 'pnl_pct', 'is_long'])

        # PnL in dollars at each timestep
        self.pnl = None

        # Equity at each timestep
        self.equity = None
        
    ################################# HELPER METHODS #################################
    def __rolling_hedge_ratios(self):
        start = max(0, self.start_i - self.hedge_ratio_window)

        h = rolling_hedge_ratios_numba(
            y = self.data.iloc[start:self.end_i][self.symbol_id_2].values, 
            x = self.data.iloc[start:self.end_i][self.symbol_id_1].values,
            window = self.hedge_ratio_window
        )

        if start == 0:
            return h
        else:
            return h[self.hedge_ratio_window:]
            
    def __rolling_spread_z_score(self):
        return rolling_spread_z_score_numba(
            rolling_hedge_ratios = self.backtest_data['rolling_hedge_ratio'].values,
            y = self.backtest_data[self.symbol_id_2].values,
            x = self.backtest_data[self.symbol_id_1].values,
            window = self.z_window
        )
    
    def __generate_trading_signals(self):
        rolling_spread_z_score = self.backtest_data['rolling_spread_z_score'].values
        rolling_spread_z_score_prev = self.backtest_data['rolling_spread_z_score'].shift(1).values

        return generate_trading_signals_numba(
            z = rolling_spread_z_score,
            z_prev = rolling_spread_z_score_prev,
            zl = self.z_score_lower_thresh,
            zu = self.z_score_upper_thresh
        )
        
    def __generate_positions(self):
        # Long and short positions throughout backtest
        positions = pd.DataFrame(index = self.backtest_data.index, columns = [self.symbol_id_2, self.symbol_id_1]).fillna(0)
        
        trades, curr_capital_arr = generate_positions_numba(
            positions = positions.values.astype(float),
            data = self.backtest_data.values.astype(float),
            dates = self.backtest_data.index.values.astype(float),
            trades = self.trades.values.astype(float),
            position = self.position,
            curr_capital = self.backtest_params['initial_capital'],
            pct_capital_per_trade = self.backtest_params['pct_capital_per_trade'],
            commission = self.backtest_params['comission'],
            slippage = self.backtest_params['max_slippage'],
            sl = self.backtest_params['sl'],
            tp = self.backtest_params['tp'],
            max_holding_time = self.max_holding_time
        )

        self.trades = pd.DataFrame(data = trades, columns = self.trades.columns)
        self.trades['entry_date'] = pd.to_datetime(self.trades['entry_date'])
        self.trades['exit_date'] = pd.to_datetime(self.trades['exit_date'])
        
        curr_capital_arr = pd.DataFrame(data = curr_capital_arr.flatten(), index = self.backtest_data.index, columns = ['curr_capital'])
        curr_capital_arr['curr_capital'] = curr_capital_arr['curr_capital'].fillna(method = 'ffill')

        self.curr_capital_arr = curr_capital_arr
                        
    def __generate_pnl(self):
        pnl_data = pd.DataFrame(index = self.backtest_data.index, columns = ['pnl']).fillna(0)
        max_slippage = self.backtest_params['max_slippage']
        
        for trade in self.trades.to_dict(orient = 'records'):
            entry_date = trade['entry_date']
            exit_date = trade['exit_date']
            
            long_symbol = self.symbol_id_2 if trade['is_long'] else self.symbol_id_1
            short_symbol = self.symbol_id_1 if trade['is_long'] else self.symbol_id_2
            
            units_long = trade[long_symbol]
            units_short = trade[short_symbol]
            
            trade_period = self.backtest_data.loc[entry_date:exit_date].copy()
            
            trade_period.at[entry_date, long_symbol] = trade_period.at[entry_date, long_symbol] * (1 + max_slippage)
            trade_period.at[exit_date, long_symbol] = trade_period.at[exit_date, long_symbol] * (1 - max_slippage)
            trade_period.at[entry_date, short_symbol] = trade_period.at[entry_date, short_symbol] * (1 - max_slippage)
            trade_period.at[exit_date, short_symbol] = trade_period.at[exit_date, short_symbol] * (1 + max_slippage)
            
            trade_period['long_pnl'] = trade_period[long_symbol].diff() * units_long
            trade_period['short_pnl'] = -trade_period[short_symbol].diff() * units_short
            trade_period['total_pnl'] = trade_period['long_pnl'] + trade_period['short_pnl']

            for date in trade_period.index:
                pnl_data.at[date, 'pnl'] = trade_period.at[date, 'total_pnl']
                
        return pnl_data.fillna(0)

    def __generate_equity_curve(self):
        equity = pd.DataFrame(
            data = (self.curr_capital_arr.values + self.pnl.cumsum().values),
            index = self.backtest_data.index,
            columns = ['equity']
        )

        return equity

    ################################# HELPER METHODS #################################
    
    def optimize_parameters(self, optimize_dict, performance_metric = 'Sharpe Ratio', minimize = False):
        """
        Optimize strategy over every combination of parameters given in
        
        optimize_dict and sets the class strategy parameters to the values that
        
        maximize/minimize the requested performance metric.        
        """
        
        lists = []
        parameter_combinations = []

        best_comb_so_far = [
            self.z_window, 
            self.hedge_ratio_window,
            self.z_score_upper_thresh, 
            self.z_score_lower_thresh, 
            self.max_holding_time
        ]

        best_metric_so_far = float('inf') if minimize else float('-inf')
        
        for key in optimize_dict.keys():
            lists.append(optimize_dict[key])

        parameter_combinations = list(itertools.product(*lists))
        
        for i in range(len(parameter_combinations)):
            
            print('{}/{}'.format(i + 1, len(parameter_combinations)), end = '\r', flush = True)            

            self.z_window = parameter_combinations[i][0]
            self.hedge_ratio_window = parameter_combinations[i][1]
            self.z_score_upper_thresh = parameter_combinations[i][2]
            self.z_score_lower_thresh = parameter_combinations[i][3]
            self.max_holding_time = parameter_combinations[i][4]

            self.backtest()

            backtest_result_metric = float(self.performance_metrics[performance_metric][0])

            if not minimize and backtest_result_metric > best_metric_so_far:
                best_comb_so_far = parameter_combinations[i]
                best_metric_so_far = backtest_result_metric

            elif minimize and backtest_result_metric < best_metric_so_far:
                best_comb_so_far = parameter_combinations[i]
                best_metric_so_far = backtest_result_metric

        self.z_window = best_comb_so_far[0]
        self.hedge_ratio_window = best_comb_so_far[1]
        self.z_score_upper_thresh = best_comb_so_far[2]
        self.z_score_lower_thresh = best_comb_so_far[3]
        self.max_holding_time = best_comb_so_far[4]

        self.backtest()

        optimal_params = {
            'z_window': self.z_window, 
            'hedge_ratio_window': self.hedge_ratio_window,
            'z_score_upper_thresh': self.z_score_upper_thresh, 
            'z_score_lower_thresh': self.z_score_lower_thresh,
            'max_holding_time': self.max_holding_time
        }
        
        return optimal_params
    
    def backtest(self):   
        # Calculate rolling hedge ratios at each timestep
        self.backtest_data['rolling_hedge_ratio'] = self.__rolling_hedge_ratios()
                
        # Calculate rolling spread z score at each timestep
        self.backtest_data['rolling_spread_z_score'] = self.__rolling_spread_z_score()

        # Calculate entry and exit signals at each timestep
        entry_signals, exit_signals = self.__generate_trading_signals()

        self.backtest_data['entry_signals'] = entry_signals
        self.backtest_data['exit_signals'] = exit_signals
        
        # Reset trades DataFrame to make sure it's empty
        self.trades = pd.DataFrame(columns = ['entry_date', 'exit_date', 'entry_i', 'exit_i', self.symbol_id_2, self.symbol_id_1, 'pnl', 'pnl_pct', 'is_long'])
        
        # Calculate long and short positions at each timestep
        self.__generate_positions()

        # Calculate PnL in dollars at each timestep
        self.pnl = self.__generate_pnl()

        # Calculate equity at each timestep
        self.equity = self.__generate_equity_curve()

        # Calculate performance metrics for backtest
        self.performance_metrics = calculate_performance_metrics(
            self.equity,
            self.trades,
            self.backtest_data[[self.backtest_data.columns[0], self.backtest_data.columns[1]]]
        )

        # Reset position variable back to default state for subsequent backtests
        self.position = 0

        # Reset initial_capital parameter back to default state for subsequent backtests
        self.backtest_params['initial_capital'] = self.curr_capital_arr['curr_capital'].iloc[0]